import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib.resources import files
import numpy as np
from torch_geometric.nn import GINEConv, global_mean_pool
from sklearn.metrics import f1_score
from mofstructure.structure import MOFstructure
from tqdm import tqdm
import os
from mofstructure import mofdeconstructor
from pymatgen.io.ase import AseAtomsAdaptor
from fairmofsyncondition.read_write import filetyper
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from fairmofsyncondition.read_write.coords_library import pytorch_geometric_to_ase
import pickle


convert_struct = {'cubic': 0,
                  'hexagonal': 1,
                  'monoclinic': 2,
                  'orthorhombic': 3,
                  'tetragonal': 4,
                  'triclinic': 5,
                  'trigonal': 6
                  }

convert_metals = {j:i for i,j in enumerate(mofdeconstructor.transition_metals()[1:])}

def _reshape_feat(tensor, d):
    # Se è Batch (ha num_graphs)
    if hasattr(d, "num_graphs"):
        return tensor.view(d.num_graphs, -1)
    else:  # è un singolo Data
        return tensor.view(1, -1)

EXTRA_GETTERS = {
    "atomic_one_hot":      lambda d: _reshape_feat(d.atomic_one_hot, d),
    "space_group_number":  lambda d: _reshape_feat(d.space_group_number, d),
    "crystal_system":      lambda d: _reshape_feat(d.crystal_system, d),
    "oms":                 lambda d: _reshape_feat(d.oms, d),
    "cordinates":          lambda d: _reshape_feat(d.cordinates, d),
}
def compute_extras_dim(sample_data, selected_extras):
    dim = 0
    for name in selected_extras:
        if name not in EXTRA_GETTERS:
            raise ValueError(f"Feature extra sconosciuta: {name}")
        dim += EXTRA_GETTERS[name](sample_data).shape[1]
    return dim

def build_extras_tensor(data, selected_extras):
    if not selected_extras:
        return None
    parts = [EXTRA_GETTERS[name](data) for name in selected_extras]
    return torch.cat(parts, dim=1)

def extras_suffix(selected_extras):
    if not selected_extras:
        return "no_extras"
    return "_".join(selected_extras)

# ==================== Model ====================

class MetalSaltGNN_Energy(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        lattice_in_dim=9,
        hidden_dim=128,
        num_gnn_layers=4,
        num_lattice_layers=2,
        num_mlp_layers=2,
        dropout=0.2,
        use_batchnorm=True,
        selected_extras=None,
        extras_dim=0
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.selected_extras = selected_extras or []
        self.extras_dim = extras_dim

        # Edge encoder (-> hidden_dim) per usare edge_dim=hidden_dim in GINE
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GINE stack
        self.gnn_layers = nn.ModuleList()
        self.gnn_bns = nn.ModuleList() if use_batchnorm else None
        for i in range(num_gnn_layers):
            in_dim = node_in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gnn_layers.append(GINEConv(mlp, edge_dim=hidden_dim))
            if use_batchnorm:
                self.gnn_bns.append(nn.BatchNorm1d(hidden_dim))

        # Lattice encoder
        lattice_layers = []
        in_dim = lattice_in_dim
        for _ in range(max(1, num_lattice_layers - 1)):
            lattice_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if use_batchnorm:
                lattice_layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        lattice_layers.append(nn.Linear(in_dim, hidden_dim))
        self.lattice_encoder = nn.Sequential(*lattice_layers)

        # Final MLP head -> output 1 per regressione
        final_in = hidden_dim * 2 + self.extras_dim
        mlp_layers = []
        in_dim = final_in
        for _ in range(max(1, num_mlp_layers - 1)):
            mlp_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if use_batchnorm:
                mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.final_mlp = nn.Sequential(*mlp_layers)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        e = self.edge_encoder(edge_attr)
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index, e)
            x = F.relu(x)
            if self.use_batchnorm:
                x = self.gnn_bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph pooling
        x_pool = global_mean_pool(x, batch)

        # Lattice
        lattice = data.lattice.view(-1, 9)
        lattice_feat = self.lattice_encoder(lattice)

        # Extras
        extras = build_extras_tensor(data, self.selected_extras)
        if extras is not None:
            final_in = torch.cat([x_pool, lattice_feat, extras], dim=1)
        else:
            final_in = torch.cat([x_pool, lattice_feat], dim=1)

        out = self.final_mlp(final_in)     # [B, 1]
        return out.squeeze(-1)             # [B]

class MetalSaltGNN_Ablation(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        lattice_in_dim=9,
        hidden_dim=128,
        num_classes=10,
        num_gnn_layers=4,
        num_lattice_layers=2,
        num_mlp_layers=2,
        dropout=0.2,
        use_batchnorm=True,
        selected_extras=None,      # NEW: lista di nomi feature extra
        extras_dim=0               # NEW: dimensione totale delle extra
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.selected_extras = selected_extras or []
        self.extras_dim = extras_dim

        # --- Edge encoder (per GINE)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # --- GINE layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_bns = nn.ModuleList() if use_batchnorm else None
        for i in range(num_gnn_layers):
            in_dim = node_in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gnn_layers.append(GINEConv(mlp, edge_dim=hidden_dim))
            if use_batchnorm:
                self.gnn_bns.append(nn.BatchNorm1d(hidden_dim))

        # --- Lattice encoder
        lattice_layers = []
        in_dim = lattice_in_dim
        for _ in range(max(1, num_lattice_layers - 1)):
            lattice_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if use_batchnorm:
                lattice_layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        lattice_layers.append(nn.Linear(in_dim, hidden_dim))
        self.lattice_encoder = nn.Sequential(*lattice_layers)

        # --- Final MLP head
        final_in = hidden_dim * 2 + self.extras_dim  # graph pooled + lattice + extras
        mlp_layers = []
        in_dim = final_in
        for _ in range(max(1, num_mlp_layers - 1)):
            mlp_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if use_batchnorm:
                mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.final_mlp = nn.Sequential(*mlp_layers)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encode edges
        e = self.edge_encoder(edge_attr)

        # GNN layers
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index, e)
            x = F.relu(x)
            if self.use_batchnorm:
                x = self.gnn_bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x_pool = global_mean_pool(x, batch)

        # Lattice encoding (sempre usato)
        lattice = data.lattice.view(-1, 9)
        lattice_feat = self.lattice_encoder(lattice)

        # Extras (abilitate in base alla lista)
        extras = build_extras_tensor(data, self.selected_extras)
        if extras is not None:
            final_in = torch.cat([x_pool, lattice_feat, extras], dim=1)
        else:
            final_in = torch.cat([x_pool, lattice_feat], dim=1)

        out = self.final_mlp(final_in)
        return out


def get_model_energy(torch_data,device="cpu"):
    node_in_dim = torch_data.x.shape[1]
    edge_in_dim = torch_data.edge_attr.shape[1]
    lattice_in_dim = 9

    EXTRA_GETTERS = {
        "atomic_one_hot":      lambda d: _reshape_feat(d.atomic_one_hot, d),
        "space_group_number":  lambda d: _reshape_feat(d.space_group_number, d),
        "crystal_system":      lambda d: _reshape_feat(d.crystal_system, d),
        "oms":                 lambda d: _reshape_feat(d.oms, d),
        "cordinates":          lambda d: _reshape_feat(d.cordinates, d),
    }

    selected_extras = ["atomic_one_hot", "cordinates", "oms","space_group_number"]
    selected_extras = np.sort(selected_extras).tolist()
    # Calcolo dinamico della dimensione delle extra
    extras_dim = 448 #compute_extras_dim(torch_data, selected_extras)
    dropout = 0.35


    model = MetalSaltGNN_Energy(
        node_in_dim=4,
        edge_in_dim=1,
        lattice_in_dim=9,
        hidden_dim=128,
        num_gnn_layers=2,
        num_lattice_layers=2,
        num_mlp_layers=2,
        dropout=dropout,
        use_batchnorm=True,
        selected_extras=selected_extras,
        extras_dim=extras_dim
    ).to(device)

    # checkpoint_path = f"trained_models/energy_regression.pt"
    checkpoint_path = f"trained_models/energy_regression.pt"
    checkpoint_name = files("fairmofsyncondition").joinpath("call_model", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_name, map_location=device))

    return model

def get_models(torch_data, device="cpu"):

    models = []

    for seed in [1]:
        node_in_dim = torch_data.x.shape[1]
        edge_in_dim = torch_data.edge_attr.shape[1]
        lattice_in_dim = 9

        # Scegli qui le extra da usare nell’ablation:
        # Esempio richiesto: ["oms", "atomic_one_hot"]


        EXTRA_GETTERS = {
            "atomic_one_hot":      lambda d: _reshape_feat(d.atomic_one_hot, d),
            "space_group_number":  lambda d: _reshape_feat(d.space_group_number, d),
            "crystal_system":      lambda d: _reshape_feat(d.crystal_system, d),
            "oms":                 lambda d: _reshape_feat(d.oms, d),
            "cordinates":          lambda d: _reshape_feat(d.cordinates, d),
        }

        selected_extras = ["atomic_one_hot", "cordinates", "oms","space_group_number"]
        selected_extras = np.sort(selected_extras).tolist()
        # Calcolo dinamico della dimensione delle extra
        extras_dim = 448 #compute_extras_dim(torch_data, selected_extras)

        # Classi
        Y_size = 791
        num_classes = Y_size + 1
        hidden_dim = 64
        dropout = 0.35

        suffix = extras_suffix(selected_extras)
        model = MetalSaltGNN_Ablation(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            lattice_in_dim=lattice_in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_gnn_layers=4,
            num_lattice_layers=2,
            num_mlp_layers=2,
            dropout=dropout,
            use_batchnorm=True,
            selected_extras=selected_extras,
            extras_dim=extras_dim
        ).to(device)

        config_name = f"HID{hidden_dim}_DO{dropout}_SEED{seed}__{suffix}"
        checkpoint_path = f"trained_models/Metal_salts_{config_name}_tmp_test.pt"
        checkpoint_name = files("fairmofsyncondition").joinpath("call_model",checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))
        models.append(model)
    return models

def get_energy_prediction(energy_model,torch_data,device="cpu"):
    torch_data = torch_data.to(device)
    energy_model = energy_model.to(device)
    energy_model.eval()
    with torch.no_grad():
        pre_energy = energy_model(torch_data).item()

    scaler_path = files("fairmofsyncondition").joinpath("call_model","trained_models/target_scaler_energy.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    predicte_energy_scaled   = scaler.inverse_transform(np.array(pre_energy).reshape(-1, 1)).ravel()

    return predicte_energy_scaled

def ensemble_predictions(models, torch_data, category_names, device="cpu"):
    """
    Soft voting ensemble: ritorna le probabilità medie per TUTTE le classi.
    """
    all_probs = []
    torch_data = torch_data.to(device)

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(torch_data)
            probs = F.softmax(logits, dim=-1)  # [batch, num_classes]
            all_probs.append(probs)

    # Media delle probabilità sui modelli
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)  # [batch, num_classes]

    # Converte in lista ordinata di tuple (classe, probabilità)
    full_list = [
        (category_names[i], avg_probs[0, i].item())
        for i in range(avg_probs.shape[1])
    ]

    # Ordina per probabilità decrescente
    full_list.sort(key=lambda x: x[1], reverse=True)

    return full_list

