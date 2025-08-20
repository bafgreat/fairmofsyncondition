import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool



class MetalSaltGNN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        lattice_in_dim,
        hidden_dim,
        num_classes,
        num_gnn_layers=4,
        num_lattice_layers=2,
        num_mlp_layers=2,
        dropout=0.2,
        use_batchnorm=True
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GINE layers: ora usiamo GINEConv
        self.gnn_layers = nn.ModuleList()
        self.gnn_bns = nn.ModuleList() if use_batchnorm else None

        for i in range(num_gnn_layers):
            in_dim = node_in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gnn_layers.append(GINEConv(mlp, edge_dim=hidden_dim))  # <-- aggiunto edge_dim!
            if use_batchnorm:
                self.gnn_bns.append(nn.BatchNorm1d(hidden_dim))

        # Lattice encoder come MLP profondo parametrico
        lattice_layers = []
        in_dim = lattice_in_dim
        for _ in range(num_lattice_layers - 1):
            lattice_layers.append(nn.Linear(in_dim, hidden_dim))
            lattice_layers.append(nn.ReLU())
            if use_batchnorm:
                lattice_layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        lattice_layers.append(nn.Linear(in_dim, hidden_dim))
        self.lattice_encoder = nn.Sequential(*lattice_layers)

        # Final MLP layers
        mlp_layers = []
        in_dim = hidden_dim * 2  # x_pool + lattice_feat
        for _ in range(num_mlp_layers - 1):
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            if use_batchnorm:
                mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.final_mlp = nn.Sequential(*mlp_layers)

    def forward(self, data):
        x, edge_index, edge_attr, batch, lattice = (
            data.x, data.edge_index, data.edge_attr, data.batch, data.lattice
        )

        # Encode edge_attr
        edge_feat = self.edge_encoder(edge_attr)

        # GINE layers
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index, edge_feat)  # ora usiamo edge_feat nella propagazione!
            x = F.relu(x)
            if self.use_batchnorm:
                x = self.gnn_bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling globale
        x_pool = global_mean_pool(x, batch)

        # Lattice processing
        lattice_flat = lattice.reshape(-1, 3 * 3)  # batch_size x 9
        lattice_feat = self.lattice_encoder(lattice_flat)

        # Combine and classify
        out = torch.cat([x_pool, lattice_feat], dim=1)
        out = self.final_mlp(out)

        return out
