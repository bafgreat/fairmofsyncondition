import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GINEConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
    AttentionalAggregation,
)

GLOBAL_DIMS = {
    "lattice": 9,                 # [3, 3] -> 9
    "atomic_one_hot": 120,
    "cordinates": 1,              # lo tengo opzionale, visto che è nel tuo Data
    "space_group_number": 231,
    "crystal_system": 7,
    "oms": 1,
}

SYNTHESIS_FIELDS = {
    "metal_salt",
    "modulator",
    "solvent",
    "synthetic_method",
    "reaction_time",
    "reaction_temperature",
}


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        dropout=0.1,
        batch_norm=True,
    ):
        super().__init__()

        layers = []
        dim = in_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))

            if batch_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers += [
                nn.SiLU(),
                nn.Dropout(dropout),
            ]

            dim = hidden_dim

        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_dim=4,
        edge_dim=1,
        hidden_dim=256,
        num_layers=5,
        dropout=0.15,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=hidden_dim,
                    train_eps=True,
                )
            )
            self.norms.append(GraphNorm(hidden_dim))

        self.attn_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        self.dropout = dropout
        self.out_dim = hidden_dim * 3

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_encoder(x)

        if edge_attr is None:
            edge_attr = torch.ones(
                edge_index.size(1),
                1,
                device=x.device,
                dtype=x.dtype,
            )

        e = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h

            h = conv(h, edge_index, e)
            h = norm(h, batch)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            h = h + h_res

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_attn = self.attn_pool(h, batch)

        return torch.cat([h_mean, h_max, h_attn], dim=-1)


class GlobalFeatureEncoder(nn.Module):
    def __init__(
        self,
        fields,
        hidden_dim=128,
        out_dim=128,
        dropout=0.1
    ):
        super().__init__()

        for field in fields:
            if field in SYNTHESIS_FIELDS:
                raise ValueError(f"Non puoi usare {field} come input: è una condizione di sintesi.")

            if field not in GLOBAL_DIMS:
                raise ValueError(f"Campo globale sconosciuto: {field}")

        self.fields = fields

        self.encoders = nn.ModuleDict({
            field: MLP(
                in_dim=GLOBAL_DIMS[field],
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_layers=2,
                dropout=dropout,
            )
            for field in fields
        })

        self.out_dim = len(fields) * out_dim

    def _get_feature(self, data, field, batch_size):
        x = getattr(data, field)

        x = x.float()
        x = x.reshape(batch_size, -1)

        expected_dim = GLOBAL_DIMS[field]
        if x.size(1) != expected_dim:
            raise ValueError(
                f"{field}: dimensione attesa {expected_dim}, trovata {x.size(1)}. "
                f"Shape originale probabilmente non batchata come previsto."
            )

        return x

    def forward(self, data):
        batch_size = int(data.batch.max().item()) + 1

        encoded = []

        for field in self.fields:
            x = self._get_feature(data, field, batch_size)
            encoded.append(self.encoders[field](x))

        if len(encoded) == 0:
            return None

        return torch.cat(encoded, dim=-1)


class MOFConditionPredictor(nn.Module):
    def __init__(
        self,
        target_dim,
        global_fields=None,
        node_dim=4,
        edge_dim=1,
        gnn_hidden_dim=256,
        gnn_layers=5,
        global_hidden_dim=128,
        global_out_dim=128,
        fusion_hidden_dim=512,
        dropout=0.15,
    ):
        super().__init__()

        if global_fields is None:
            global_fields = []

        self.graph_encoder = GNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        self.global_encoder = GlobalFeatureEncoder(
            fields=global_fields,
            hidden_dim=global_hidden_dim,
            out_dim=global_out_dim,
            dropout=dropout,
        )

        fusion_dim = self.graph_encoder.out_dim + self.global_encoder.out_dim

        self.fusion = MLP(
            in_dim=fusion_dim,
            hidden_dim=fusion_hidden_dim,
            out_dim=fusion_hidden_dim,
            num_layers=3,
            dropout=dropout,
        )

        self.head = nn.Linear(fusion_hidden_dim, target_dim)

    def forward(self, data):
        graph_emb = self.graph_encoder(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.float() if data.edge_attr is not None else None,
            batch=data.batch,
        )

        global_emb = self.global_encoder(data)

        if global_emb is not None:
            emb = torch.cat([graph_emb, global_emb], dim=-1)
        else:
            emb = graph_emb

        emb = self.fusion(emb)

        logits = self.head(emb)

        return logits