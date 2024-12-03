
#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

##############################################################################
# fairmofsyncondition is a machine learning package for predicting the        #
# synthesis condition of the crystal structures of MOFs. It is also intended  #
# for predicting all MOFs the can be generated from a given set of conditions #
# In addition the package also predicts the stability of MOFs, compute their  #
# their PXRD and crystallite sizes. This package is part of our effort to     #
# to accelerate the discovery and optimization of the synthesises of novel    #
# high performing MOFs. This package is being developed by Dr Dinga Wonanke   #
# as part of hos MSCA post doctoral fellowship at TU Dresden.                 #
#                                                                             #
###############################################################################

import os
import torch
import random
from torch import nn
from torch.nn import functional
from ase.db import connect
from torch import optim
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from fairmofsyncondition.read_write import coords_library, filetyper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EnergyGNN(nn.Module):
    """
    A Graph Neural Network class for predicting the thermodynamic
    stability of MOFs using Graph Attention Networks (GATv2).

    Arg:
        input_dim (int): Number of input node features.
        hidden_dim (int): Number of hidden units in the GATv2 layers.
        output_dim (int): Number of output units (e.g., 1 for regression).
        heads (int, optional): Number of attention heads. Default is 1.
        dropout (float, optional): Dropout rate. Default is 0.2.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, heads=1, dropout=0.2):
        super(EnergyGNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim * heads)
        self.fc1 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        """
        Forward pass of the model.

        **parameters:**
            data (torch_geometric.data.Data): Input graph data

        **returns:**
            torch.Tensor: Predicted energy value.
        """
        # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = functional.relu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = functional.relu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = global_mean_pool(x, batch)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def prepare_dataset(ase_obj, energy):
    """
    Prepares a dataset from ASE Atoms objects and their corresponding energy values.

    **parameters:**
        ase_obj (ASE.Atoms): ASE Atoms object.
        energy (float): Energy value of the crystal structure.

    **returns:**
        torch_geometric.data.Data: PyTorch Geometric Data object with input features, edge indices, and energy value.
    """
    data = coords_library.ase_to_pytorch_geometric(ase_obj)
    data.y = torch.tensor([energy], dtype=torch.float)
    return data



def data_from_aseDb(path_to_db):
    """
    Load data from ASE database and prepare it for training.

    **parameters:**
        path_to_db (str): Path to the ASE database file.

    **returns:**
        list: List of PyTorch Geometric Data objects for training.
    """
    dataset = []
    counter = 0
    db = connect(path_to_db)
    for row in db.select():
        data = prepare_dataset(row.toatoms(), row.r_energy)
        dataset.append(data)
        if counter >= 5000:
            break
        counter += 1
    return dataset


def train(model, dataloader, optimizer, criterion, device):
    """
    Train the model using the given data and optimizer.

    **parameters:**
        model (nn.Module): The GNN model to train
        dataloader (DataLoader): DataLoader for batching the dataset during training.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function (e.g., MSELoss) to compute the training loss.
        device (torch.device): The device (CPU or GPU) for computation.

    **returns:**
        float: The average training loss over the epoch.
    """

    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)



def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model using the given data and loss function.

    **parameters:**
        model (nn.Module): The trained GNN model to evaluate.
        dataloader (DataLoader): DataLoader for batching the dataset during evaluation.
        criterion (nn.Module): Loss function (e.g., MSELoss) to compute the evaluation loss.
        device (torch.device): The device (CPU or GPU) for computation.

    **returns:**
        float: The average evaluation loss over the epoch.
    """

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            predictions = model(data)
            loss = criterion(predictions, data.y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def save_model(model, optimizer, path="model.pth"):
    """
    Save the trained model to a file.

    **parameters:**
        model (nn.Module): The trained GNN model to save.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        path (str, optional): Path to save the model. Default is "model.pth".
    """

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def load_model(model, optimizer, path="model.pth", device="cpu"):
    """
    Load a saved model from a file.

    **parameters:**
        model (nn.Module): The model to load.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        path (str, optional): Path to load the model. Default is "model.pth".
        device (torch.device, optional): The device (CPU or GPU) for computation. Default is "cpu".
    """

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


if __name__ == "__main__":
    dataset = data_from_aseDb('../../data/MOF_stability.db')
    random.shuffle(dataset)
    train_data = dataset[:int(0.8 * len(dataset))]
    val_data = dataset[int(0.8 * len(dataset)):]

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=100, shuffle=False)
    model = EnergyGNN(input_dim=4, hidden_dim=128,edge_dim=1,  output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    epoch = 5000
    for i in range(epoch):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch: {i+1}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        save_model(model, optimizer, path="energy_gnn.pth")

    print (model)