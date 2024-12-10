
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
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, heads=2, dropout=0.2277220314204774):
        super(EnergyGNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim * heads)
        self.fc1 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = functional.leaky_relu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = functional.leaky_relu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = functional.leaky_relu(self.norm3(self.conv3(x, edge_index, edge_attr)))
        x = global_mean_pool(x, batch)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
        predictions = model(data).view(-1)
        loss = criterion(predictions, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model using the given data and loss function, and compute accuracy.

    **parameters:**
        model (nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for batching the dataset during evaluation.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss) to compute the evaluation loss.
        device (torch.device): The device (CPU or GPU) for computation.

    **returns:**
        tuple: A tuple containing the average evaluation loss and accuracy.
               (average_loss, accuracy)
    """

    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            labels = data.y
            predictions = model(data)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            _, predicted_classes = predictions.max(dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


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


def main():
    hidden_dim = 124
    learning_rate = 0.009687380458236276
    batch_size = 21
    dropout = 0.2277220314204774


    dataset = data_from_aseDb('../../data/mof_ligand_stability.db')
    random.shuffle(dataset)
    train_data = dataset[:int(0.8 * len(dataset))]
    val_data = dataset[int(0.8 * len(dataset)):]

    train_loader = DataLoader(train_data, batch_size=21, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=21, shuffle=False)
    model = EnergyGNN(input_dim=4, hidden_dim=124, edge_dim=1, output_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.009687380458236276)
    criterion = nn.MSELoss()
    epoch = 20
    for i in range(epoch):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch: {i+1}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        save_model(model, optimizer, path="test.pth")

# if __name__ == "__main__":
#     main()

# model = EnergyGNN(input_dim=4, hidden_dim=128,edge_dim=1,  output_dim=1).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model, optimizer = load_model(model,optimizer,  'test.pth')

# data = coords_library.ase_to_pytorch_geometric('../../../../../FAIRDATA/MOF_Data/GFN_CIFs/ABAVIJ.cif')

# predictions = model(data).detach().numpy()
# print (predictions)