
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
import sys
import torch
import random
import argparse
from torch import nn
from torch.nn import functional
from torch import optim
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
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

    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, heads=4, dropout=0.2):
        super(EnergyGNN, self).__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)
        edge_dim = int(edge_dim)
        heads = int(heads)
        self.conv1 = GATv2Conv(input_dim, hidden_dim,
                               heads=heads, edge_dim=edge_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATv2Conv(
            hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv3 = GATv2Conv(
            hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim * heads)
        self.fc1 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = functional.leaky_relu(self.norm1(
            self.conv1(x, edge_index, edge_attr)))
        x = functional.leaky_relu(self.norm2(
            self.conv2(x, edge_index, edge_attr)))
        x = functional.leaky_relu(self.norm3(
            self.conv3(x, edge_index, edge_attr)))
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



def normalize_data(dataset, method='z-score'):
    """
    Normalize the target values (data.y) in the dataset.

    Parameters:
        dataset (Dataset): The dataset object containing the data.
        method (str): The normalization method ('min-max' or 'z-score').

    Returns:
        tuple: Normalized dataset and the normalization parameters.
    """
    y_values = torch.cat([data.y for data in dataset])  # Gather all target values
    if method == 'min-max':
        y_min, y_max = y_values.min(), y_values.max()
        for data in dataset:
            data.y = (data.y - y_min) / (y_max - y_min)
        return dataset, {'min': y_min, 'max': y_max}

    elif method == 'z-score':
        y_mean, y_std = y_values.mean(), y_values.std()
        for data in dataset:
            data.y = (data.y - y_mean) / y_std
        return dataset, {'mean': y_mean, 'std': y_std}

    else:
        raise ValueError("Unsupported normalization method. Choose 'min-max' or 'z-score'.")


def inverse_normalize(predictions, normalization_params, method='z-score'):
    """
    Inverse the normalization of predictions to the original scale.

    Parameters:
        predictions (Tensor): The normalized predictions.
        normalization_params (dict): The parameters used for normalization.
        method (str): The normalization method ('min-max' or 'z-score').

    Returns:
        Tensor: Predictions in the original scale.
    """
    if method == 'min-max':
        return predictions * (normalization_params['max'] - normalization_params['min']) + normalization_params['min']
    elif method == 'z-score':
        return predictions * normalization_params['std'] + normalization_params['mean']
    else:
        raise ValueError("Unsupported normalization method. Choose 'min-max' or z-score")

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
            predictions = model(data).view(-1)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)

    return average_loss


def save_model(model, optimizer, normalise_parameter, path="model.pth"):
    """
    Save the trained model and optimizer state to a file.

    **parameters:**
        model (nn.Module): The trained GNN model to save.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        path (str, optional): Path to save the model. Default is "model.pth".
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_architecture': model.__class__,
        'model_args': model.__dict__.get('_modules'),
        'normalise_parameter': normalise_parameter
    }
    torch.save(checkpoint, path)


def load_model(path="model.pth", device="cpu"):
    """
    Load a saved model and optimizer state from a file.

    **parameters:**
        path (str, optional): Path to load the model. Default is "model.pth".
        device (torch.device, optional): The device (CPU or GPU) for computation. Default is "cpu".

    **returns:**
        tuple: The loaded model and optimizer.
    """
    checkpoint = torch.load(path, map_location=device)
    model_class = checkpoint['model_architecture']
    model_args = checkpoint['model_args']
    normalise_parameter = checkpoint['normalise_parameter']
    model = model_class(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters())  # Rebuild the optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, normalise_parameter

def transform_target(test_data, normalize_param, method='z-score'):
    """
    Transform the target values in the test dataset according to the normalization parameters.

    Parameters:
        test_data (Dataset): The test dataset containing the target values.
        normalize_param (dict): The normalization parameters (mean and std).
        method (str, optional): The normalization method ('z-score' or'min-max'). Default is 'z-score'.

    Returns:
        Dataset: The transformed test dataset.
    """
    if method == 'z-score':
        mean, std = normalize_param['mean'], normalize_param['std']
        for data in test_data:
            data.y = (data.y - mean) / std
    elif method =='min-max':
        min_val, max_val = normalize_param['min'], normalize_param['max']
        for data in test_data:
            data.y = (data.y - min_val) / (max_val - min_val)
    else:
        raise ValueError("Unsupported normalization method. Choose 'z-score' or'min-max'.")
    return test_data


def load_dataset(path_to_lmdb, batch_size, train_size=0.9, random_seed=42, shuffle=True, normalize='full'):
    """
    Loads a dataset from an LMDB file and splits it into training, validation, and test sets.

    The function uses the `coords_library.LMDBDataset` to load the dataset and splits it into
    training and test datasets. The training dataset is further split into training and validation
    sets. Data loaders are created for the training and validation datasets.

    Parameters:
        path_to_lmdb (str):
            Path to the LMDB file containing the dataset.
        batch_size (int):
            Batch size for the DataLoader.
        train_size (float, optional):
            Fraction of the data to use for training. The rest is used for testing. Default is 0.8.
        random_seed (int, optional):
            Random seed for splitting the data. Ensures reproducibility. Default is 42.
        shuffle (bool, optional):
            Whether to shuffle the data before splitting. Default is True.
        normalize (str, optional):
            Normalization method to use. Can be 'full' for full normalization or 'batch' for
            batch normalization. Default is 'full'.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - test_dataset (Dataset): Dataset object containing the test data.
    """
    dataset = coords_library.LMDBDataset(path_to_lmdb)
    train_dataset, test_dataset = dataset.split_data(
        train_size=train_size, random_seed=random_seed, shuffle=shuffle)

    train_indices, val_indices = coords_library.list_train_test_split(list(range(len(train_dataset))))
    train_data = train_dataset[train_indices]
    val_data = train_dataset[val_indices]
    if normalize == 'full':
        train_data_norm, normalise_parameter = normalize_data(train_data)
        train_loader = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True)
        val_data = transform_target(val_data, normalise_parameter)
        test_dataset = transform_target(test_dataset, normalise_parameter)
    elif normalize == 'batch':
        train_loader = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True)
        train_loader, normalise_parameter = normalize_data(train_loader)
        val_data = transform_target(val_data, normalise_parameter)
        test_dataset = transform_target(test_dataset, normalise_parameter)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_dataset, normalise_parameter


def main_model(path_to_lmdb, hidden_dim, learning_rate, batch_size, dropout, heads, epoch, save_path):
    writer = SummaryWriter(log_dir="errorlogger/energy_gat")
    model = EnergyGNN(input_dim=4, hidden_dim=hidden_dim,
                      output_dim=1, edge_dim=1, heads=heads, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loader, val_loader, test_dataset, normalise_parameter = load_dataset(path_to_lmdb, batch_size)


    for i in tqdm(range(epoch)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch: {i+1}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


        # intermediate_save_path = f"{save_path}_epoch_{i+1}.pth"
        # save_model(model, optimizer, normalise_parameter, path=intermediate_save_path)
        writer.add_scalar('Loss/Train', train_loss, i)
        writer.add_scalar('Loss/Validation', val_loss, i)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], i)
    writer.close()
    final_save_path = f"{save_path}_final.pth"
    save_model(model, optimizer, normalise_parameter, path=final_save_path)
    print(f"Final model saved to: {final_save_path}")


def print_argument_defaults():
    """
    Print the default values for all arguments in a table.
    """
    args_table = [
        ["Argument", "Description", "Default Value"],
        ["--path_to_lmdb", "Path to the LMDB dataset.", "Required"],
        ["--hidden_dim", "Number of hidden units in the model.", "184"],
        ["--learning_rate", "Learning rate for training.", "0.0008163"],
        ["--batch_size", "Batch size for training.", "113"],
        ["--dropout", "Dropout rate for the model.", "0.32357"],
        ["--epoch", "Number of epochs for training.", "500"],
        ["--heads", "Number of attention heads in the model.", "8"],
        ["--save_path", "Base path for saving model checkpoints.",
            "Thermalstability_model"],
    ]
    print("Argument Definitions and Defaults:")
    print(tabulate(args_table, headers="firstrow", tablefmt="fancy_grid"))


def parse_arguments():
    """
    Parse command-line arguments for training the GNN model.

    **returns:**
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="""
        Train a GAT model for predicting the bond dissociation enthalpy of
        MOFs. The model can be used to train any graph-like structure with
        a specified target."""
    )
    parser.add_argument('-p', '--path_to_lmdb', type=str,
                        required=True, help="Path to the LMDB dataset.")
    parser.add_argument('-hd', '--hidden_dim', type=int, default=184,
                        help="Number of hidden units in the model.")
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.0008163, help="Learning rate for training.")
    parser.add_argument('-bs', '--batch_size', type=int, default=113,
                        help="Batch size for training.")
    parser.add_argument('-d', '--dropout', type=float,
                        default=0.32357, help="Dropout rate for the model.")
    parser.add_argument('-head', '--heads', type=int, default=8,
                        help="Number of attention heads in the model.")
    parser.add_argument('-e', '--epoch', type=int, default=500,
                        help="Number of epochs for training.")
    parser.add_argument('-s', '--save_path', type=str, default="Thermalstability_model",
                        help="Base path for saving model checkpoints.")
    return parser


def main(path_to_lmdb, hidden_dim, learning_rate, batch_size, dropout, heads, epoch, save_path):
    """
    Train the GNN model with specified parameters.
    """
    print(f"Training with the following parameters:")
    print(f"  LMDB Path: {path_to_lmdb}")
    print(f"  Hidden Dimension: {hidden_dim}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Heads: {heads}")
    print(f"  Dropout: {dropout}")
    print(f"  Epochs: {epoch}")
    print(f"  Save Path: {save_path}")
    main_model(path_to_lmdb, hidden_dim, learning_rate,
               batch_size, dropout, heads, epoch, save_path)


def entry_point():
    """
    Entry point for the train_bde CLI command.
    Handles argument parsing and calls the main function with parsed arguments.
    """
    parser = parse_arguments()
    args = parser.parse_args()

    main(
        path_to_lmdb=args.path_to_lmdb,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        heads=args.heads,
        dropout=args.dropout,
        epoch=args.epoch,
        save_path=args.save_path
    )

if __name__ == "__main__":
    entry_point()