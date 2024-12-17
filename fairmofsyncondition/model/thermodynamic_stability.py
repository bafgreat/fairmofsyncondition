
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
import inspect
from torch import nn
from torch.nn import functional
from torch import optim
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATv2Conv, global_mean_pool, GINConv, BatchNorm, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
from fairmofsyncondition.read_write import coords_library, filetyper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EnergyGNN_GAT2Conv(nn.Module):
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

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 edge_dim,
                 heads=4,
                 dropout=0.2
                 ):
        super(EnergyGNN_GAT2Conv, self).__init__()
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


class EnergyGNN_GIN(nn.Module):
    """
    Implements a Graph Neural Network (GNN) using Graph Isomorphism Network (GIN) layers
    for graph-level prediction tasks such as regression or classification.

    Attributes:
        input_dim (int): Dimensionality of input node features.
        hidden_dim (int): Dimensionality of hidden layers.
        output_dim (int): Dimensionality of the output.
        dropout (float): Dropout rate for regularization. Default is 0.2.

    Layers:
        - Three GINConv layers, each using an MLP for message passing.
        - Batch normalization after each GINConv layer.
        - Global mean pooling for aggregating node-level features to graph-level.
        - Fully connected layers for output prediction.

    Methods:
        forward(data):
            Performs a forward pass of the model on input graph data.
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 edge_dim=None,
                 heads=None,
                 dropout=0.2
                 ):
        """
        Initializes the GNN model.

        Args:
            input_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Dimensionality of the output.
            edge_dim (int, optional): Dimensionality of edge features. Not used in this implementation.
            heads (int, optional): Number of attention heads. Not used in this implementation.
            dropout (float, optional): Dropout rate for regularization. Default is 0.2.
        """
        super(EnergyGNN_GIN, self).__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)

        # Define MLPs for GINConv layers
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(self.mlp2)

        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv3 = GINConv(self.mlp3)

        # Batch normalization layers
        self.norm1 = BatchNorm(hidden_dim)
        self.norm2 = BatchNorm(hidden_dim)
        self.norm3 = BatchNorm(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        """
        Defines the forward pass for the model.

        Args:
            data (torch_geometric.data.Data): Input graph data object containing:
                - x (torch.Tensor): Node features of shape [num_nodes, input_dim].
                - edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
                - edge_attr (torch.Tensor, optional): Edge features (not used here).
                - batch (torch.Tensor): Batch indices for mini-batch training.

        Returns:
            torch.Tensor: Final output predictions of shape [num_graphs, output_dim].
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply GIN layers with normalization and ReLU activations
        x = functional.relu(self.norm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = functional.relu(self.norm2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = functional.relu(self.norm3(self.conv3(x, edge_index)))
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)  # Mean pooling

        # Fully connected layers with dropout
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return functional.relu(x)


class GraphLatticeModel(nn.Module):
    """
    A Graph Neural Network (GNN) model that incorporates both graph-based features and lattice matrix features
    for graph-level prediction tasks such as property regression or classification.

    Attributes:
        input_dim (int): Dimensionality of input node features.
        gnn_hidden_dim (int): Dimensionality of GNN hidden layers.
        lattice_hidden_dim (int): Dimensionality of hidden layers for lattice matrix processing.
        output_dim (int): Dimensionality of the output (e.g., target properties).
        num_layers (int): Number of GNN layers. Default is 3.
        dropout (float): Dropout rate for regularization. Default is 0.2.

    Methods:
        forward(data):
            Performs a forward pass through the model using the input graph and lattice data.
    """

    def __init__(self,
                 input_dim,
                 gnn_hidden_dim,
                 lattice_hidden_dim,
                 output_dim,
                 num_layers=3,
                 dropout=0.2
                 ):
        """
        Initializes the GraphLatticeModel.

        Args:
            input_dim (int): Dimensionality of input node features.
            gnn_hidden_dim (int): Dimensionality of GNN hidden layers.
            lattice_hidden_dim (int): Dimensionality of hidden layers for lattice matrix processing.
            output_dim (int): Dimensionality of the output.
            num_layers (int, optional): Number of GNN layers. Default is 3.
            dropout (float, optional): Dropout rate for regularization. Default is 0.2.
        """
        super(GraphLatticeModel, self).__init__()
        input_dim = int(input_dim)
        hidden_dim = int(gnn_hidden_dim)
        lattice_hidden_dim = int(lattice_hidden_dim)
        output_dim = int(output_dim)
        num_layers = int(num_layers)

        # Define GINConv layers, their MLPs, and batch normalization dynamically
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(BatchNorm(hidden_dim))

        # MLP for lattice matrix processing
        self.mlp = nn.Sequential(
            # Lattice matrix flattened (3x3 = 9)
            nn.Linear(9, lattice_hidden_dim),
            nn.ReLU(),
            nn.Linear(lattice_hidden_dim, lattice_hidden_dim),
            nn.ReLU()
        )

        # Fully connected layers for final prediction
        combined_dim = gnn_hidden_dim + lattice_hidden_dim
        self.fc1 = nn.Linear(combined_dim, gnn_hidden_dim)
        self.fc2 = nn.Linear(gnn_hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        """
        Performs a forward pass of the model.

        Args:
            data (torch_geometric.data.Data): Input data containing:
                - x (torch.Tensor): Node features of shape [num_nodes, input_dim].
                - edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
                - edge_attr (torch.Tensor, optional): Edge features (not used here).
                - batch (torch.Tensor): Batch indices for mini-batch training.
                - lattice (torch.Tensor): Lattice matrix of shape [num_graphs, 3, 3].

        Returns:
            torch.Tensor: Final predictions of shape [num_graphs, output_dim].
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        lattice = data.lattice.view(data.lattice.size(0), -1)

        # Apply GIN layers with normalization and ReLU activations
        for conv, norm in zip(self.convs, self.norms):
            x = functional.relu(norm(conv(x, edge_index)))
            x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)  # Mean pooling

        # MLP processing of lattice matrix
        lattice_rep = self.mlp(lattice)

        # Concatenate graph and lattice representations
        combined_rep = torch.cat([x, lattice_rep], dim=1)

        # Fully connected layers with dropout
        combined_rep = functional.relu(self.fc1(combined_rep))
        combined_rep = self.dropout(combined_rep)
        out = self.fc2(combined_rep)

        # Scale output to [0, 1]
        out = torch.tanh(out) * 0.5 + 0.5
        return out


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    Attributes:
        patience (int): Number of epochs/iteractions to wait for an improvement in validation loss before stopping. Default is 5.
        delta (float): Minimum improvement in validation loss required to reset the patience counter. Default is 0.001.
        best_loss (float): The lowest validation loss observed so far. Initialized to infinity.
        counter (int): Tracks the number of consecutive epochs without improvement in validation loss.
        early_stop (bool): Flag indicating whether early stopping condition is met.
        path (str): File path to save the best model checkpoint. Default is 'checkpoint.pth'.

    Methods:
        __call__(val_loss, model):
            Evaluates the validation loss and decides whether to stop training or save a checkpoint.
        save_checkpoint(model):
            Saves the model's state dictionary to the specified checkpoint file.
    """

    def __init__(self,
                 patience=5,
                 delta=0.001,
                 path='checkpoint.pth'
                 ):
        """
        Initializes the EarlyStopping object with specified parameters.

        Args:
            patience (int, optional): Number of epochs to wait for an improvement in validation loss. Default is 5.
            delta (float, optional): Minimum improvement in validation loss required. Default is 0.001.
            path (str, optional): Path to save the best model checkpoint. Default is 'checkpoint.pth'.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.path = path

    def __call__(self,
                 val_loss,
                 model,
                 optimizer,
                 normalise_parameter
                 ):
        """
        Evaluates the validation loss and decides whether to stop training or save a model checkpoint.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The PyTorch model being trained.

        Behavior:
            - If the validation loss improves by at least `delta`, resets the counter and saves a checkpoint.
            - Otherwise, increments the counter.
            - Stops training if the counter exceeds or equals `patience`.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, optimizer, normalise_parameter)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


    def save_checkpoint(self,
                        model,
                        optimizer,
                        normalise_parameter
                        ):
        """
        Saves the current model state to the checkpoint file.

        Args:
            model (torch.nn.Module): The PyTorch model being trained.
            optimizer (optim.Optimizer): Optimizer for updating model parameters.
            normalise_parameter (float): Normalization parameter for target values.

        Behavior:
            - Saves the model's state dictionary to the file specified by `path`.
        """
        save_model(model, optimizer, normalise_parameter, self.path)


def train(model,
          dataloader,
          optimizer,
          criterion,
          device
          ):
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
        data.lattice = data.lattice.reshape(-1,3,3)
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
    y_values = torch.cat([data.y for data in dataset]
                         )  # Gather all target values
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
        raise ValueError(
            "Unsupported normalization method. Choose 'min-max' or 'z-score'.")


def inverse_normalize(predictions,
                      normalization_params,
                      method='z-score'):
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
        raise ValueError(
            "Unsupported normalization method. Choose 'min-max' or z-score")


def evaluate(model,
             dataloader,
             criterion,
             device
             ):
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
            data.lattice = data.lattice.reshape(-1,3,3)
            data = data.to(device)
            labels = data.y
            predictions = model(data).view(-1)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)

    return average_loss


def save_model(model,
               optimizer,
               normalise_parameter,
               path="model.pth"
               ):
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


def transform_target(test_data,
                     normalize_param,
                     method='z-score'
                     ):
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
    elif method == 'min-max':
        min_val, max_val = normalize_param['min'], normalize_param['max']
        for data in test_data:
            data.y = (data.y - min_val) / (max_val - min_val)
    else:
        raise ValueError(
            "Unsupported normalization method. Choose 'z-score' or'min-max'.")
    return test_data


def load_dataset(path_to_lmdb,
                 batch_size,
                 train_size=0.9,
                 random_seed=42,
                 shuffle=True,
                 normalize='full'
                 ):
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
        print("error")
        train_loader = DataLoader(train_data_norm, batch_size=batch_size, shuffle=True)
        train_loader, normalise_parameter = normalize_data(train_loader)
        val_data = transform_target(val_data, normalise_parameter)
        test_dataset = transform_target(test_dataset, normalise_parameter)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_dataset, normalise_parameter


def GAT2Conv_main_model(path_to_lmdb,
                        hidden_dim,
                        learning_rate,
                        batch_size,
                        dropout,
                        heads,
                        epoch,
                        patience,
                        print_every,
                        save_path
                        ):
    """
    Main function for training the GAT2Conv model.

    **parameters**:
        path_to_lmdb (str): Path to the LMDB file containing the dataset.
        hidden_dim (int): Dimensionality of the hidden layer.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate for the model.
        heads (int): Number of attention heads for the GAT layer.
        epoch (int): Number of training epochs.
        patience (int): Patience for early stopping.
        print_every (int): Frequency of printing training loss and validation loss.
        save_path (str): Path to save the best model.

    **Bahaviors **
        This function loads the dataset, trains the GAT2Conv model, and saves the best model.
    """

    writer = SummaryWriter(log_dir="errorlogger/energy_gat")
    early_stopping = EarlyStopping(
        patience=patience, path= 'best_'+save_path)
    model = EnergyGNN_GAT2Conv(input_dim=4,
                               hidden_dim=hidden_dim,
                               output_dim=1,
                               edge_dim=1,
                               heads=heads,
                               dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

    criterion = nn.L1Loss()

    train_loader, val_loader, test_dataset, normalise_parameter = load_dataset(
        path_to_lmdb, batch_size)
    for i in tqdm(range(epoch)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        early_stopping(val_loss, model, optimizer, normalise_parameter)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        if i % print_every == 0:
            print(
                f"Epoch: {i+1}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # model.load_state_dict(torch.load(save_path))
    # model, optimizer, normalise_parameter = load_model(save_path)
    save_model(model, optimizer, normalise_parameter, save_path)

    val_loss = evaluate(model, val_loader, criterion, device)
    print("Loaded the best model saved during training.", val_loss)


def EnergyGNN_GIN_main_model(path_to_lmdb,
                             hidden_dim,
                             learning_rate,
                             batch_size,
                             dropout,
                             epoch,
                             patience,
                             print_every,
                             save_path):
    """
    Main function for training the EnergyGNN_GIN model.

    **parameters**:
        path_to_lmdb (str): Path to the LMDB file containing the dataset.
        hidden_dim (int): Dimensionality of the hidden layer.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate for the model.
        epoch (int): Number of training epochs.
        patience (int): Patience for early stopping.
        print_every (int): Frequency of printing training loss and validation loss.
        save_path (str): Path to save the best model.

    **Bahaviors **
        This function loads the dataset, trains the EnergyGNN_GIN model, and saves the best model.
    """


    writer = SummaryWriter(log_dir="errorlogger/energy_gat")
    early_stopping = EarlyStopping(patience=patience, path=save_path)
    model = EnergyGNN_GIN(input_dim=4,
                          hidden_dim=hidden_dim,
                          output_dim=1,
                          edge_dim=1,
                          dropout=dropout).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.00001)
    criterion = nn.L1Loss()

    train_loader, val_loader, test_dataset, normalise_parameter = load_dataset(
        path_to_lmdb, batch_size)

    for i in tqdm(range(epoch)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        early_stopping(val_loss, model, optimizer, normalise_parameter)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        if i % print_every == 0:
            print(
                f"Epoch: {i+1}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    save_model(model, optimizer, normalise_parameter, save_path)

    val_loss = evaluate(model, val_loader, criterion, device)
    print("Loaded the best model saved during training.", val_loss)


def GraphLatticeModel_main_model(path_to_lmdb,
                                 hidden_dim,
                                 lattice_hidden_dim,
                                 num_layers,
                                 learning_rate,
                                 batch_size,
                                 dropout,
                                 epoch,
                                 patience,
                                 print_every,
                                 save_path
                                 ):
    """
    Main function for training the EnergyGNN_GIN model.

    **parameters**:
        path_to_lmdb (str): Path to the LMDB file containing the dataset.
        hidden_dim (int): Dimensionality of the hidden layer.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate for the model.
        epoch (int): Number of training epochs.
        patience (int): Patience for early stopping.
        print_every (int): Frequency of printing training loss and validation loss.
        save_path (str): Path to save the best model.

    **Bahaviors **
        This function loads the dataset, trains the EnergyGNN_GIN model, and saves the best model.
    """

    writer = SummaryWriter(log_dir="errorlogger/gat2conv")
    early_stopping = EarlyStopping(patience=patience, path=save_path)

    model = GraphLatticeModel(input_dim=4,
                              gnn_hidden_dim=hidden_dim,
                              lattice_hidden_dim=lattice_hidden_dim,
                              output_dim=1,
                              num_layers=num_layers,
                              dropout=dropout).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.00001)
    criterion = nn.L1Loss()

    train_loader, val_loader, test_dataset, normalise_parameter = load_dataset(
        path_to_lmdb, batch_size)

    for i in tqdm(range(epoch)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        early_stopping(val_loss, model, optimizer, normalise_parameter)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        if i % print_every == 0:
            print(
                f"Epoch: {i+1}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    save_model(model, optimizer, normalise_parameter, save_path)

    val_loss = evaluate(model, val_loader, criterion, device)
    print("Loaded the best model saved during training.", val_loss)


def display_model_helper(model):
    """
    Display clear and concise information about the selected model and its key parameters.

    Args:
        model (str): The name of the model. Options are:
                     - "EnergyGNN_GAT2Conv": Graph Attention Networks (GATv2) based model.
                     - "EnergyGNN_GIN": Graph Isomorphism Network (GIN) based model.
                     - "GraphLatticeModel": Combines graph features with lattice matrix processing.

    Instructions:
        - Review the description for each model.
        - Identify the required parameters for your use case.
        - Ensure you provide the necessary inputs when configuring the model.

    Example Usage:
        display_model_helper("EnergyGNN_GIN")
    """
    model_details = {
        "EnergyGNN_GAT2Conv": """
        Model: EnergyGNN_GAT2Conv
        -------------------------
        - Type: Graph Neural Network using GATv2 (Graph Attention Networks).
        - Suitable for: Models that require attention mechanisms on graph edges.
        - Key Parameters:
            - `hidden_dim`: Number of hidden units in the network layers (e.g., 128 or 256).
            - `heads`: Number of attention heads for the GAT layer (e.g. 2, 4 ,6 , 8).
            - `dropout`: Regularization rate to prevent overfitting (e.g., 0.2 - 0.9).
        """,
        "EnergyGNN_GIN": """
        Model: EnergyGNN_GIN
        --------------------
        - Type: Graph Neural Network using GIN (Graph Isomorphism Networks).
        - Suitable for: Dense and message-passing networks for graph regression tasks.
        - Key Parameters:
            - `hidden_dim`: Number of hidden units in the network layers (e.g., 128 or 256).
            - `dropout`: Regularization rate to prevent overfitting (e.g., 0.1 or 0.3).
        """,
        "GraphLatticeModel": """
        Model: GraphLatticeModel
        ------------------------
        - Type: Hybrid Graph and Lattice Neural Network.
        - Combines: Graph-based features and 3x3 lattice matrix for property predictions.
        - Suitable for: Tasks requiring structural and lattice-based properties.
        - Key Parameters:
            - `gnn_hidden_dim`: Number of hidden units for the graph network (e.g., 128 or 256).
            - `lattice_hidden_dim`: Number of hidden units for lattice matrix processing (e.g., 64 or 128).
            - `num_layers`: Number of GNN layers to use (e.g., 3 or 5).
            - `dropout`: Regularization rate to prevent overfitting (e.g., 0.2 or 0.5).
        """
    }

    if model in model_details:
        print(model_details[model])
    else:
        print(f"No helper available for the model '{model}'. Please choose from:")
        print("- EnergyGNN_GAT2Conv")
        print("- EnergyGNN_GIN")
        print("- GraphLatticeModel")

def parse_arguments():
    """
    Parse command-line arguments for training the GNN model.

    Returns:
        argparse.Namespace: Parsed command-line arguments with detailed descriptions.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a Graph Neural Network (GNN) model for training a bond dissociation enthalpy predictor of Metal-Organic Frameworks (MOFs).\n"
            "Select the appropriate model and provide necessary hyperparameters.\n\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,  # Automatically show defaults in help
    )

    # General arguments
    parser.add_argument(
        '-p', '--path_to_lmdb',
        type=str,
        required=True,
        help="Path to the LMDB file containing the dataset.\n\n"
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=["EnergyGNN_GAT2Conv", "EnergyGNN_GIN", "GraphLatticeModel"],
        required=True,
        help=(
            "Model to train. Choose one of the following:\n\n"
            "  - EnergyGNN_GAT2Conv: Graph Attention Network (GATv2).\n\n"
            "  - EnergyGNN_GIN: Graph Isomorphism Network (GIN).\n\n"
            "  - GraphLatticeModel: Combines graph features with lattice matrix.\n\n\n"
        )
    )

    # Model hyperparameters
    parser.add_argument(
        '-hd', '--hidden_dim',
        type=int,
        default=256,
        help="Number of hidden units in the network layers.\n\n"
    )
    parser.add_argument(
        '-lhd', '--lattice_hidden_dim',
        type=int,
        default=3,
        help=(
            "Number of hidden units for processing lattice matrix (only for GraphLatticeModel).\n\n"
        )
    )
    parser.add_argument(
        '-nl', '--num_layers',
        type=int,
        default=3,
        help=(
            "Number of GNN layers (used only in GraphLatticeModel).\n"
            "Ignored for other models.\n\n"
        )
    )

    # Training hyperparameters
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=0.001,
        help="Learning rate for training the model.\n\n"
    )
    parser.add_argument(
        '-bs', '--batch_size',
        type=int,
        default=2048,
        help="Batch size for training.\n\n"
    )
    parser.add_argument(
        '-d', '--dropout',
        type=float,
        default=0.1,
        help="Dropout rate for regularization to prevent overfitting.\n\n"
    )
    parser.add_argument(
        '-head', '--heads',
        type=int,
        default=4,
        help="Number of attention heads (only for GATv2-based models).\n\n"
    )
    parser.add_argument(
        '-patience', '--patience',
        type=int,
        default=200,
        help="Number of epochs to wait without improvement before early stopping.\n\n"
    )
    parser.add_argument(
        '-print_every',
        type=int,
        default=10,
        help="Frequency (in epochs) of printing training progress and validation loss.\n\n"
    )
    parser.add_argument(
        '-e', '--epoch',
        type=int,
        default=500,
        help="Total number of epochs for training.\n\n"
    )

    # Output options
    parser.add_argument(
        '-s', '--save_path',
        type=str,
        default="model_checkpoint",
        help="Path to save the best model checkpoint.\n\n"
    )

    return parser



def entry_point():
    """
    Entry point for the training script. Dynamically calls the appropriate main function based on the selected model.
    """
    parser = parse_arguments()
    args = parser.parse_args()

    # Display helper for the selected model
    display_model_helper(args.model)

    # Dynamically select the appropriate main function for the chosen model
    if args.model == "EnergyGNN_GAT2Conv":
        GAT2Conv_main_model(
            path_to_lmdb=args.path_to_lmdb,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            dropout=args.dropout,
            heads=args.heads,
            epoch=args.epoch,
            patience=args.patience,
            print_every=args.print_every,
            save_path=args.save_path,
        )
    elif args.model == "EnergyGNN_GIN":
        EnergyGNN_GIN_main_model(
            path_to_lmdb=args.path_to_lmdb,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            dropout=args.dropout,
            epoch=args.epoch,
            patience=args.patience,
            print_every=args.print_every,
            save_path=args.save_path,
        )
    elif args.model == "GraphLatticeModel":
        GraphLatticeModel_main_model(
            path_to_lmdb=args.path_to_lmdb,
            hidden_dim=args.hidden_dim,
            lattice_hidden_dim=args.lattice_hidden_dim,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            dropout=args.dropout,
            epoch=args.epoch,
            patience=args.patience,
            print_every=args.print_every,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    entry_point()
