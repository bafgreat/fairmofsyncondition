import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class EnergyPredictionGCN(nn.Module):
    """
    A GCN-based model for predicting energy from molecular graph data.

    Parameters:
    -----------
    node_feature_dim : int
        Dimensionality of node features (e.g., atomic numbers).
    hidden_dim : int
        Number of hidden units in the GCN layers.
    output_dim : int
        Dimensionality of the output (for energy prediction, this will be 1).
    """

    def __init__(self, node_feature_dim, hidden_dim, output_dim=1):
        super(EnergyPredictionGCN, self).__init__()

        # Define two Graph Convolutional layers
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Batch normalization for stable training
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # MLP for graph-level energy prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data: Data):
        """
        Forward pass of the model.

        Parameters:
        -----------
        data : torch_geometric.data.Data
            A batched data object containing node features, edge index, and batch information.

        Returns:
        --------
        energy_pred : torch.Tensor
            Predicted energy for the input batch of graphs.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        # Apply batch normalization
        x = self.batch_norm(x)

        # Global pooling to aggregate node-level embeddings into graph-level representations
        x = global_mean_pool(x, batch)

        # Predict energy using the MLP
        energy_pred = self.mlp(x)

        return energy_pred

def train_model(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Parameters:
    -----------
    model : nn.Module
        The GCN model.
    dataloader : DataLoader
        DataLoader providing batches of graph data.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model weights.
    criterion : nn.Module
        Loss function (e.g., Mean Squared Error Loss).
    device : torch.device
        The device (CPU or GPU) for computation.

    Returns:
    --------
    float
        The average training loss over the epoch.
    """
    model.train()
    total_loss = 0

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation/test dataset.

    Parameters:
    -----------
    model : nn.Module
        The GCN model.
    dataloader : DataLoader
        DataLoader providing batches of graph data.
    criterion : nn.Module
        Loss function (e.g., Mean Squared Error Loss).
    device : torch.device
        The device (CPU or GPU) for computation.

    Returns:
    --------
    float
        The average loss over the dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def save_model(model, optimizer, epoch, filepath='gcn_model.pth'):
    """
    Save the trained model and optimizer state for future use.

    Parameters:
    -----------
    model : nn.Module
        The trained GCN model.
    optimizer : torch.optim.Optimizer
        The optimizer used during training.
    epoch : int
        The epoch number (useful for resuming training).
    filepath : str
        Path where the model will be saved.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, model, optimizer=None):
    """
    Load a saved model and optionally its optimizer state.

    Parameters:
    -----------
    filepath : str
        Path to the saved model file.
    model : nn.Module
        The model instance where the state_dict will be loaded.
    optimizer : torch.optim.Optimizer, optional
        The optimizer instance to load its state (for resuming training).

    Returns:
    --------
    int
        The epoch number to resume training from.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Model loaded from {filepath}, resuming from epoch {epoch}")
    return epoch

# Example usage:

if __name__ == "__main__":
    # Hyperparameters
    node_feature_dim = 118  # For atomic number embeddings
    hidden_dim = 64         # Hidden units in GCN
    output_dim = 1          # Energy prediction
    lr = 0.001              # Learning rate
    epochs = 100            # Number of epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    model = EnergyPredictionGCN(node_feature_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Assuming `train_loader` and `val_loader` are prepared DataLoader instances for your dataset
    # train_loader = ...
    # val_loader = ...

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, filepath='best_gcn_model.pth')

    # Optionally, load and evaluate the best model on a test set
    # test_loader = ...
    # load_model('best_gcn_model.pth', model)
    # test_loss = evaluate_model(model, test_loader, criterion, device)
    # print(f"Test Loss: {test_loss:.4f}")
