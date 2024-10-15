import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from fairmofsyncondition.read_write import coords_library, filetyper
from typing import List, Dict, Tuple, Optional
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield import base
from torch.nn.utils.rnn import pad_sequence


class CrystalDataset(Dataset):
    """
    Dataset class for loading crystal structures and their corresponding energies.

    Attributes:
    -----------
    cif_files : list of str
        List of file paths to the CIF files.
    energies : list of float
        List of corresponding energies for the CIF files.

    Methods:
    --------
    __len__():
        Returns the total number of samples in the dataset.

    __getitem__(idx):
        Returns the AtomGraphs object and corresponding energy for the given index.

    cif_to_atomgraphs(cif_file):
        Converts the CIF file at the specified path into an AtomGraphs object.
    """

    def __init__(self, cif_files: list, energies: list) -> None:
        """
        Initializes the dataset with CIF files and their corresponding energies.

        Parameters:
        -----------
        cif_files : list of str
            List of paths to CIF files.
        energies : list of float
            List of energies corresponding to each CIF file.
        """
        self.cif_files = cif_files
        self.energies = energies

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.cif_files)

    def __getitem__(self, idx: int) -> Tuple[AtomGraphs, torch.Tensor]:
        """
        Returns the AtomGraphs object and energy for a given index.

        Parameters:
        -----------
        idx : int
            The index of the sample to retrieve.

        Returns:
        --------
        tuple : A tuple containing the AtomGraphs object and the corresponding energy (tensor).
        """
        cif_file = self.cif_files[idx]
        energy = self.energies[idx]

        # Convert the CIF file to an AtomGraphs object
        atom_graph = self.cif_to_atomgraphs(cif_file)

        return atom_graph, torch.tensor(energy, dtype=torch.float32)

    def cif_to_atomgraphs(self, cif_file: str) -> AtomGraphs:
        """
        Converts a CIF file to an AtomGraphs object.

        Parameters:
        -----------
        cif_file : str
            Path to the CIF file to be converted.

        Returns:
        --------
        AtomGraphs : The AtomGraphs object representing the graph structure of the CIF file.
        """
        atom_graph = coords_library.ase_graph(cif_file)  # Custom method to convert CIF to AtomGraphs
        return atom_graph


def pad_collate_fn(batch):
    """
    Custom collate function to batch AtomGraphs and pad node and edge features to the same size.

    Parameters:
    -----------
    batch : list of tuples
        Each tuple contains an AtomGraphs object and the corresponding energy.

    Returns:
    --------
    batch_graphs : Batched AtomGraphs object
    energies : Tensor
        A batched AtomGraphs object and a tensor of energies.
    """
    graphs, energies = zip(*batch)  # Unpack the batch of tuples into separate lists

    # Batch the AtomGraphs objects
    batch_graphs = base.batch_graphs(list(graphs))

    # Convert energies to a tensor
    energies = torch.stack(energies)

    return batch_graphs, energies


class GNNModel(nn.Module):
    """
    Graph Neural Network (GNN) model for predicting the energy of crystal structures.

    The model takes batched AtomGraphs as input, consisting of node and edge features, and predicts the energy.
    """

    def __init__(self, node_feature_size: int, edge_feature_size: int, hidden_size: int = 128, output_size: int = 1) -> None:
        """
        Initializes the GNN model with fully connected layers.

        Parameters:
        -----------
        node_feature_size : int
            The size of the node feature vectors.
        edge_feature_size : int
            The size of the edge feature vectors.
        hidden_size : int, optional
            The number of hidden units in the fully connected layers (default is 128).
        output_size : int, optional
            The size of the output (default is 1, for energy prediction).
        """
        super(GNNModel, self).__init__()

        # Adjust the input size based on your node and edge feature dimensions
        self.node_fc1 = nn.Linear(node_feature_size, hidden_size)
        self.edge_fc1 = nn.Linear(edge_feature_size, hidden_size)

        # Output layer for energy prediction
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, graph: AtomGraphs) -> torch.Tensor:
        """
        Performs a forward pass of the GNN model.

        Parameters:
        -----------
        graph : AtomGraphs
            The input batched AtomGraphs object, containing node features and edge features.

        Returns:
        --------
        torch.Tensor : The predicted energy.
        """
        # Extract node and edge features from the AtomGraphs object
        node_features = graph.node_features['atomic_numbers'].float()

        # Make sure node_features has the correct shape. Reshape if necessary.
        node_features = node_features.view(-1, node_features.size(-1))

        edge_features = graph.edge_features['vectors']

        # Pass node and edge features through the fully connected layers
        node_rep = torch.relu(self.node_fc1(node_features))
        edge_rep = torch.relu(self.edge_fc1(edge_features))

        # Combine node and edge representations (example: simple summation)
        combined_rep = node_rep + edge_rep.mean(dim=1)

        # Final prediction
        energy_pred = self.fc_out(combined_rep)

        return energy_pred



def train_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = 20, device: str = "cpu") -> None:
    """
    Trains the GNN model for a specified number of epochs using the given data and optimizer.

    Parameters:
    -----------
    model : nn.Module
        The GNN model to train.
    dataloader : DataLoader
        DataLoader for batching the dataset during training.
    criterion : nn.Module
        Loss function (e.g., MSELoss) to compute the training loss.
    optimizer : optim.Optimizer
        Optimizer for updating the model parameters (e.g., Adam).
    num_epochs : int, optional
        Number of epochs for training (default is 20).
    device : str, optional
        The device to train the model on ("cpu" or "cuda", default is "cpu").
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data in dataloader:
            graph, energy = data

            # Move the data to the device
            graph = graph.to(device)
            energy = energy.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(graph)
            loss = criterion(output, energy)
            print(loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss per epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training completed.")


def predict_energy(model: nn.Module, cif_file: str, device: str = "cpu") -> float:
    """
    Predicts the energy for a given CIF file using the trained GNN model.

    Parameters:
    -----------
    model : nn.Module
        The trained GNN model.
    cif_file : str
        Path to the CIF file to be converted to a graph for prediction.
    device : str, optional
        The device to run the prediction on ("cpu" or "cuda", default is "cpu").

    Returns:
    --------
    float : The predicted energy for the CIF file.
    """
    model.eval()

    # Convert CIF file to AtomGraphs
    graph = coords_library.ase_graph(cif_file)  # Custom method to convert CIF to AtomGraphs
    graph = graph.to(device)

    # Predict energy
    with torch.no_grad():
        energy_pred = model(graph)

    return energy_pred.item()


def main() -> None:
    """
    Main function to load CIF files and energies, train the GNN model, and save it.
    """
    # Example file paths and energies
    path_to_cif = '../../../../../FAIRDATA/MOF_Data/Experiment_cif/'
    path_to_energy = filetyper.load_data('../../data/test_data.json')

    all_refcode = path_to_energy.keys()
    energies = [path_to_energy[i] for i in all_refcode]
    cif_files = [f"{path_to_cif}{i}.cif" for i in all_refcode]

    # Dataset and Dataloader
    dataset = CrystalDataset(cif_files, energies)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=pad_collate_fn)

    # Initialize model, criterion, and optimizer
    node_feature_size = 578 # Adjust based on your node features
    edge_feature_size = 3   # Adjust based on your edge features
    model = GNNModel(node_feature_size, edge_feature_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=20)

    # Save the trained model
    torch.save(model.state_dict(), "gnn_model.pth")


if __name__ == "__main__":
    main()
