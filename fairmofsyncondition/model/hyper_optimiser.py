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

import argparse
import random
import optuna
from torch import nn
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from fairmofsyncondition.read_write import coords_library, filetyper
from fairmofsyncondition.model.thermodynamic_stability import  EnergyGNN_GAT2Conv, EnergyGNN_GIN, GraphLatticeModel, train, evaluate, load_dataset


def fine_opt_paramter(path_to_lmbd, mol_def):
    """
    Fine-tuning the optimization parameters using Optuna.

    Parameters:
        path_to_lmbd (str): Path to the lambda file.
        mol_def (str): Molecule definition ('GAT' or 'GIN').

    Returns:
        float: Validation loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(trial):
        """
        Objective function for hyperparameter optimization using Optuna.

        Parameters:
            trial (optuna.Trial): A trial object for sampling hyperparameters.

        Returns:
            float: Validation loss.
        """

        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        learning_rate = trial.suggest_categorical(
            "learning_rate", [1e-4, 1e-3, 1e-2, 1e-1])
        batch_size = trial.suggest_categorical("batch_size", [2048])
        dropout = trial.suggest_float("dropout", 0.1, 0.9)
        heads = trial.suggest_int("heads", 2, 6)
        # epoch = trial.suggest_int("epoch", 100,  5000)

        train_loader, val_loader, test_dataset, normalise_parameter = load_dataset(path_to_lmbd, batch_size=batch_size)
        if mol_def == 'GAT':
            model = EnergyGNN_GAT2Conv(input_dim=4, hidden_dim=hidden_dim, output_dim=1,
                          edge_dim=1, heads=heads, dropout=dropout).to(device)
        elif mol_def == 'GIN':
            model = EnergyGNN_GIN(input_dim=4, hidden_dim=hidden_dim, output_dim=1,
                          edge_dim=1, heads=heads, dropout=dropout).to(device)
        elif mol_def == 'lattice':
            lattice_hidden_dim = trial.suggest_categorical("lattice_hidden_dim",[ 5, 10, 15, 20, 25])
            num_layers= trial.suggest_categorical("num_layers",[ 2, 3, 4])
            model = GraphLatticeModel(input_dim=4,
                                gnn_hidden_dim=hidden_dim,
                                lattice_hidden_dim=lattice_hidden_dim,
                                output_dim=1,
                                num_layers=num_layers,
                                dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()

        for epoch in tqdm(range(100)):
            train_loss = train(model, train_loader,
                               optimizer, criterion, device)

        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Trial: Hidden_dim={hidden_dim}, LR={learning_rate:.5f}, Batch_size={batch_size}, Heads={heads}, Dropout={dropout}, train_loss={train_loss:.4f} Val_loss={val_loss:.4f}")
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    print(f"Optimal parameters: {best_params}")
    print(f"Best validation loss: {study.best_value}")

    best_model_config = {
        "hidden_dim": best_params["hidden_dim"],
        "learning_rate": best_params["learning_rate"],
        "batch_size": best_params["batch_size"],
        "heads": best_params["heads"],
        "dropout": best_params["dropout"],
        "epoch": best_params["epoch"],
        "val_loss": study.best_value
    }
    if mol_def == 'lattice':
        best_model_config = {
        "hidden_dim": best_params["hidden_dim"],
        "learning_rate": best_params["learning_rate"],
        "batch_size": best_params["batch_size"],
        "heads": best_params["heads"],
        "dropout": best_params["dropout"],
        "epoch": best_params["epoch"],
        "lattice_hidden_dim": best_params["lattice_hidden_dim"],
        "num_layers": best_params["num_layers"],
        "val_loss": study.best_value
    }

    output_file = f"{mol_def}_best_model_config.txt"
    with open(output_file, "w") as f:
        for key, value in best_model_config.items():
            f.write(f"{key}: {value}\n")

    print(f"Best model configuration written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for EnergyGNN model.")
    parser.add_argument("--path_to_lmbd", type=str,
                        required=True, help="Path to the dataset directory")
    parser.add_argument("--mol_def", type=str,
                        required=True, choices=["GAT", "GIN", "lattice"],
                        help="Model definition: GAT, GIN, or lattice")
    args = parser.parse_args()
    fine_opt_paramter(args.path_to_lmbd, args.mol_def)


if __name__ == "__main__":
    main()
