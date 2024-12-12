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
from fairmofsyncondition.model.thermodynamic_stability import EnergyGNN, train, evaluate, load_dataset


def fine_op_paramter(path_to_lmbd):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(trial):
        """
        Objective function for hyperparameter optimization using Optuna.

        Parameters:
            trial (optuna.Trial): A trial object for sampling hyperparameters.

        Returns:
            float: Validation loss.
        """

        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-6, 1e-1, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.9)
        heads = trial.suggest_int("heads", 2, 24)

        train_loader, val_loader, test_dataset, normalise_parameter = load_dataset(path_to_lmbd, batch_size=batch_size)

        model = EnergyGNN(input_dim=4, hidden_dim=hidden_dim, output_dim=1,
                          edge_dim=1, heads=heads, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(20)):
            train_loss = train(model, train_loader,
                               optimizer, criterion, device)

        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Trial: Hidden_dim={hidden_dim}, LR={learning_rate:.5f}, Batch_size={batch_size}, Heads={heads}, Dropout={dropout}, Val_loss={val_loss:.4f}")
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print(f"Optimal parameters: {best_params}")
    print(f"Best validation loss: {study.best_value}")

    best_model_config = {
        "hidden_dim": best_params["hidden_dim"],
        "learning_rate": best_params["learning_rate"],
        "batch_size": best_params["batch_size"],
        "heads": best_params["heads"],
        "dropout": best_params["dropout"],
        "val_loss": study.best_value
    }

    output_file = "best_model_config.txt"
    with open(output_file, "w") as f:
        for key, value in best_model_config.items():
            f.write(f"{key}: {value}\n")

    print(f"Best model configuration written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for EnergyGNN model.")
    parser.add_argument("--path_to_lmbd", type=str,
                        required=True, help="Path to the dataset directory")
    args = parser.parse_args()
    fine_op_paramter(args.path_to_lmbd)


if __name__ == "__main__":
    main()
