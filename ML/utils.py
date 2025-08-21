
import random
import numpy as np
import torch
from torch_geometric.data import Data



from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fix_target_shapes(data_list,attr=None):
    cleaned_data = []
    for data in data_list:
        target = getattr(data, attr, None)
        if target is not None and target.dim() == 2 and target.shape[0] > 1:
            # Prendi solo la prima riga
            setattr(data, attr, target[1].unsqueeze(0))  # Shape diventa [1, y]
        cleaned_data.append(data)
    return cleaned_data

def remove_unused_onehot_columns(data_list, target_field="metal_salts"):
    # Controllo esistenza campo
    if not hasattr(data_list[0], target_field):
        raise ValueError(f"Il campo '{target_field}' non esiste nei dati.")
    
    # Trova colonne non nulle
    field_matrix = getattr(data_list[0], target_field)
    used_columns = torch.zeros(field_matrix.shape[1], dtype=torch.bool)

    # Accumula colonne usate
    for data in data_list:
        matrix = getattr(data, target_field)
        used_columns |= matrix.bool().squeeze(0)

    # Filtra ogni grafo
    for data in data_list:
        matrix = getattr(data, target_field)
        filtered = matrix[:, used_columns]
        setattr(data, target_field, filtered)

    return data_list



from ase.data import chemical_symbols
def filter_metals(atomic_number_list):
    # Set of atomic numbers for known nonmetals and noble gases
    non_metals = { 1, 2, 5, 6, 7, 8, 9, 10,14, 15, 16, 17, 18,33, 34, 35, 36,52, 53, 54,85, 86}
    # Filter out nonmetals and invalid/blank entries

    metals = [

        Z for Z in atomic_number_list

        if Z not in non_metals and Z < len(chemical_symbols) and chemical_symbols[Z] != ''

    ]
    return metals