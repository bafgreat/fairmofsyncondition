from fairmofsyncondition.model.models import MOFConditionPredictor
from pathlib import Path
from mofstructure import mofdeconstructor
import numpy as np

from fairmofsyncondition.crystal.crystal2data import DataStructure
from fairmofsyncondition.read_write import coords_library

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from config_anto import IDX_TO_METALSALT,IDX_TO_MODULATOR,IDX_TO_SOLVENT


def get_metal_salt_recomendation(ase,data,top_x=3):

    target_field, target_dim = "metal_salt", 134
    device= "cpu"
    model_path = "trained_models/best_model_metal_salt.pt"
    mask = [bool(mofdeconstructor.is_metal(i)[0]) for i in ase.get_chemical_symbols()]
    metal = np.unique(np.array(ase.get_chemical_symbols())[mask], return_counts=True)[0].item()

    make_right_metal = []
    for k,v in IDX_TO_METALSALT.items():
        if v[:2] == metal:
            make_right_metal.append(True)
        else:
            make_right_metal.append(False)


    model = MOFConditionPredictor(
        target_dim=target_dim,
        global_fields=["lattice", "atomic_one_hot", "oms"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))


    for batch in DataLoader([data], batch_size=1):
        break

    model.eval()

    with torch.no_grad():
        logits = model(batch)  # shape: [batch_size, num_classes]
        logits = logits[0]  # shape: [num_classes]
        make_right_metal = torch.as_tensor(
            make_right_metal,
            dtype=torch.bool,
            device=logits.device
        )

        # get indices of valid classes (where make_right_metal is True)
        valid_class_indices = torch.where(make_right_metal)[0]

        # filter
        valid_logits = logits[valid_class_indices]
        # softmax only valid salts
        valid_probs = F.softmax(logits, dim=-1)[valid_class_indices]
        # top-k only on compatible salts
        top_probs, top_pos = torch.topk(
            valid_probs,
            k=min(top_x, valid_probs.numel()),
            dim=-1
        )
        # convert to original indices
        top_indices = valid_class_indices[top_pos]


    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        idx = int(idx.item())
        prob = float(prob.item())
        name = IDX_TO_METALSALT[idx]
        print(f"{rank}. idx={idx} | prob={prob:.4f} | {name}")

def get_modulator_reccomenation(data,top_x=3):
    target_field, target_dim = "modulator", 18
    device= "cpu"

    model_path = "trained_models/best_model_modulator.pt"

    model = MOFConditionPredictor(
        target_dim=target_dim,
        global_fields=["lattice", "atomic_one_hot", "space_group_number", "crystal_system", "oms"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))



    for batch in DataLoader([data], batch_size=1):
        break

    model.eval()

    with torch.no_grad():
        logits = model(batch)

        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(
            probs,
            k=top_x,
            dim=-1
        )

    for rank, (idx, prob) in enumerate(zip(top_indices[0], top_probs[0]), start=1):
        idx = int(idx.item())
        prob = float(prob.item())

        name = IDX_TO_MODULATOR[idx]

        print(f"{rank}. idx={idx} | prob={prob:.4f} | {name}")  


def get_solvent_reccomenation(data,top_x=3):
    target_field, target_dim = "solvent", 23
    device= "cpu"
    model_path = "trained_models/best_model_solvent.pt"

    model = MOFConditionPredictor(
        target_dim=target_dim,
        global_fields=["lattice", "atomic_one_hot", "space_group_number" ]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    for batch in DataLoader([data], batch_size=1):
        break

    model.eval()

    with torch.no_grad():
        logits = model(batch)

        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(
            probs,
            k=top_x,
            dim=-1
        )

    for rank, (idx, prob) in enumerate(zip(top_indices[0], top_probs[0]), start=1):
        idx = int(idx.item())
        prob = float(prob.item())

        name = IDX_TO_SOLVENT[idx]

        print(f"{rank}. idx={idx} | prob={prob:.4f} | {name}")


import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run salt, modulator, and solvent recommendations for a MOF file."
    )

    parser.add_argument(
        "--mof_path",
        type=Path,
        help="Path to the MOF structure file, e.g. ZnC5HO5.cif",
    )

    parser.add_argument(
        "--top-x",
        type=int,
        default=3,
        help="Number of recommendations to return. Default: 3",
    )

    args = parser.parse_args()

    cif_path = args.mof_path

    if not cif_path.exists():
        raise FileNotFoundError(f"MOF file not found: {cif_path}")

    # Load MOF
    data = DataStructure(str(cif_path)).complete_torch_data()
    ase = coords_library.pytorch_geometric_to_ase(data)

    print("Metal salt recommendation:")
    get_metal_salt_recomendation(ase, data, top_x=args.top_x)

    print("\nModulator recommendation:")
    get_modulator_reccomenation(data, top_x=args.top_x)

    print("\nSolvent recommendation:")
    get_solvent_reccomenation(data, top_x=args.top_x)


if __name__ == "__main__":
    main()