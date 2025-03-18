import os
import sys
import glob
import pickle
import lmdb
import torch
from pathlib import Path
from fairmofsyncondition.featurizer import encoder
from fairmofsyncondition.read_write import filetyper, coords_library

cif_paths = "/Volumes/X9/FAIRDATA/MOF_Data/Experiment_cif"
other = "/Volumes/X9/FAIRDATA/Curated_data"

paths = [Path(cif_paths), Path(other)]

lmdb_path = '../data/lmdb_data/mof_syncondition_data.lmdb'

size_strain_path = "/Volumes/X9/src/Python/fairmofsyncondition/data/json_data/mof_size_strain2.json"
ligand_solvent_path = "/Volumes/X9/src/Python/fairmofsyncondition/data/json_data/ligand_salt_solvent.json"
category_path = "/Volumes/X9/src/Python/fairmofsyncondition/data/json_data/catergories.json"
oms_path = "/Volumes/X9/src/Python/fairmofsyncondition/data/json_data/structure_oms_and_general_info.json"

def ase2data(filename):
    """
    Function to convert to convert from an ase atom object to
    a pytorch geometric data. The function reads any ase readable
    file and covert it to a standard pytorch geometric data. Note that
    this function is works on a simple structure and create a single
    pytorch data.

    **parameter**
        filename (str): file containing structure

    **return**
        data (torch.data): pytorch data
    """
    py_data = coords_library.ase_to_pytorch_geometric(filename)

    return py_data

def data_files(ligand_solvent_path, size_strain_path, oms_path, category_path):
    ligand_solvent = filetyper.load_data(ligand_solvent_path)
    size_strain = filetyper.load_data(size_strain_path)
    all_oms_data = filetyper.load_data(oms_path)
    category = filetyper.load_data(category_path)
    solvent_cat = category.get("solvents")
    ligand_cat = category.get("linker_reagent")
    salt_cat = category.get("metal_salts")

    return ligand_solvent, size_strain, all_oms_data, ligand_cat, salt_cat, solvent_cat


def save2lmbt(lmdb_path, ligand_solvent_path, size_strain_path, oms_path, category_path):
    file_mapper = {}
    unfinish = []
    with lmdb.open(lmdb_path, map_size=int(1e12)) as lmdb_env:
        with lmdb_env.begin(write=True) as txn:
            count = 0
            ligand_salt_solvent, size_strain, all_oms_data, ligand_cat, salt_cat, solvent_cat = data_files(ligand_solvent_path, size_strain_path, oms_path, category_path)
            # test = list(ligand_salt_solvent.keys())[:5]
            # all_refs = [i for i in ligand_salt_solvent if i.startswith("L")]
            for i, refcode in enumerate(ligand_salt_solvent):
                try:
                    print(i, refcode)
                    cif_file = str(next((p / f"{refcode}.cif" for p in paths if (p / f"{refcode}.cif").exists()), None))
                    # cif_file = f"{cif_paths}/{refcode}.cif"
                    ligands = ligand_salt_solvent[refcode].get('ligands')
                    solvents = ligand_salt_solvent[refcode].get('solvents')
                    metal_salt = ligand_salt_solvent[refcode].get('metal_salts')
                    modified_scherrer = torch.tensor(size_strain[refcode].get("modified_scherrer"), dtype=torch.float16)
                    av_strain = torch.tensor(size_strain[refcode].get("av_strain_size"), dtype=torch.float16)
                    solvent_encoder = encoder.onehot_encoder_pyg(solvents, solvent_cat)
                    ligand_encoder = encoder.onehot_encoder_pyg(ligands, ligand_cat)
                    salt_encoder = encoder.onehot_encoder_pyg(metal_salt, salt_cat)
                    has_oms = all_oms_data[refcode].get("has_oms")
                    oms = torch.tensor([1 if has_oms else 0], dtype=torch.int16)
                    data = ase2data(cif_file)
                    data.metal_salts = salt_encoder
                    data.ligands = ligand_encoder
                    data.solvents = solvent_encoder
                    data.modified_scherrer = modified_scherrer
                    data.microstrain = av_strain
                    data.oms = oms
                    txn.put(f"{i}".encode(), pickle.dumps(data))
                    file_mapper[f"{i}"] = refcode

                    count += 1

                except Exception:
                    unfinish.append(refcode)
                    unfinish.append('\n')
                    pass
                txn.put(b"__len__", pickle.dumps(count))

            print(count)
    filetyper.put_contents('unfinish.txt', unfinish)
    filetyper.write_json(file_mapper, "../data/json_data/torch_data_mapper.json")


save2lmbt(lmdb_path, ligand_solvent_path, size_strain_path, oms_path, category_path)

# test = coords_library.LMDBDataset(lmdb_path=lmdb_path)
# print(test[0])
# data = []
# ligand_salt_solvent, size_strain, ligand_cat, salt_cat, solvent_cat = data_files(ligand_solvent_path, size_strain_path, category_path)

# cif_file = [os.path.basename(i).split('.c')[0] for i in glob.glob(f"{cif_paths}/*cif")]
# for i in ligand_salt_solvent:
#     if i not in cif_file:
#         data.append(i)
# print(data)
# print(len(data))
# refcode = "ZZZBNA02"
# found_file = next((p / f"{refcode}.cif" for p in paths if (p / f"{refcode}.cif").exists()), None)

# found = str(found_file)
# print(type(found))