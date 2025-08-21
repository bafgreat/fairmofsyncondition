import os
from mofstructure import structure, mofdeconstructor
from mofstructure.filetyper import load_iupac_names
from fairmofsyncondition.read_write import cheminfo2iupac, coords_library, filetyper
from ase.data import atomic_numbers
from ase.io import read
import torch

data = "/Volumes/Extreme_Pro/FAIRDATA/Selected_Folder/hMOF-9958.cif"

inchi_corrector = {
    "FDTQOZYHNDMJCM-UHFFFAOYSA-N":"benzene-1,4-dicarboxylic acid"
}
def get_ligand_iupacname(ligand_inchi):
    print(ligand_inchi)
    name = load_iupac_names().get(ligand_inchi, None)
    if name is None:
        pubchem = cheminfo2iupac.pubchem_to_inchikey(ligand_inchi, name='inchikey')
        if pubchem is None:
            name = inchi_corrector.get(ligand_inchi, ligand_inchi)
        else:
            pubchem.get('iupac_name', ligand_inchi)
    return name


def load_system(filename):
    """
    A function to extract
    """
    data = {}
    os.makedirs('LigandsXYZ', exist_ok=True)
    ase_data = read(filename)
    structure_data = structure.MOFstructure(ase_atoms=ase_data)
    _, ligands = structure_data.get_ligands()

    inchikeys = [ligand.info.get('inchikey') for ligand in ligands]
    for inchi, ligand in zip(inchikeys, ligands):
        ligand.write(f'LigandsXYZ/{inchi}.xyz')
    ligands_names = [get_ligand_iupacname(i) for i in inchikeys]
    general = structure_data.get_oms()
    oms = general.get('has_oms')
    metal_symbols = general.get('metals')
    metals_atomic_number = [atomic_numbers[i] for i in metal_symbols]
    torch_data = coords_library.ase_to_pytorch_geometric()
    oms = torch.tensor([1 if  oms else 0], dtype=torch.int16)
    torch_data.oms = oms
    data['general'] = general

    return torch_data, metals_atomic_number, inchikeys, ligands_names


# load_system(data)

print(filetyper.category_names().keys())
