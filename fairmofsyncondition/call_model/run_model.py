import os
import sys
import re
import argparse
import torch
from ase import Atoms
import numpy as np
from ase.io import read
from mofstructure.structure import MOFstructure
from mofstructure import mofdeconstructor
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mofstructure.filetyper import load_iupac_names
from fairmofsyncondition.read_write import cheminfo2iupac, coords_library, filetyper
from fairmofsyncondition.call_model.utils_model import get_models, ensemble_predictions, get_model_energy, get_energy_prediction
import warnings
from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
warnings.filterwarnings(
    "ignore",
    message="dict interface is deprecated. Use attribute interface instead",
    category=DeprecationWarning,
    module=r"pymatgen\.symmetry\.analyzer"
)

warnings.filterwarnings(
    "ignore",
    message=r"We strongly discourage using implicit binary/text `mode`",
    category=FutureWarning,
    module=r"pymatgen\.io\.cif"
)

smile_to_iupac = filetyper.smile_names_iupac()

solvent_and_inchi = filetyper.solvent_and_inchi()
ligand2solvent = filetyper.ligand2solvent()
salt2solvent = filetyper.salt2solvent()

convert_struct = {'cubic': 0,
                  'hexagonal': 1,
                  'monoclinic': 2,
                  'orthorhombic': 3,
                  'tetragonal': 4,
                  'triclinic': 5,
                  'trigonal': 6
                  }

device = "cpu"

def solvent_from_inchikey(inchikey: str, data: dict):
    """
    Return the FIRST matching name (key in `data`) for a given InChIKey.
    If none is found, return None.
    """
    for name, info in data.items():
        if info.get("inchikey") == inchikey:
            return name
    return None


def inchikey_proportions(counts_by_inchikey: dict, data: dict, use_names: bool = True, round_to: int = 2):
    """
    Convert an InChIKey->count mapping into a sorted list of (label, percent) tuples.

    - `counts_by_inchikey`: e.g. {'IOLCXVTUBQKXJR-UHFFFAOYSA-M': 1, 'ZMXDDKWLCZADIW-UHFFFAOYSA-N': 4}
    - `data`: the dictionary containing names -> {inchikey, ...}
    - `use_names`: if True, map each InChIKey to the first matching name via `solvent_from_inchikey`;
                   if no name is found, the InChIKey is used as the label.
                   If False, always use the InChIKey as the label.
    - `round_to`: rounding for percentage values.

    Returns:
        List[Tuple[str, float]] sorted by percentage descending.
    """
    total = sum(counts_by_inchikey.values())
    if total == 0:
        return []

    results = []
    for ik, cnt in counts_by_inchikey.items():
        pct = (cnt / total) * 100.0
        if use_names:
            label = solvent_from_inchikey(ik, data) or ik
        else:
            label = ik
        results.append((label, round(pct, round_to)))

    results.sort(key=lambda t: (-t[1], t[0]))
    return results

def get_ligand_iupacname(ligand_inchi, smile):
    name = load_iupac_names().get(ligand_inchi, None)

    if name is None:
        name = smile_to_iupac.get(smile, None)
    if name is None:
        pubchem = cheminfo2iupac.pubchem_to_inchikey(smile, name='smiles')
        if pubchem is None:
            name = None
        else:
            name = pubchem.get('iupac_name', None)
            smile = pubchem.get('smiles', None)
            inchikey = pubchem.get('inchikey', None)
    return name, smile, inchikey

class Synconmodel(object):
    def __init__(self,
                 ase_atom,
                 outfile="mof_prediction.txt"
                 ):
        """
        A class to run the metal salt prediction GNN model and for
        extracting organic ligands and predicting metal salts.
        """
        if isinstance(ase_atom, Atoms):
            self.ase_atom = ase_atom
            self.basename = 'MOF Data'
        elif isinstance(ase_atom, str) and os.path.isfile(ase_atom):
            self.ase_atom = read(ase_atom)
            self.basename = os.path.basename(ase_atom).split('.')[0]
        else:
            print("sorry input type is not recorgnised")
        self.outfile = outfile
        self.structure = MOFstructure(self.ase_atom)
        self.pymat = AseAtomsAdaptor.get_structure(self.ase_atom)
        self.torch_data = coords_library.ase_to_pytorch_geometric(self.ase_atom)
        self.category = filetyper.category_names()

    def convert_metals(self):
        '''
        return metals dict as a dictionary of symbols with index
        '''
        return {j: i for i, j in enumerate(mofdeconstructor.transition_metals()[1:])}

    def get_species_conc(self):
        """
        Create a one hot encodeing for the concentration of each
        atomic species found in the system. Similar to the empirical
        formular of the system.
        """
        emb = torch.zeros(120)
        atomic_num = self.ase_atom.get_atomic_numbers()
        a, b = np.unique(atomic_num, return_counts=True)
        for aa, bb in zip(a, b):
            emb[aa] = bb
        return emb

    def get_coordination_and_oms(self):
        """
        Get the oms embeding
        """
        general = self.structure.get_oms()
        metals = general['metals']
        tmp_dict = dict()
        emb = torch.zeros(96)
        for i in general["metal_info"]:
            cord = i["coordination_number"]
            metal = i["metal"]
            if metal in tmp_dict:
                if cord > tmp_dict[metal]:
                    tmp_dict[metal] = cord
            else:
                tmp_dict[metal] = cord

        for i, j in tmp_dict.items():
            emb[self.convert_metals()[i]] = j
        oms = general["has_oms"]
        return emb, oms, metals

    def get_organic_ligands(self):
        ligands_data = {}
        _, ligands = self.structure.get_ligands()
        inchikeys = [ligand.info.get('inchikey') for ligand in ligands]
        smiles = [ligand.info.get('smi') for ligand in ligands]
        ligands_names = [get_ligand_iupacname(i, j) for i, j in  zip(inchikeys, smiles)]
        return ligands_names # inchikeys, smiles, ligands_names

    def get_ligands_solvents(self):
        "Get potential solvents for ligands"
        all_solvents = {}
        ligands_names = self.get_organic_ligands()
        for data_name in ligands_names:
            names, smile, inchikey = data_name
            solvent = ligand2solvent.get(inchikey, None)
            if solvent != None:
                all_solvents[names] = inchikey_proportions(solvent, solvent_and_inchi, use_names=True)
            else:
                all_solvents[names] = None
        return all_solvents

    def get_metal_solvents(self, metal_salts):
        all_salt_solvents = {}
        for metal_salt in metal_salts:
            solvent = salt2solvent.get(metal_salt, None)
            if solvent != None:
                all_salt_solvents[metal_salt ] = inchikey_proportions(solvent, solvent_and_inchi, use_names=True)
            else:
                all_salt_solvents[metal_salt ] = None
        return all_salt_solvents

    def get_space_group(self):
        "Get space group embedding"
        emb_sg = torch.zeros(231)
        emb_cs = torch.zeros(7)
        sga = SpacegroupAnalyzer(self.pymat)
        space_group_number = sga.get_space_group_number()
        emb_sg[space_group_number] = 1
        get_crystal_system = sga.get_crystal_system()
        emb_cs[convert_struct[get_crystal_system]] = 1
        return emb_sg, emb_cs, space_group_number, get_crystal_system

    def get_general_torch_data(self):
        emb_sg, emb_cs, _, _ = self.get_space_group()
        cn_emb, oms, _ = self.get_coordination_and_oms()
        atom_conc = self.get_species_conc()
        self.torch_data.atomic_one_hot = atom_conc
        self.torch_data.cordinates = cn_emb,
        self.torch_data.space_group_number = emb_sg
        self.torch_data.crystal_system = emb_cs
        self.torch_data.oms = torch.tensor([[oms]], dtype=torch.float)
        return self.torch_data

    def complete_torch_data(self):
        '''
        Get torch data
        '''
        emb_sg, emb_cs, _, _ = self.get_space_group()
        cn_emb, oms, metals = self.get_coordination_and_oms()
        atom_conc = self.get_species_conc()
        self.torch_data.atomic_one_hot = atom_conc
        self.torch_data.cordinates = cn_emb,
        self.torch_data.space_group_number = emb_sg
        self.torch_data.crystal_system = emb_cs
        self.torch_data.oms = torch.tensor([[oms]], dtype=torch.float)
        return self.torch_data.to(device), metals

    def predict_condition(self):
        '''
        predict conditions
        '''
        torch_data, _ = self.complete_torch_data()
        torch_data = torch_data.to(device)
        torch_data.cordinates = torch_data.cordinates[0]
        models = get_models(torch_data, device=device)
        models = models[0:1]
        category_names = filetyper.category_names()["metal_salts"]
        pred_list = ensemble_predictions(models, torch_data, category_names, device=device)

        energy_model = get_model_energy(torch_data, device=device)
        energy_prediction = get_energy_prediction(energy_model, torch_data, device=device)

        # print("energy_prediction\t", energy_prediction[0])

        return pred_list, energy_prediction[0]

    def compile_data(self):
        data = []
        ligands_names = self.get_organic_ligands()
        _, _, space_group_number, get_crystal_system = self.get_space_group()
        pred_list, stability_energy = self.predict_condition()
        salts_data = [tmp[0] for tmp in pred_list[:3]]
        salt_solvents = self.get_metal_solvents(salts_data)
        ligand_solvents = self.get_ligands_solvents()

        data.append("\n")
        data.append(f"Predicted Synthetic Data Report\n")
        data.append(f"For: {self.basename}\n")
        data.append("=" * 80 + "\n")
        data.append(f"{'Space group number:':25} {space_group_number}\n")
        data.append(f"{'Crystal system:':25} {get_crystal_system}\n\n")
        data.append("=" * 80 + "\n")
        data.append(f"{'Thermodynamic Stability:':25} {stability_energy} kJ/mol\n")


        data.append("Organic Ligands\n")
        data.append("-" * 80 + "\n")
        data.append(f"{'InChIKey':<28} {'SMILES':<30} {'IUPAC Name':<20}\n")
        data.append("-" * 80 + "\n")
        for names in ligands_names:
            iupac, smi, inchi = names
            inchi_str = str(inchi) if inchi is not None else "N/A"
            smi_str = str(smi) if smi is not None else "N/A"
            iupac_str = str(iupac) if iupac is not None else "N/A"
            data.append(f"{inchi_str:<28} {smi_str:<30} {iupac_str:<20}\n")
        data.append("\n")

        data.append("Top 3 Predicted Metal Salts\n")
        data.append("-" * 80 + "\n")
        data.append(f"{'Metal Salt':<40} {'Probability':>15}\n")
        data.append("-" * 80 + "\n")
        for metal_salt, prob in pred_list[:3]:
            salt_str = str(metal_salt) if metal_salt is not None else "N/A"
            prob_str = f"{prob:.4f}" if prob is not None else "N/A"
            data.append(f"{salt_str:<40} {prob_str:>15}\n")
        data.append("=" * 80 + "\n")
        data.append("Percentage probability of solvents\n")
        data.append("-" * 80 + "\n")
        data.append("Organic ligands\n")
        for ligand in ligand_solvents:
            data.append("-" * 80 + "\n")
            data.append(f"Ligands : {ligand }\n")
            data.append("-" * 80 + "\n")
            data.append(f"{'Solvent':<40} {'Predicted probability':>15}\n")
            data.append("-" * 80 + "\n")
            solvents = ligand_solvents[ligand]
            if solvents == None:
                data.append(f"No predicted solvents. Maybe try water\n")
            else:
                for solvent_data in solvents[:3]:
                    solvent, prob = solvent_data
                    data.append(f"{solvent:<40} {prob:>15} %\n")
        data.append("=" * 80 + "\n")
        data.append("Metal Salts\n")
        for salts in salt_solvents:
            data.append("-" * 80 + "\n")
            data.append(f"Metal Salts: {salts}\n")
            data.append("-" * 80 + "\n")
            data.append(f"{'Solvent':<40} {'Predicted probability':>15}\n")
            data.append("-" * 80 + "\n")
            solvents = salt_solvents[salts]
            if solvents == None:
                data.append(f"No predicted solvents. Maybe try water\n")
            else:
                for solvent_data in solvents[:3]:
                    solvent, prob = solvent_data
                    data.append(f"{solvent:<40} {prob:>15} %\n")

        data.append("=" * 80 + "\n")
        data.append("\n")
        data.append("Report generated by fairmofsyncondition\n")
        data.append("Authors: Dinga Wonanke & Antonio Longa\n")
        data.append("=" * 80 + "\n")

        filetyper.put_contents(self.outfile, data)
        print(f" Predicted data has been written in {self.outfile}")
        print(f" Thank you for using fairsyncondition")


def print_helpful_information():
    '''
    Prints helpful information about using the fairmofsyncondition script.
    '''
    help_text = """
    Usage: fairmofsyncondition.py [CIF_FILE] [OPTIONS]

    This script predicts synthetic conditions for MOFs given a CIF file
    or any ase readable fileformat.,
    extracts organic ligand information, space group, and top predicted
    metal salts, and writes a formatted report to a text file.

    Positional Arguments:
      CIF_FILE              Path to the input CIF file.

    Optional Arguments:
      -o, --output FILE     The path to the output report file (default: prediction_report.txt).

    Examples:
      fairmofsyncondition_syncon my_mof.cif -o my_mof_report.txt
      fairmofsyncondition_syncon ../data/sample.cif
    """
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="Predict synthetic conditions for MOFs from CIF files."
    )
    parser.add_argument(
        'cif_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to the CIF file to analyze.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='mof_prediction_report.txt',
        help='Path to the output report file (default: prediction_report.txt).'
    )

    args = parser.parse_args()

    if args.cif_file is None:
        print_helpful_information()
        sys.exit(1)

    try:
        print (f"Processing {args.cif_file}")
        print (f"Give it a few seconds\n")
        ml_data = Synconmodel(args.cif_file)
        ml_data.outfile = args.output
        ml_data.compile_data()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
