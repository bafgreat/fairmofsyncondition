import os
import re
import argparse
import torch
from ase import Atoms
import torch.nn as nn
import numpy as np
from ase.io import read
from mofstructure.structure import MOFstructure
from mofstructure import mofdeconstructor
from fairmofsyncondition.read_write import filetyper, coords_library
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

convert_struct = {'cubic': 0,
                  'hexagonal': 1,
                  'monoclinic': 2,
                  'orthorhombic': 3,
                  'tetragonal': 4,
                  'triclinic': 5,
                  'trigonal': 6
                  }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example layers (replace with your real architecture)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Synconmodel(object):
    def __init__(self,
                 ase_atom,
                 outfile="mof_prediction.txt",
                 module="best_model.pth",
                 ):
        """
        A class to run the metal salt prediction GNN model and for
        extracting organic ligands and predicting metal salts.
        """
        if isinstance(ase_atom, Atoms):
            self.ase_atom = ase_atom
        elif isinstance(ase_atom, str) and os.path.isfile(ase_atom):
            self.ase_atom = read(ase_atom)
        else:
            print("sorry input type is not recorgnised")
        self.outfile = outfile
        self.gnn_model = module
        self.structure = MOFstructure(self.ase_atom)
        self.pymat = AseAtomsAdaptor.get_structure(self.ase_atom)
        self.torch_data = coords_library.ase_to_pytorch_geometric(self.ase_atom)
        self.category = filetyper.category_names()

    def convert_metals(self):
        '''
        return metals dict as a dictionary of symbols with index
        '''
        return {j: i for i, j in
                enumerate(mofdeconstructor.transition_metals()[1:])}

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

    def get_space_group(self):
        "Get space group embedding"
        emb_sg = torch.zeros(231)
        emb_cs = torch.zeros(7)
        sga = SpacegroupAnalyzer(self.pymat)
        space_group_number = sga.get_space_group_number()
        emb_sg[space_group_number] = 1
        get_crystal_system = sga.get_crystal_system()
        emb_cs[convert_struct[get_crystal_system]] = 1
        return emb_sg, emb_cs

    def complete_torch_data(self):
        '''
        Get torch data
        '''
        emb_sg, emb_cs = self.get_space_group()
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
        torch_data, metals = self.complete_torch_data()
        model = MyModel()
        state_dict = torch.load("best_model.pth", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():        # no gradient tracking during inference
            y_pred = model(torch_data)
        print("Prediction:", y_pred.item())



        print(torch_data)
        print(metals)











ml_data = Synconmodel('../../tests/test_data/EDUSIF.cif')

print(ml_data.predict_condition())