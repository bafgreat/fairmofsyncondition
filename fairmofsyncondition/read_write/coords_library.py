#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

import os
import re
import random
import pickle
import torch
import lmdb
import numpy as np
from ase.io import read
from ase import Atoms, Atom
from ase.db import connect
from torch_geometric.data import Data
from mofstructure import mofdeconstructor
# from orb_models.forcefield import atomic_system
from fairmofsyncondition.read_write import filetyper


def read_and_return_ase_atoms(filename):
    """
    Function to read the ase atoms

    **parameter**
        filename: string
    """
    ase_atoms = read(filename)
    return ase_atoms


def write_ase_atoms(ase_atoms, filename):
    """
    Function to write the ase atoms

    **parameter**
        ase_atoms: ase.Atoms object
        filename: string
    """
    ase_atoms.write(filename)


def ase_coordinate(filename):
    """
    Read any ASE readable file and returns coordinates and lattices
    which should be use for setting up AMS calculations.

    **parameter**
        filename  (string) : Name of file containing the coordinate

    **Returns**
        ase_coord (list) : List of coordinate strings
        lattice (list) : List of lattice vectors strings
    """
    molecule = read(filename)
    atoms = Atoms(molecule)
    ase_cell = atoms.get_cell(complete=True)
    elements = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    ase_coord = []
    for ele, xyz in zip(elements, positions):
        cods = '\t'.join([ele]+[str(i) for i in xyz])
        ase_coord.append(cods)
    lattice = []
    for i in range(3):
        a = [' '] + [str(i) for i in ase_cell[i]]
        b = '\t'.join(a)
        lattice.append(b)
    return ase_coord, lattice


def gjf_coordinate(filename):
    """
    Reading coordinates from a gaussian .gjf file

    **parameter**
        filename  (string) : Name of file containing the coordinate

    **Returns**
        coords : List of coordinate strings
        lattice (list) : List of lattice vectors strings
    """
    qc_input = filetyper.get_contents(filename)
    file_lines = []
    for line in qc_input:
        file_lines.append(line.split())

    coords = []
    lattice = []
    ase_line = file_lines[2]
    if 'ASE' not in ase_line:
        for row in file_lines[6:]:
            if len(row) > 0:
                if 'Tv' not in row:
                    b = '\t'.join(row)
                    coords.append(b)
                else:
                    b = '\t'.join(row[1:])
                    lattice.append(b)
            else:
                break
    else:
        for row in file_lines[5:]:
            if len(row) > 0:
                if 'TV' not in row:
                    b = '\t'.join(row)
                    coords.append(b)
                else:
                    b = '\t'.join(row[1:])
                    lattice.append(b)
            else:
                break

    return coords, lattice


def xyz_coordinates(filename):
    """
    Read any xyz coordinate file

    **parameter**
        filename  (string) : Name of file containing the coordinate

    **Returns**
        coords : List of coordinate strings
    """
    qc_input = filetyper.get_contents(filename)
    coords = []
    file_lines = []
    for line in qc_input:
        file_lines.append(line.split())
    for row in file_lines[2:]:
        a = [' '] + row
        b = '\t'.join(a)
        coords.append(b)
    return coords


def check_periodicity(filename):
    """
    Function to check periodicity in an scm output file

    **parameter**
        filename  (string) : Name of file containing the coordinate
    """
    qc_input = filetyper.get_contents(filename)
    verdict = False
    for line in qc_input:
        if 'Lattice vectors (angstrom)' in line:
            verdict = True
            break
    return verdict


def scm_out(qcin):
    """
    Extract coordinates from scm output files

    **parameter**
        qcin  (string) : scm output file

    **return**
        coords (list) : list of coordinates
        lattice_coords (list) : list of lattice coordinates
    """
    qc_input = filetyper.get_contents(qcin)
    verdict = check_periodicity(qcin)
    coords = []
    lattice_coords = []
    lattice = []
    length_value = []
    if verdict:
        cods = filetyper.get_section(qc_input, 'Index Symbol   x (angstrom)   y (angstrom)   z (angstrom)', 'Lattice vectors (angstrom)', 1, -2)

        for lines in cods:
            data = lines.split()
            length_value.append(data[0])
            b = '\t'.join(data[1:])
            coords.append(b)
        lat_index = 0
        for i, line in enumerate(qc_input):
            data = line.split()
            lattice.append(data)
            if 'Lattice vectors (angstrom)' in line:
                lat_index = i

        parameters = [lattice[lat_index+1], lattice[lat_index+2], lattice[lat_index+3]]

        for line in parameters:
            a = line[1:]
            if len(a) > 2:
                b = '\t'.join(a)
                lattice_coords.append(b)

    else:
        cods = filetyper.get_section(qc_input, 'Index Symbol   x (angstrom)   y (angstrom)   z (angstrom)', 'Total System Charge', 1, -2)
        for lines in cods:
            data = lines.split()
            length_value.append(data[0])
            b = '\t'.join(data[1:])
            coords.append(b)
        # length = str(len(length_value))
        lattice_coords = ['']
    return coords, lattice_coords


def qchemcout(filename):
    """
    Read coordinates from qchem output file

    **parameter**
        filename  (string) : Name of file containing the coordinate

    **Returns**
        coords : List of coordinate strings
    """
    qc_input = filetyper.get_contents(filename)
    cods = filetyper.get_section(qc_input, 'OPTIMIZATION CONVERGED', 'Z-matrix Print:', 5, -2)
    # cods = filetyper.get_section(qc_input, '$molecule', '$end', 2, -1)
    coords = []
    for row in cods:
        data = row.split()
        b = '\t'.join(data[1:])
        coords.append(b)
    return coords


def qchemin(filename):
    """
    Read coordinates from qchem input file

    **parameter**
        filename (string) : filename

    **Returns**
        coords : list of coordinate strings
    """
    qc_input = filetyper.get_contents(filename)
    coords = filetyper.get_section(qc_input, '$molecule', '$end', 2, -1)
    return coords


def format_coords(coords, atom_labels):
    """
    create coords containing symbols and positions

    **parameters**
        coords (list) : list of coordinates
        atom_labels (list) : list of atom labels

    **returns**
        coordinates (list) : list of formatted coordinates
    """

    coordinates = []
    # file_obj.write('%d\n\n' %len(atom_types))
    for labels, row in zip(atom_labels, coords):
        b = [labels] + [str(atom)+' ' for atom in row]
        printable_row = '\t'.join(b)
        coordinates.append(printable_row + '\n')
    return coordinates


def coordinate_definition(filename):
    """
    define how coordinates should be extracted
    """
    # print (filename)
    # Robust algorithm for finding file extention (check)
    iter_index = re.finditer(r'\.', filename)
    check = [filename[i.span()[0]+1:] for i in iter_index][-1]
    coords, lattice = [], []
    # check = filename.split('.')[1]
    if check == 'gjf':
        coords, lattice = gjf_coordinate(filename)
    elif check == 'xyz':
        coords = xyz_coordinates(filename)
    elif check == 'out':
        coords, lattice = scm_out(filename)
    elif check == 'cout':
        coords = qchemcout(filename)
    elif check == 'cin':
        coords = qchemin(filename)
    else:
        coords, lattice = ase_coordinate(filename)

    return coords, lattice


def collect_coords(filename):
    '''
    Collect coordinates

    **parameters**
        filename (string) : filename

    **returns**
        elements (list) : list of elements
        positions (numpy array) : numpy array of positions
        cell (numpy array) : numpy array of cell parameters if present in the file
    '''
    coords, lattice = coordinate_definition(filename)
    elements = []
    positions = []
    cell = []
    for lines in coords:
        data = lines.split()
        elements.append(data[0])
        positions.append([float(i) for i in data[1:]])

    positions = np.array(positions)

    if len(lattice) != 0:
        cell = np.array([[float(i) for i in j.split()] for j in lattice])

    return elements, positions, cell


def load_data_as_ase(filename):
    """
    Load data as an ase atoms object
    **parameter**
        filename (string) : Any file type that has been defined in this module
                            including ase readable filetypes
    **return**
        ase_atoms : ase atoms object
    """
    elements, positions, cell = collect_coords(filename)
    ase_atoms = Atoms(symbols=elements, positions=positions)
    if len(cell) > 0:
        ase_atoms = Atoms(symbols=elements, positions=positions, cell=cell, pbc=True)
    return ase_atoms


# def ase_graph(input_system):
#     """
#     Create a graph from an ase atoms object

#     **parameter**
#         **input_system** : Atoms or Atom object or meolcular file name e.g molecule.xyz or mof.cif

#     **return**
#         graph object: ase graph object
#     """
#     if isinstance(input_system, Atoms) or isinstance(input_system, Atom):
#         graph = atomic_system.ase_atoms_to_atom_graphs(input_system)
#     else:
#         ase_atoms = load_data_as_ase(input_system)
#         graph = atomic_system.ase_atoms_to_atom_graphs(ase_atoms)
#     return graph


def xtb_input(filename):
    """
    Creating a gfn-xtb input file from any ase readable filetype or filetype
    that can be read by this module.

    **parameter**
        filename (string) : Any file type that has been defined in this module

    **return**
        xtb_coords : list of strings containing xtb input
    """
    elements, positions, cell = collect_coords(filename)
    xtb_coords = []
    # xtb_coords.append('> cat coord \n')
    xtb_coords.append('$coord angs\n')
    for labels, row in zip(elements, positions):
        tmp_coord =  [str(atom)+ ' ' for atom in row] + [' '] + [labels]
        xtb_coords.append('\t'.join(tmp_coord) +'\n')
    if len(cell) > 0:
        xtb_coords.append('$periodic ' +str(len(cell)) +'\n')
        xtb_coords.append('$lattice angs \n')
        for lattice in cell:
            lat_vector = '\t'.join(lattice) + '\n'
            xtb_coords.append(lat_vector)
    xtb_coords.append('$end')
    # xtb_coords.append('> xtb coord\n')
    return xtb_coords


def ase_to_xtb(ase_atoms):
    """
    Create a gfn-xtb input from an ase atom object.

    **parameter**
        ase_atoms (ase Atoms or Atom): The ase atoms object to be converted.

    **return**
        xtb_coords = ase_to_xtb_coords(ase_atoms)
    """
    check_pbc = ase_atoms.get_pbc()
    ase_cell = []
    xtb_coords = []
    if any(check_pbc):
        ase_cell = ase_atoms.get_cell(complete=True)
    elements = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()
    # xtb_coords.append('> cat coord \n')
    xtb_coords.append('$coord angs\n')
    for labels, row in zip(elements, positions):
        tmp_coord = [str(atom) + ' ' for atom in row] + [' '] + [labels]
        xtb_coords.append('\t'.join(tmp_coord) + '\n')
    if len(ase_cell) > 0:
        xtb_coords.append('$periodic cell vectors \n')
        # xtb_coords.append('$lattice angs \n')
        for lattice in ase_cell:
            tmp_lattice = [str(lat) + ' ' for lat in lattice] + [' ']
            xtb_coords.append('\t'.join(tmp_lattice) + '\n')
    xtb_coords.append('$end')
    return xtb_coords


def get_pairwise_connections(graph):
    """
    Extract unique pairwise connections from an adjacency dictionary efficiently.

    **Parameters**
        graph (dict):
            An adjacency dictionary where keys are nodes and values are arrays
            or lists of nodes representing neighbors.

    **returns**
        list of tuple
            A list of unique pairwise connections, each represented as a tuple (i, j) where i < j.

    """
    pairwise_connections = []
    seen = set()

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edge = (min(node, neighbor), max(node, neighbor))
            if edge not in seen:
                seen.add(edge)
                pairwise_connections.append(edge)
    return pairwise_connections


def calculate_distances(pair_indices, ase_atoms, mic=True):
    """
    Calculate distances between pairs of atoms in an ase atoms object.
    """
    return np.array([
        ase_atoms.get_distance(pair[0], pair[1], mic=mic)
        for pair in pair_indices])


def ase_to_pytorch_geometric(input_system):
    """
    Convert an ASE Atoms object to a PyTorch Geometric graph

    **parameters**
        input_system (ASE.Atoms or ASE.Atom or filename): The input system to be converted.

    **returns**
        torch_geometric.data.Data: The converted PyTorch Geometric Data object.
    """

    if isinstance(input_system, Atoms) or isinstance(input_system, Atom):
        ase_atoms = input_system
    else:
        ase_atoms = load_data_as_ase(input_system)
    mic = ase_atoms.pbc.any()
    if mic:
        lattice_parameters = torch.tensor(np.array(ase_atoms.cell), dtype=torch.float)
    else:
        lattice_parameters = torch.tensor(np.zeros(3, 3), dtype=torch.float)

    graph, _ = mofdeconstructor.compute_ase_neighbour(ase_atoms)
    pair_connection = np.array(get_pairwise_connections(graph))
    distances = calculate_distances(pair_connection, ase_atoms, mic)
    nodes = np.array([[atom.number, *atom.position] for atom in ase_atoms])

    node_features = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(pair_connection, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(distances, dtype=torch.float).unsqueeze(1)
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, lattice=lattice_parameters)
    return data


def pytorch_geometric_to_ase(data):
    """
    Convert a PyTorch Geometric Data object back to an ASE Atoms object.

    **Parameters**
        data (torch_geometric.data.Data): The PyTorch Geometric Data object.

    **Returns**
        ase_atoms (ase.Atoms): The converted ASE Atoms object.
    """
    node_features = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
    atomic_numbers = node_features[:, 0].astype(int)
    positions = node_features[:, 1:4]

    lattice = data.lattice.numpy() if isinstance(data.lattice, torch.Tensor) else data.lattice

    ase_atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=lattice,
        pbc=(lattice.any())
    )

    return ase_atoms


def prepare_dataset(ase_obj, energy):
    """
    Prepares a dataset from ASE Atoms objects and their corresponding energy values.

    **parameters**
        ase_obj (ASE.Atoms): ASE Atoms object.
        energy (float): Energy value of the crystal structure.

    **returns**
        torch_geometric.data.Data: PyTorch Geometric Data object with input features, edge indices, and energy value.
    """
    data = ase_to_pytorch_geometric(ase_obj)
    data.y = torch.tensor([energy], dtype=torch.float)
    return data


def data_from_aseDb(path_to_db, num_data=25000):
    """
    Load data from ASE database and prepare it for training.

    **parameters**
        path_to_db (str): Path to the ASE database file.

    **returns**
        list: List of PyTorch Geometric Data objects for training.
    """
    dataset = []
    counter = 0
    db = connect(path_to_db)
    for row in db.select():
        data = prepare_dataset(row.toatoms(), row.r_energy)
        dataset.append(data)
        if counter >= num_data:
            break
        counter += 1
    return dataset


def ase_database_to_lmdb(ase_database, lmdb_path):
    """
    Converts an ASE database into an LMDB file for
    efficient storage and retrieval.

    **parameter**
        ase_database (str): path to ase database.
        lmdb_path (str): Path to the LMDB file where
            the dataset will be saved.
    """
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    try:
        with connect(ase_database) as db:
            with lmdb.open(lmdb_path, map_size=int(1e12)) as lmdb_env:
                with lmdb_env.begin(write=True) as txn:
                    count = 0
                    for i, row in enumerate(db.select()):
                        data = prepare_dataset(row.toatoms(), row.r_energy)
                        txn.put(f"{i}".encode(), pickle.dumps(data))
                        count += 1
                    txn.put(b"__len__", pickle.dumps(count))
        print(f"Data successfully saved to {lmdb_path} with {count} entries.")
    except lmdb.Error as e:
        print(f"An error occurred with LMDB: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

class LMDBDataset:
    """
    A class for loading PyTorch data stored in an LMDB file. The code is originally
    intended for graph structured graphs that can work with pytorch_geometric data.
    But it should also load all types of PyTorch data.

    This class enables on-the-fly loading of serialized data stored in LMDB format,
    providing an efficient way to handle large datasets that cannot fit into memory.

    **parameters**
        lmdb_path (str): Path to the LMDB file containing the dataset.

    **Attributes**
        lmdb_env (lmdb.Environment): The LMDB environment for data access.
        length (int): The total number of entries in the dataset.

    **Methods**
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the specified index.
        split_data(train_size, random_seed, shuffle): Lazily returns train and test data.

    **Examples**

    This class provides an efficient way for loading huge datasets without
    consuming so much memory.

        data = coords_library.LMDBDataset(path_to_lmdb)

        Length of the dataset

        print(len(data))

        # Accessing a sample at index 0

        sample = data[0]

        print(sample.x.shape)

        print(sample)

        # Accessing a list of samples at different indexes

        samples = data[[1,4,8,9,18, 50]]

    """

    def __init__(self, lmdb_path):
        try:
            self.lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with self.lmdb_env.begin() as txn:
                length_data = txn.get(b"__len__")
                if length_data is None:
                    raise ValueError(
                        f"The LMDB file at '{lmdb_path}' does not contain the key '__len__'. "
                        f"Ensure the data was saved correctly and includes this key."
                    )
                self.length = pickle.loads(length_data)
        except ValueError as ve:
            raise RuntimeError(
                f"ValueError: {ve}\nCheck if the LMDB file is correctly created with a '__len__' key."
            ) from ve
        except lmdb.Error as le:
            raise RuntimeError(
                f"An LMDB error occurred while accessing the file at '{lmdb_path}': {le}"
            ) from le
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while initializing the dataset: {e}"
            ) from e

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
            int: The number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves the sample(s) at the specified index or indices from the LMDB file.

        **parameters**
            idx (int or list of int): The index or indices of the sample(s) to retrieve.

        **Returns**
            Any or list: The deserialized data corresponding to the specified index/indices.
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= self.length:
                raise IndexError(
                    f"Index {idx} is out of range for dataset of size {self.length}."
                )
            with self.lmdb_env.begin() as txn:
                data = txn.get(f"{idx}".encode())
                if data is None:
                    raise ValueError(
                        f"No data found for index {idx}. Ensure the dataset is correctly saved."
                    )
                return pickle.loads(data)
        elif isinstance(idx, list) or isinstance(idx, np.ndarray) or isinstance(idx, tuple):
            results = []
            for i in idx:
                if i < 0 or i >= self.length:
                    raise IndexError(
                        f"Index {i} is out of range for dataset of size {self.length}."
                    )
                with self.lmdb_env.begin() as txn:
                    data = txn.get(f"{i}".encode())
                    if data is None:
                        raise ValueError(
                            f"No data found for index {i}. Ensure the dataset is correctly saved."
                        )
                    results.append(pickle.loads(data))
            return results
        else:
            raise TypeError(
                f"Index must be an int or list, or nd.array or tuple."
            )

    def split_data(self, train_size=0.8, random_seed=None, shuffle=True):
        """
        Lazily splits the dataset into train and test data with class-like behavior.

        Args:
            train_size (float): The proportion of the data to be used as the training set (default is 0.8).
            random_seed (int, optional): A random seed for reproducibility (default is None).
            shuffle (bool): Whether to shuffle the data before splitting (default is True).

        Returns:
            tuple: A tuple containing train data and test data.
        """
        indices = list(range(self.length))

        if random_seed is not None:
            random.seed(random_seed)

        if shuffle:
            random.shuffle(indices)

        split_index = int(self.length * train_size)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        class Subset:
            def __init__(self, parent, indices):
                self.parent = parent
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                if isinstance(idx, int):
                    return self.parent[self.indices[idx]]
                elif isinstance(idx, list) or isinstance(idx, np.ndarray) or isinstance(idx, tuple):
                    return [self.parent[self.indices[i]] for i in idx]
                else:
                    raise TypeError("Index must be an int or list, or nd.array or tuple.")

        train_data = Subset(self, train_indices)
        test_data = Subset(self, test_indices)

        return train_data, test_data


def list_train_test_split(data, train_size=0.8, random_seed=42, shuffle=True):
    """
    A function that take Splits a list into train and test sets based on the specified train_size.

    **parameter**
        data (list): The input list to split.
        train_size (float): The proportion of the data to be used as the training set (default is 0.8).
        random_seed (int, optional): A random seed for reproducibility (default is None).
        shuffle (bool): Whether to shuffle the data before splitting (default is True).

    **return**
        train_data: indices of data to be selected for training.
        test_data: indices of data to be selected for testing.
    """
    if random_seed is not None:
        random.seed(random_seed)

    if shuffle:
        data = data.copy()
        random.shuffle(data)

    split_index = int(len(data) * train_size)
    train_data = data[:split_index]
    test_data = data[split_index:]

    return train_data, test_data
