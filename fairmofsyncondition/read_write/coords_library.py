#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

import numpy as np
import re
from ase.io import read
from ase import Atoms, Atom
from orb_models.forcefield import atomic_system
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
    "Read coordinate using ase"
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
    Read coordinates from .gjf extension
    """
    qc_input = filetyper.get_contents(filename)
    file_lines = []
    for line in qc_input:
        file_lines.append(line.split())
    coords=[]
    lattice =[]
    ase_line = file_lines[2]
    if not 'ASE' in ase_line:
        for row in  file_lines[6:]:
            if len(row)>0:
                if not 'Tv' in row:
                    b = '\t'.join(row)
                    coords.append(b)
                else:
                    b = '\t'.join(row[1:])
                    lattice.append(b)
            else:
                break
    else:
        for row in  file_lines[5:]:
            if len(row)>0:
                if not 'TV' in row:
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
    read xyz coordinates
    """
    qc_input = filetyper.get_contents(filename)
    coords=[]
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
    """
    qc_input = filetyper.get_contents(filename)
    verdict = ''
    for line in qc_input:
        if 'Lattice vectors (angstrom)' in line:
            verdict = 'True'
            break
    return verdict

def scm_out(qcin):
    """
    Extract coordinates from scm output files
    """
    qc_input = filetyper.get_contents(qcin)
    verdict = check_periodicity(qcin)
    coords=[]
    lattice_coords =[]
    lattice=[]
    length_value = []

    if verdict == 'True':
        cods = filetyper.get_section(qc_input, 'Index Symbol   x (angstrom)   y (angstrom)   z (angstrom)', 'Lattice vectors (angstrom)', 1, -2)

        for  lines in cods:
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

        for  line in parameters:
            a = line[1:]
            if len(a) > 2:
                b = '\t'.join(a)
                lattice_coords.append(b)

    else:
        cods = filetyper.get_section(qc_input, 'Index Symbol   x (angstrom)   y (angstrom)   z (angstrom)', 'Total System Charge', 1, -2)
        for  lines in cods:
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
    """
    qc_input = filetyper.get_contents(filename)
    cods = filetyper.get_section(qc_input, 'OPTIMIZATION CONVERGED', 'Z-matrix Print:', 5, -2)
    #cods = filetyper.get_section(qc_input, '$molecule', '$end', 2, -1)
    coords=[]
    for row in cods:
        data = row.split()
        b = '\t'.join(data[1:])
        coords.append(b)
    return coords

def qchemin(filename):
    """
    Read coordinates from qchem input file
    """
    qc_input = filetyper.get_contents(filename)
    coords = filetyper.get_section(qc_input, '$molecule', '$end', 2, -1)
    return coords

def format_coords(coords, atom_labels):
    """
    create coords containing symbols and positions
    """
    coordinates =[]
    #file_obj.write('%d\n\n' %len(atom_types))
    for labels,row in zip(atom_labels,coords):
        b = [labels] + [str(atom)+' ' for atom in row]
        printable_row = '\t'.join(b)
        coordinates.append(printable_row + '\n')
    return coordinates


def coordinate_definition(filename):
    """
    define how coordinates should be extracted
    """
    print (filename)
    #Robust algorithm for finding file extention (check)
    iter_index = re.finditer(r'\.', filename)
    check = [filename[i.span()[0]+1:] for i in iter_index  ][-1]
    coords, lattice = [],[]
    #check = filename.split('.')[1]
    if check == 'gjf':
        coords, lattice =  gjf_coordinate(filename)
    elif check =='xyz':
        coords =xyz_coordinates(filename)
    elif check =='out':
        coords, lattice = scm_out(filename)
    elif check =='cout':
        coords= qchemcout(filename)
    elif check =='cin':
        coords= qchemin(filename)
    else:
        coords, lattice = ase_coordinate(filename)

    return coords, lattice

def collect_coords(filename):
    '''
    Collect coordinates
    '''
    coords, lattice = coordinate_definition(filename)
    elements = []
    positions =[]
    cell = []
    for lines in coords:
        data = lines.split()
        elements.append(data[0])
        positions.append([float(i) for i in data[1:]])

    positions = np.array(positions)

    if len(lattice)!=0:
        cell = np.array([[float(i) for i in j.split()] for j in lattice])

    return elements, positions, cell

def load_data_as_ase(filename):
    """
    Load data as an ase atoms object
    """
    elements, positions, cell = collect_coords(filename)
    ase_atoms = Atoms(symbols=elements, positions=positions)
    if len(cell) > 0:
        ase_atoms = Atoms(symbols=elements, positions=positions, cell=cell, pbc=True)
    return ase_atoms


def ase_graph(input_system):
    """
    Create a graph from an ase atoms object
    **parameter**
        **input_system** : Atoms or Atom object or meolcular file name e.g molecule.xyz or mof.cif
    **return**
        graph object: ase graph object
    """
    if isinstance(input_system, Atoms) or isinstance(input_system, Atom):
        graph = atomic_system.ase_atoms_to_atom_graphs(input_system)
    else:
        ase_atoms = load_data_as_ase(input_system)
        graph = atomic_system.ase_atoms_to_atom_graphs(ase_atoms)
    return graph


def xtb_input(filename):
    """
    Create an xtb input
    """
    elements, positions, cell = collect_coords(filename)
    xtb_coords = []
    # xtb_coords.append('> cat coord \n')
    xtb_coords.append('$coord angs\n')
    for labels,row in zip(elements, positions):
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
    Create an xtb input from an ase atom
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
    for labels,row in zip(elements, positions):
        tmp_coord =  [str(atom)+ ' ' for atom in row] + [' '] + [labels]
        xtb_coords.append('\t'.join(tmp_coord) +'\n')
    if len(ase_cell) > 0:
        xtb_coords.append('$periodic cell vectors \n')
        # xtb_coords.append('$lattice angs \n')
        for lattice in ase_cell :
            tmp_lattice =  [str(lat)+ ' ' for lat in lattice] + [' ']
            xtb_coords.append('\t'.join( tmp_lattice) +'\n')
    xtb_coords.append('$end')
    return xtb_coords
