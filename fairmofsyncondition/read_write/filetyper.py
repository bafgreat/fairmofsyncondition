#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"
import os
import pickle
import csv
import json
import re
import codecs
from zipfile import ZipFile
import numpy as np
import pandas as pd
from ase import Atoms
import ase
# import mmap
# from mmap_json import mmap_loads

class AtomsEncoder(json.JSONEncoder):
    '''
    ASE atom type encorder for json to enable serialising
    ase atom object.
    '''

    def default(self, encorder_obj):
        '''
        define different encoder to serialise ase atom objects
        '''
        if isinstance(encorder_obj, Atoms):
            coded = dict(positions=[list(pos) for pos in encorder_obj.get_positions()], lattice_vectors=[
                         list(c) for c in encorder_obj.get_cell()], labels=list(encorder_obj.get_chemical_symbols()))
            if len(encorder_obj.get_cell()) == 3:
                coded['periodic'] = ['True', 'True', 'True']
            coded['n_atoms'] = len(list(encorder_obj.get_chemical_symbols()))
            coded['atomic_numbers'] = encorder_obj.get_atomic_numbers().tolist()
            keys = list(encorder_obj.info.keys())
            if 'atom_indices_mapping' in keys:
                info = encorder_obj.info
                coded.update(info)
            return coded
        if isinstance(encorder_obj, ase.spacegroup.Spacegroup):
            return encorder_obj.todict()
        return json.JSONEncoder.default(self, encorder_obj)

def json_to_aseatom(data, filename):
    '''
    serialise an ase atom type and write as json
    '''
    encoder = AtomsEncoder
    with open(filename, 'w', encoding='utf-8') as f_obj:
        json.dump(data, f_obj, indent=4, sort_keys=False, cls=encoder)
    return


def append_json_atom(data, filename):
    '''
    append a data containing an ase atom object
    '''
    encoder = AtomsEncoder
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f_obj:
            f_obj.write('{}')
    elif os.path.getsize(filename) == 0:
        with open(filename, 'w', encoding='utf-8') as f_obj:
            f_obj.write('{}')
    with open(filename, 'r+', encoding='utf-8') as f_obj:
        # First we load existing data into a dict.
        file_data = json.load(f_obj)
        # Join new_data with file_data inside emp_details
        file_data.update(data)
        # Sets file's current position at offset.
        f_obj.seek(0)
        # convert back to json.

        json.dump(data, f_obj, indent=4, sort_keys=True, cls=encoder)


def numpy_to_json(ndarray, file_name):
    '''
    Serialise a numpy object
    '''
    json.dump(ndarray.tolist(), codecs.open(file_name, 'w',
              encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    return


def list_2_json(list_obj, file_name):
    '''
    write a list to json
    '''
    json.dump(list_obj, codecs.open(file_name, 'w', encoding='utf-8'))

def write_json(json_obj, file_name):
    '''
    write a python dictionary object to json
    '''
    # Serializing json
    json_object = json.dumps(json_obj, indent=4, sort_keys=True)
    with open(file_name, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)

def json_to_numpy(json_file):
    '''
    serialised a numpy array to json
    '''
    json_reader = codecs.open(json_file, 'r', encoding='utf-8').read()
    json_reader = np.array(json.loads(json_reader))
    return read_json


def append_json(new_data, filename):
    '''
    append a new data in an existing json file
    '''
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write('{}')
    elif os.path.getsize(filename) == 0:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write('{}')
    with open(filename, 'r+', encoding='utf-8') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Overwrite existing keys with new_data
        file_data.update(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4, sort_keys=True)

def read_json(file_name):
    '''
    load a json file
    '''
    with open(file_name, 'r', encoding='utf-8') as f_obj:
        data = json.load(f_obj)

    return data


def csv_read(csv_file):
    '''
    Read a csv file
    '''
    f_obj = open(csv_file, 'r', encoding='utf-8')
    data = csv.reader(f_obj)
    return data


def get_contents(filename):
    '''
    Read a file and return a list content
    '''
    with open(filename, 'r', encoding='utf-8') as f_obj:
        contents = f_obj.readlines()
    return contents


def put_contents(filename, output):
    '''
    write a list object into a file
    '''
    with open(filename, 'w', encoding='utf-8') as f_obj:
        f_obj.writelines(output)
    return


def append_contents(filename, output):
    '''
    append contents into a file
    '''
    with open(filename, 'a', encoding='utf-8') as f_obj:
        f_obj.writelines(output)
    return


def save_pickle(model, file_path):
    '''
    write to a pickle file
    '''
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def append_pickle(new_data, filename):
    '''
    append to a pickle file
    '''
    with open(filename, 'ab') as f_:
        pickle.dump(new_data, f_)
    f_.close()


def pickle_load(filename):
    '''
    load a pickle file
    '''
    data = open(filename, 'rb')
    data = pickle.load(data)
    return data


def read_zip(zip_file):
    '''
    read a zip file
    '''
    content = ZipFile(zip_file, 'r')
    content.extractall(zip_file)
    content.close()
    return content

def remove_trailing_commas(json_file):
    '''
    Function to clean training commas in json files.
    It function reads a json file then returns the cleaned up file
    '''
    with open(json_file, 'r') as file:
        json_string = file.read()
    trailing_object_commas_re = re.compile(r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    trailing_array_commas_re = re.compile(r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    objects_fixed = trailing_object_commas_re.sub("}", json_string)
    return trailing_array_commas_re.sub("]", objects_fixed)


def query_data(ref, data_object, col=None):
    '''
    Function to query data from a csv or json
    '''
    if isinstance(data_object, dict):
        return data_object[ref]
    else:
        return data_object.loc[data_object[col] == ref]


def combine_json_files(file1_path, file2_path, output_path):
    '''
    A function to combine two json  files
    '''
    with open(file1_path, 'r') as file1:
        data1 = json.load(file1)

    # Read data from the second file
    with open(file2_path, 'r') as file2:
        data2 = json.load(file2)

    # Combine the data from both files
    combined_data = {**data1, **data2}

    # Write the combined data to the output file
    with open(output_path, 'w') as output_file:
        json.dump(combined_data, output_file, indent=2)

# def read_json_mmap(filename):

#     with open(filename, 'r') as f:

#         mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#         json_data = mmap_loads(mmapped_file)

#         mmapped_file.close()
#     return json_data

def load_data(filename):
    '''
    function that recognises file extenion and chooses the correction
    function to load the data.
    '''
    file_ext = filename[filename.rindex('.')+1:]
    if file_ext == 'json':
        # file_size = os.path.getsize(filename)
        # if file_size > 10 * 1024 * 1024 * 1024:  # 10GB in bytes
        #     data = read_json_mmap(filename)
        # else:
        data = read_json(filename)
    elif file_ext == 'csv':
        data = pd.read_csv(filename)
    elif file_ext == 'p' or file_ext == 'pkl':
        data = pickle_load(filename)
    elif file_ext == 'xlsx':
        data = pd.read_excel(filename)
    else:
        data = get_contents(filename)
    return data
