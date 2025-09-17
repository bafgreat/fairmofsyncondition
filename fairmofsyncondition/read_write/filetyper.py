#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"
import os
import re
import pickle
from pathlib import Path
import csv
import json
import codecs
from zipfile import ZipFile
import ase
import numpy as np
import pandas as pd
from ase import Atoms
import msgpack
from importlib.resources import files


class AtomsEncoder(json.JSONEncoder):
    '''
    Custom JSON encoder for serializing ASE `Atoms` objects and related data.

    This encoder converts ASE `Atoms` objects into JSON-serializable dictionaries.
    It also handles the serialization of ASE `Spacegroup` objects.

    **Methods**
    default(obj)
        Serializes objects that are instances of ASE `Atoms` or `Spacegroup`,
        or falls back to the default JSON encoder for unsupported types.

    **Examples**
        >>> from ase import Atoms
        >>> import json
        >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0.76, 0], [0.76, 0, 0]])
        >>> json_data = json.dumps(atoms, cls=AtomsEncoder)
        >>> print(json_data)
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
    Serialize an ASE `Atoms` object and write it to a JSON file.
    This function uses the custom `AtomsEncoder` to convert an ASE `Atoms` object
    into a JSON format and writes the serialized data to the specified file.

    **parameters**
        data : Atoms or dict
            The ASE `Atoms` object or dictionary to serialize.
        filename : str
            The path to the JSON file where the serialized data will be saved.
    '''
    encoder = AtomsEncoder
    with open(filename, 'w', encoding='utf-8') as f_obj:
        json.dump(data, f_obj, indent=4, sort_keys=False, cls=encoder)
    return


def get_section(contents, start_key, stop_key, start_offset=0, stop_offset=0):
    """
    Extracts a section of lines from a list of strings between specified start and stop keys.
    This function searches through a list of strings (e.g., file contents) to find the last occurrence
    of a start key and extracts all lines up to and including the first occurrence of a stop key,
    with optional offsets for flexibility.

    **parameters**
        contents : list of str
            A list of strings representing the lines of a file or text content.
        start_key : str
            The key string that marks the start of the section.
        stop_key : str
            The key string that marks the end of the section.
        start_offset : int, optional
            The number of lines to include before the start key. Default is 0.
        stop_offset : int, optional
            The number of lines to include after the stop key. Default is 0.

    **returns**
        list of str
            The extracted lines from `contents` between the start and stop keys, including the offsets.
    """
    all_start_indices = []
    for i, line in enumerate(contents):
        if start_key in line:
            all_start_indices.append(i + start_offset)
    start_index = all_start_indices[-1]
    for i in range(start_index, len(contents)):
        line = contents[i]
        if stop_key in line:
            stop_index = i + 1 + stop_offset
            break
    data = contents[start_index:stop_index]
    return data


def append_json_atom(data, filename):
    '''
    Appends or updates a JSON file with data containing an ASE `Atoms` object.
    If the file does not exist or is empty, it creates a new JSON file with an empty dictionary
    as the initial content. The function then updates the file with the provided data using the
    custom `AtomsEncoder` for serializing ASE `Atoms` objects.

    **parameters**
        data : dict
            A dictionary containing data with an ASE `Atoms` object or other serializable content.
        filename : str
            The path to the JSON file where the data will be appended.
    '''
    encoder = AtomsEncoder
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f_obj:
            f_obj.write('{}')
    elif os.path.getsize(filename) == 0:
        with open(filename, 'w', encoding='utf-8') as f_obj:
            f_obj.write('{}')
    with open(filename, 'r+', encoding='utf-8') as f_obj:
        file_data = json.load(f_obj)
        file_data.update(data)
        f_obj.seek(0)

        json.dump(data, f_obj, indent=4, sort_keys=True, cls=encoder)


def numpy_to_json(ndarray, file_name):
    '''
    Serializes a NumPy array and saves it to a JSON file.
    This function converts a NumPy array into a list format, which is JSON-serializable,
    and writes it to the specified file.

    **parameters**
        ndarray : numpy.ndarray
            The NumPy array to serialize.
        file_name : str
            The path to the JSON file where the serialized data will be saved.
    '''
    json.dump(ndarray.tolist(), codecs.open(file_name, 'w',
              encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    return


def list_2_json(list_obj, file_name):
    '''
    Writes a list to a JSON file.
    This function serializes a Python list and saves it to a
    specified JSON file.

    **parameters**
        list_obj : list
            The list to serialize and write to the file.
        file_name : str
            The path to the JSON file where the list will be saved.
    '''
    json.dump(list_obj, codecs.open(file_name, 'w', encoding='utf-8'))


def write_json(json_obj, file_name):
    '''
    Writes a Python dictionary object to a JSON file.

    This function serializes a Python dictionary into JSON
    format and writes it to the specified file and ensures that the
    JSON is human-readable with proper indentation.

    **parameters**
        json_obj : dict
            The Python dictionary to serialize and write to the JSON file.
        file_name : str
            The path to the JSON file where the data will be saved.
    '''
    json_object = json.dumps(json_obj, indent=4, sort_keys=True)
    with open(file_name, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)


def json_to_numpy(json_file):
    '''
    Deserializes a JSON file containing a NumPy array
    back into a NumPy array. This function reads a JSON file,
    deserializes the data, and converts it into a NumPy array.

    **parameters**
        json_file : str
            The path to the JSON file containing the serialized NumPy array.

    **returns**
        numpy.ndarray
            The deserialized NumPy array.
    '''
    json_reader = codecs.open(json_file, 'r', encoding='utf-8').read()
    json_reader = np.array(json.loads(json_reader))
    return read_json


def append_json(new_data, filename):
    '''
    Appends new data to an existing JSON file. If the file does
    not exist or is empty, it creates a new JSON file with an
    empty dictionary. The function then updates the file with the
    provided data, overwriting existing keys if they are already present.

    **parameters**
        new_data : dict
            A dictionary containing the new data to append to the JSON file.
        filename : str
            The path to the JSON file.
    '''
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write('{}')
    elif os.path.getsize(filename) == 0:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write('{}')
    with open(filename, 'r+', encoding='utf-8') as file:
        file_data = json.load(file)
        file_data.update(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4, sort_keys=True)


def read_json(file_name):
    '''
    Loads and reads a JSON file. This function
    opens a JSON file, reads its content, and deserializes it into
    a Python object (e.g., a dictionary or list).

    **Parameters**
        file_name : str
            The path to the JSON file to be read.

    **returns**
        dict or list
            The deserialized content of the JSON file.
    '''
    with open(file_name, 'r', encoding='utf-8') as f_obj:
        data = json.load(f_obj)

    return data


def csv_read(csv_file):
    '''
    Reads a CSV file and returns its content as a list of rows. This function reads
    the content of a CSV file and returns it as a list.

    **parameters**
        csv_file : str
            The path to the CSV file to be read.

    **returns**
        list of list of str
            A list of rows from the CSV file. Each row is a list of strings.

    '''
    f_obj = open(csv_file, 'r', encoding='utf-8')
    data = csv.reader(f_obj)
    return data


def get_contents(filename):
    '''
    Reads the content of a file and returns it as a list of lines.
    This function opens a file, reads its content line by line,
    and returns a list where each element is a line from the file,
    including newline characters.

    **parameters**
        filename : str
            The path to the file to be read.

    **returns**
        list of str
            A list containing all lines in the file.
    '''
    with open(filename, 'r', encoding='utf-8') as f_obj:
        contents = f_obj.readlines()
    return contents


def put_contents(filename, output):
    '''
    Writes a list of strings into a file. This function writes the content of a list to a file, where each element
    in the list represents a line to be written. If the file already exists,
    it will be overwritten.

    **parameters**
        filename : str
            The path to the file where the content will be written.
        output : list of str
            A list of strings to be written to the file. Each string represents
            a line, and newline characters should be included if needed.
    '''
    with open(filename, 'w', encoding='utf-8') as f_obj:
        f_obj.writelines(output)
    return


def append_contents(filename, output):
    '''
    Appends a list of strings to a file. This function appends
    the content of a list to a file, where each element in the
    list represents a line to be written. If the file does not exist,
    it will be created.

    **parameters**
        filename : str
            The path to the file where the content will be appended.
        output : list of str
            A list of strings to be appended to the file. Each string represents
            a line, and newline characters should be included if needed.
    '''
    with open(filename, 'a', encoding='utf-8') as f_obj:
        f_obj.writelines(output)
    return


def save_pickle(model, file_path):
    '''
    Saves a Python object to a file using pickle. This function serializes
    a Python object and saves it to a specified file
    in binary format using the `pickle` module.

    **parameters**
        model : object
            The Python object to serialize and save.
        file_path : str
            The path to the file where the object will be saved.
    '''
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def append_pickle(new_data, filename):
    '''
    Appends new data to a pickle file. This function appends new data to an existing pickle file. If the file does not
    exist, it will be created. Data is appended in binary format, ensuring that
    previously stored data is not overwritten.

    **parameters**
        new_data : object
            The Python object to append to the pickle file.
        filename : str
            The path to the pickle file where the data will be appended.
    '''
    with open(filename, 'ab') as f_:
        pickle.dump(new_data, f_)
    f_.close()


def pickle_load(filename):
    '''
    Loads and deserializes data from a pickle file. This
    function reads a pickle file and deserializes its content into
    a Python object.

    **parameters**
        filename : str
            The path to the pickle file to be loaded.

    **returns**
        object
            The deserialized Python object from the pickle file
    '''
    data = open(filename, 'rb')
    data = pickle.load(data)
    return data


def read_zip(zip_file):
    '''
    Reads and extracts the contents of a zip file.

    This function opens a zip file and extracts its
    contents to the specified directory. If no directory
    is provided, it extracts to the current working
    directory.

    **parameters**
        zip_file : str
            The path to the zip file to be read and extracted.
        extract_to : str, optional
            The directory where the contents of the zip file will be extracted.
            If not provided, the current working directory is used.

    **returns**
        list of str
            A list of file names contained in the zip file.
    '''
    content = ZipFile(zip_file, 'r')
    content.extractall(zip_file)
    content.close()
    return content


def remove_trailing_commas(json_file):
    '''
    Cleans trailing commas in a JSON file and returns
    the cleaned JSON string. This function reads a JSON file,
    removes trailing commas from objects and arrays,
    and returns the cleaned JSON string. It is useful
    for handling improperly formatted JSON files with
    trailing commas that are not compliant with the JSON standard.

    **parameters**
        json_file : str
            The path to the JSON file to be cleaned.

    **returns**
        cleaned_json str
            A cleaned JSON string with trailing commas removed.

    '''
    with open(json_file, 'r', encoding='utf-8') as file:
        json_string = file.read()

    trailing_object_commas_re = re.compile(r',(?!\s*?[\{\[\"\'\w])\s*}')
    trailing_array_commas_re = re.compile(r',(?!\s*?[\{\[\"\'\w])\s*\]')

    objects_fixed = trailing_object_commas_re.sub("}", json_string)
    cleaned_json = trailing_array_commas_re.sub("]", objects_fixed)

    return cleaned_json


def query_data(ref, data_object, col=None):
    '''
    Queries data from a CSV (as a DataFrame) or JSON (as a dictionary).

    This function retrieves data based on a reference key or value from either
    a dictionary (JSON-like object) or a pandas DataFrame (CSV-like object).

    **parameters**
        ref : str or int
            The reference key or value to query.
        data_object : dict or pandas.DataFrame
            The data source, which can be a dictionary (for JSON) or a pandas
            DataFrame (for CSV).
        col : str, optional
            The column name to query in the DataFrame. This parameter is
            required if the data source is a DataFrame and ignored if
            the data source is a dictionary.

    **returns**
        object
            The queried data. For a dictionary, it returns the value
            associated with the reference key.
            For a DataFrame, it returns the rows
            where the specified column matches the reference value.
    '''
    if isinstance(data_object, dict):
        return data_object[ref]
    else:
        return data_object.loc[data_object[col] == ref]


def combine_json_files(file1_path, file2_path, output_path):
    '''
    Queries data from a CSV (as a DataFrame) or JSON (as a dictionary).

    This function retrieves data based on a reference key or value from either
    a dictionary (JSON-like object) or a pandas DataFrame (CSV-like object).

    **parameters**
        ref : str or int
            The reference key or value to query.
        data_object : dict or pandas.DataFrame
            The data source, which can be a dictionary (for JSON) or a pandas
            DataFrame (for CSV).
        col : str, optional
            The column name to query in the DataFrame. This parameter is
            required if the data source
            is a DataFrame and ignored if the data source is a dictionary.

    **returns**
        object
            The queried data. For a dictionary, it returns the value
            associated with the reference key.
            For a DataFrame, it returns the rows where the specified
            column matches the reference value.
    '''
    with open(file1_path, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    with open(file2_path, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    combined_data = {**data1, **data2}

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(combined_data, output_file, indent=2)


def save_dict_msgpack(data: dict, filename: str) -> None:
    """Save a dictionary to a file using MessagePack."""
    with open(filename, "wb") as f:
        msgpack.pack(data, f, use_bin_type=True)


def load_dict_msgpack(filename: str) -> dict:
    """Load a dictionary from a MessagePack file."""
    with open(filename, "rb") as f:
        return msgpack.unpack(f, raw=False, strict_map_key=False)


def load_data(filename):
    '''
    Automatically detects the file extension and loads the data using the
    appropriate function. This function reads a file and returns
    its content, choosing the correct loading method based on the file
    extension. Supported file formats include JSON, CSV, Pickle, Excel,
    and plain text files.

    **parameters**
        filename : str
            The path to the file to be loaded.

    **returns**
        object
            The loaded data, which can be a dictionary, DataFrame, list, or other Python object,
            depending on the file type.
    '''
    filename = Path(filename)
    file_ext = filename.suffix.lstrip('.')
    if file_ext == 'json':
        data = read_json(filename)
    elif file_ext == 'csv':
        data = pd.read_csv(filename)
    elif file_ext == 'p' or file_ext == 'pkl':
        data = pickle_load(filename)
    elif file_ext == 'xlsx':
        data = pd.read_excel(filename)
    elif file_ext == 'msgpack':
        data = load_dict_msgpack(filename)
    else:
        data = get_contents(filename)
    return data


def category_names():
    "load category"
    msgpack_path = files("fairmofsyncondition").joinpath("db/category_names.msgpack")
    return load_data(msgpack_path)


def smile_names_iupac():
    "load iupac"
    msgpack_path = files("fairmofsyncondition").joinpath("db/iupacname_smiles.msgpack")
    return load_data(msgpack_path)

def solvent_and_inchi():
    "load solvent names and inchi"
    solven_path = files("fairmofsyncondition").joinpath("db/solvent_to_inchi_and_smile.json")
    return load_data(solven_path)

def ligand2solvent():
    "load solvent names and inchi"
    solven_path = files("fairmofsyncondition").joinpath("db/ligands_to_solvents.json")
    return load_data(solven_path)

def salt2solvent():
    "load solvent names and inchi"
    solven_path = files("fairmofsyncondition").joinpath("db/metal_salts_to_solvents.json")
    return load_data(solven_path)