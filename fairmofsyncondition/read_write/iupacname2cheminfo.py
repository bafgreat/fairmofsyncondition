#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

##############################################################################
# fairmofsyncondition is a machine learning package for predicting the        #
# synthesis condition of the crystal structures of MOFs. It is also intended  #
# for predicting all MOFs the can be generated from a given set of conditions #
# In addition the package also predicts the stability of MOFs, compute their  #
# their PXRD and crystallite sizes. This package is part of our effort to     #
# to accelerate the discovery and optimization of the synthesises of novel    #
# high performing MOFs. This package is being developed by Dr Dinga Wonanke   #
# as part of hos MSCA post doctoral fellowship at TU Dresden.                 #
#                                                                             #
###############################################################################
import pandas as pd
import argparse
import requests
import pubchempy as pcp


def opsin_name_to_smile(chemical_name):
    '''
    Function that uses opsin api to convert iupac names
    to cheminformatic identifiers

    **parameters**:
        chemical_name (str): The iupac name of the chemical.
    **returns**:
        smiles (str): The chemical structure in smiles format.
        inchi (str): The chemical structure in inchi format.
    '''
    opsin_url = 'https://opsin.ch.cam.ac.uk/opsin/'
    response = requests.get(opsin_url + chemical_name)
    print (response)
    output = response.json()
    if output['status'] == 'SUCCESS':
        return output
    else:
        return None


def pubchem_to_inchikey(identifier):
    '''
    Function that uses pubchem api to convert chemical names
    to inchi keys and inchi

    **parameters**:
        identifier (str): The name of the chemical.
    **returns**:
        inchikey (str): The inchi key of the chemical.
        smiles (str): The chemical structure in smiles format.
    '''
    pubchem = pcp.get_compounds(identifier, 'name')
    if len(pubchem) > 0:
        return pubchem[0].inchikey, pubchem[0].canonical_smiles
    else:
        return None


def name_to_cheminfo(chemical_name):
    """
    Converts a chemical name into cheminformatic identifiers
    using OPSIN and PubChem APIs.

    This function attempts to retrieve the InChIKey and SMILES
    representation of a
    given chemical name. It first tries to use the OPSIN API to fetch the data.
    If OPSIN does not return a result, it falls back to using the PubChem API.

    Parameters:
        chemical_name (str): The name of the chemical to be converted.
                             Can be an IUPAC name or other
                             recognized chemical name.

    Returns:
        dict: A dictionary containing the following keys:
              - 'inchikey': The InChIKey of the chemical (str),
              or None if not found.
              - 'smiles': The SMILES representation
              of the chemical (str), or None if not found.

              If no information is found from either API,
              returns an empty dictionary.
    """
    tmp = {}
    data = opsin_name_to_smile(chemical_name)
    if data != None:
        tmp['inchikey'] = data['stdinchikey']
        tmp['smiles'] = data['smiles']
    else:
        data = pubchem_to_inchikey(chemical_name)
        if data != None:

            tmp['inchikey'] = data[0]
            tmp['smiles'] = data[1]
    if len(tmp) > 0:
        return {'name': chemical_name, **tmp}
    else:
        return tmp


def print_helpful_information():
    '''
    Prints helpful information about using the chemical_parser script.
    '''
    help_text = """
    Usage: chemical_parser.py [CHEMICAL_NAME] [OPTIONS]

    This script takes a chemical name as input, retrieves cheminformatic identifiers
    (InChIKey and SMILES), and writes the information to a CSV file.

    Positional Arguments:
      CHEMICAL_NAME       The name of the chemical to parse.

    Optional Arguments:
      -o, --output FILE   The path to the output CSV file (default: cheminfor).
      -n, --name NAME     The name of the chemical to parse (alternative to positional argument).

    Examples:
      python chemical_parser.py "water" -o water_info.csv
      python chemical_parser.py -n "ethanol"
    """
    print(help_text)

def main():
    parser = argparse.ArgumentParser(description="Parse a chemical name and write cheminformatic identifiers to a CSV file.")
    parser.add_argument('chemical_name', type=str, nargs='?', default=None, help='The name of the chemical to parse.')
    parser.add_argument('-n', '--name', type=str, help='The name of the chemical to parse (alternative to positional argument).')
    parser.add_argument('-o', '--output', type=str, default='cheminfor', help='The path to the output CSV file (default: output.csv).')

    args = parser.parse_args()

    chemical_name = args.name if args.name else args.chemical_name
    output_csv = args.output

    if chemical_name is None:
        print_helpful_information()
        return


    chemical_info = name_to_cheminfo(chemical_name)

    if chemical_info:
        df = pd.DataFrame([chemical_info])
        df.to_csv(f'{output_csv}.csv', index=False)
        print(f"Chemical information written to {output_csv}.csv")
    else:
        print("No chemical information found.")

if __name__ == '__main__':
    main()
