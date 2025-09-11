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
import os
import pandas as pd
import argparse
import requests
import pubchempy as pcp


def pubchem_to_inchikey(identifier, name='smiles'):
    '''
    A function that retrieves chemical properties from PubChem using the given identifier.

    This function queries the PubChem database via the PubChemPy package and extracts
    a set of cheminformatic properties for the first matching compound. The properties
    include InChIKey, CID, IUPAC name, canonical SMILES, hydrogen bond donor and acceptor counts,
    rotatable bond count, and charge.

    Parameters:
        identifier (str): The chemical identifier to query (e.g., chemical name or SMILES).
        search_type (str): The type of identifier provided. Options include 'name' or 'smiles'.
                           Default is 'name'.

    Returns:
        dict or None: A dictionary containing the chemical properties if a matching compound is found;
                      otherwise, None.
    '''
    properties = {}

    pubchem = pcp.get_compounds(identifier, name)
    # print('inchikey', pubchem[0].inchikey)
    # print('name', pubchem[0].synonyms)
    # print('iupac', pubchem[0].iupac_name)
    # print('isomeric_smile', pubchem[0].isomeric_smiles)
    if len(pubchem) > 0:
        all_prop = pubchem[0]
        properties['inchikey'] = all_prop.inchikey if all_prop.inchikey else None
        properties['cid'] = all_prop.cid if all_prop.cid else None
        properties['iupac_name'] = all_prop.iupac_name if all_prop.iupac_name else None
        # properties['smiles'] = all_prop.canonical_smiles if all_prop.canonical_smiles else None
        properties['smiles'] = all_prop.connectivity_smiles if all_prop.connectivity_smiles else None
        properties['h_bond_donor_count'] = all_prop.h_bond_donor_count if all_prop.h_bond_donor_count else None
        properties['h_bond_acceptor_count'] = all_prop.h_bond_acceptor_count if all_prop.h_bond_acceptor_count else None
        properties['rotatable_bond_count'] = all_prop.rotatable_bond_count if all_prop.rotatable_bond_count else None
        properties['charge'] = all_prop.charge if all_prop.charge else None
        # properties['synonyms'] = all_prop.synonyms if all_prop.synonyms else None
        # properties['sids'] = all_prop.sids if all_prop.sids else None
        return properties
    else:
        return None

def print_helpful_information():
    """
    Prints helpful information about using the chemical_parser script.
    """
    help_text = """
        Usage: chemical_parser.py [CHEMICAL_IDENTIFIER] [OPTIONS]

        This script takes a chemical identifier as input (either a chemical name or a SMILES string),
        retrieves cheminformatic properties from PubChem (e.g., InChIKey, CID, IUPAC name, SMILES, etc.),
        and writes the information to a CSV file.

        Positional Arguments:
        CHEMICAL_IDENTIFIER      The chemical name or SMILES string to parse.

        Optional Arguments:
        -o, --output FILE        The path to the output CSV file (default: cheminfor.csv).
        -t, --type TYPE          The type of chemical identifier provided:\n
                                ['name', 'smiles, cid'] (default: 'smiles').

        Examples:
        cheminfor2iupac "water" -t name -o water_info.csv
        cheminfor2iupac  "CCO" -t smiles -o ethanol_info.csv
    """
    print(help_text)

def main():
    """
    Main function to parse command-line arguments, retrieve chemical information from PubChem,
    and write (or append) the data to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Parse a chemical identifier and write cheminformatic properties to a CSV file."
    )
    parser.add_argument(
        'chemical_identifier',
        type=str,
        nargs='?',
        default=None,
        help='The chemical identifier (name or SMILES) to parse.'
    )
    parser.add_argument(
        '-t', '--type',
        type=str,
        default='smiles',
        choices=['name', 'smiles', 'inchikey'],
        help="The type of chemical identifier provided (default: 'smiles')."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='cheminfor.csv',
        help='The path to the output CSV file (default: cheminfor.csv).'
    )

    args = parser.parse_args()

    chemical_identifier = args.chemical_identifier
    search_type = args.type
    output_csv = args.output

    if chemical_identifier is None:
        print_helpful_information()
        return

    chemical_info = pubchem_to_inchikey(chemical_identifier, search_type)

    if chemical_info:
        df = pd.DataFrame([chemical_info])
        # If the output file exists, append without the header; otherwise, create a new file with the header.
        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', index=False, header=False)
            print(f"Chemical information appended to {output_csv}")
        else:
            df.to_csv(output_csv, index=False)
            print(f"Chemical information written to {output_csv}")
    else:
        print("No chemical information found.")

if __name__ == '__main__':
    main()