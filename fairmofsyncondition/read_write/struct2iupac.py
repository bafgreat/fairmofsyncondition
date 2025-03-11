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
from mofstructure import mofdeconstructor
from fairmofsyncondition.read_write import coords_library



def pubchem_to_inchikey(filename):
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
    ase_atom = coords_library.load_data_as_ase(filename)
    smile, inchi, inchikey = file2smile(filename)
    print(smile)
    print(inchikey)
    pubchem = pcp.get_compounds(smile, 'smiles')
    properties['inchikey'] = inchikey
    properties['smiles'] = smile
    properties['inchi'] = inchi

    if len(pubchem) > 0:
        all_prop = pubchem[0]
        properties['cid'] = all_prop.cid if all_prop.cid else None
        properties['iupac_name'] = all_prop.iupac_name if all_prop.iupac_name else None
        properties['h_bond_donor_count'] = all_prop.h_bond_donor_count if all_prop.h_bond_donor_count else None
        properties['h_bond_acceptor_count'] = all_prop.h_bond_acceptor_count if all_prop.h_bond_acceptor_count else None
        properties['rotatable_bond_count'] = all_prop.rotatable_bond_count if all_prop.rotatable_bond_count else None
        properties['charge'] = all_prop.charge if all_prop.charge else None
        # properties['synonyms'] = all_prop.synonyms if all_prop.synonyms else None
        # properties['sids'] = all_prop.sids if all_prop.sids else None
        return properties
    else:
        return None


def file2smile(filename):
    """
    Function that reads a filename and uses openbabel to compute smi, inChi and inChiKey
    **Parameters:**
        filename (str): name of file containing the structure. It can be in any ase readable
        file format as well as qchem out, AMS out, Gaussian out.

    **Return:**
        smi (str): SMILE strings
        inchi : inchi hashing of structure
        inChiKey : 28 character hashing


    """
    ase_atom = coords_library.load_data_as_ase(filename)
    smi, inChi, inChiKey = mofdeconstructor.compute_openbabel_cheminformatic(ase_atom)
    return smi, inChi, inChiKey

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
        File Name     Name of file containing structures.
                    ['any ase readable file', 'qchem.out', 'ams.out'] (Only non periodic systems)).

        Optional Arguments:
        -o, --output FILE        The path to the output CSV file (default: cheminfor.csv).


        Examples:
        struct2iupac water.xyz  -o water_info.csv
        struct2iupac  benzene.out
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
        'filename',
        type=str,
        nargs='?',
        default=None,
        help='filename of the molecule to determine name'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='cheminfor.csv',
        help='The path to the output CSV file (default: cheminfor.csv).'
    )

    args = parser.parse_args()

    filename =  args.filename

    output_csv = args.output

    if filename is None:
        print_helpful_information()
        return

    chemical_info = pubchem_to_inchikey(filename)

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