from fairmofsyncondition.read_write import filetyper, iupacname2cheminfo, cheminfo2iupac, struct2iupac


def get_cheminfofromiupac(iupac_name):
    """
    A function to extract the smile strings, inchikey and inchi from
    a correctly written iupac name or common name.

    **parameter:**
        iupac_name (str): correctly chemical name
    **return**
        data (dic): dictionary containing name and cheminformatic information

    **NOTE**
    This code can also be ran from the command line with the below command

    ```bash
    iupac2cheminfor -n iupac_name
    ```
    """

    data = iupacname2cheminfo.name_to_cheminfo(iupac_name)
    return data


def cheminfor2name(indentifier, name_type='smile'):
    """
    A function that determines iupac names from a cheminformatic
    identifier. It should expect a smile string, inchi, inchikey
    or the cid (pubchem indentification number). If the indentifier is
    a smile string them the name_type should be "smile". And if it is
    an inchikey then the name_type should be "inchikey".

    **parameter:**
        indentifier (str): The cheminformatic identifier
        name_type (str): The type of identifier [smile, inchi, inchikey, cid]
    **return**
        data (dic): dictionary containing name and cheminformatic information

    **NOTE**
    This code can also be ran from the command line with the below command

    ```bash
    cheminfo2iupac -n 'O'
    ```
    """

    data = cheminfo2iupac.pubchem_to_inchikey(indentifier, name=name_type)
    return data


def structure2name(filename):
    """
    A function that determine iupac names and cheminformatic
    identifier from a structure. simple parse it any ase readable
    files and it will compute the cheminformatic indentifiers and
    extract the name of the structure.

    **parameter:**
        filename (str): file containing the chemical structure.
    **return**
        data (dic): dictionary containing name and cheminformatic information

    **NOTE**
    This code can also be ran from the command line with the below command

    ```bash
    struct2iupac filename
    ```
    """

    data = cheminfo2iupac.pubchem_to_inchikey(filename)
    return data
