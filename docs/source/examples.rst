===========================================
Example Usage
===========================================

User-Friendly examples to ease understanding and usage of `fairmofsyncondition`

materials to pytorch geometric
--------------------------------------------------------

**ase to pytorch geometric**
create a pytorch geometric data from a structure file. It can either be
periodic system or molecular systems and should be ASE-readable.

.. code-block:: python

    from fairmofsyncondition.read_write import coords_library
    py_data = coords_library.ase_to_pytorch_geometric(input_system)

**NOTE**

**input_system** can either be an ase_atom object or a filename (like cif or xyz)

**pytorch geometric to ase**

Another useful tool is to convert from the pytorch geometric data back to ase_atom

.. code-block:: python

    from fairmofsyncondition.read_write import coords_library
    ase_atom = coords_library.pytorch_geometric_to_ase(py_data)


create and save dataset to lmdb
---------------------------------

Given a folder containing a series of cif files for which one wishes to convert
to pytorch geometric dataset. The code below can create this dataset and directly
save in and lmdb file format in a memory effecient manner.

**standard create and save**

The below command is an effecient way to quickly create a new pytorch gemeotric
dataset and directly save to disc.

.. code-block:: python

    import pickle
    import lmdb
    import torch
    from fairmofsyncondition.read_write import coords_library

    path_to_cif = 'folder_containing'
    path_to_lmdb = 'data.lmdb'
    count = 0
    with lmdb.open(path_to_lmdb , map_size=int(1e12)) as lmdb_env:
        with lmdb_env.begin(write=True) as txn:
            for i, filenames in enumerate(path_to_cif):
                py_data = coords_library.ase_to_pytorch_geometric(filenames)
                txn.put(f"{i}".encode(), pickle.dumps(py_data ))
                count += 1
            txn.put(b"__len__", pickle.dumps(count))


The above code creates `data.lmdb` file containing the pytorch geometric data.

**Create, add properties and save**

It is possible to add any property to the structure, like energy, forces, hessians
even categorical data. The below code is a snippet of how this can be archeived

.. code-block:: python

    import pickle
    import lmdb
    import torch
    from fairmofsyncondition.read_write import coords_library
    from fairmofsyncondition.featurizer import encoder

    path_to_cif = 'folder_containing'
    path_to_lmdb = 'data.lmdb'
    list_of_energies = [...]
    list_of_hessians = [...]
    list_of_forces = [...]
    categories = [...]
    list_of_list_of_catagories = [[...], [...] ...[...]]
    count = 0
    with lmdb.open(path_to_lmdb , map_size=int(1e12)) as lmdb_env:
        with lmdb_env.begin(write=True) as txn:
            for i, filenames in enumerate(path_to_cif):
                py_data = coords_library.ase_to_pytorch_geometric(filenames)
                py_data.energy = torch.tensor(list_of_energies[i], dtype=torch.float16)
                py_data.hessians = torch.tensor(list_of_hessians[i], dtype=torch.float16)
                py_data.forces  = torch.tensor(list_of_forces[i], dtype=torch.float16)
                py_data.category_name = encoder.onehot_encoder_pyg(list_of_list_of_catagories[i], categories)
                txn.put(f"{i}".encode(), pickle.dumps(py_data ))
                count += 1
            txn.put(b"__len__", pickle.dumps(count))


The above code will create pytorch geometric dataset and save to `data.lmdb`.

reading lmdb pytorch dataset
-----------------------------
The code below provides a memory efficient way to load the dataset with consuming
so much memory as well as an efficient way to split data

.. code-block:: python

    from fairmofsyncondition.read_write import coords_library
    path_to_mdb = 'data.lmdb'
    data = coords_library.LMDBDataset(lmdb_path=path_to_mdb)
   # check all methods available
   print(dir(data))

   # print for energy
   print(data[0])

   # split data
   train_data, test_data = data.split_data(train_size=0.8, random_seed=42, shuffle=True)


cheminformatics
---------------------------
You can use `fairmofsyncondition` to quickly convert from `iupac names` to `iupac identifiers`
and vice versa. One can also convert `chemical structures` to `iupac names` and `iupac identifiers`
by following the these examples.

iupacname2cheminfo
-------------------------------
This function extracts SMILES strings, InChIKey, and InChI from a correctly written IUPAC name or common name.

.. code-block:: python

    from fairmofsyncondition.read_write import iupacname2cheminfo
    data = iupacname2cheminfo.name_to_cheminfo("ethanol")
    print(data)

cheminfo2iupac
------------------------
This function determines the IUPAC name from a cheminformatic identifier (SMILES, InChI, InChIKey, or CID).
If the indentifier is a SMILES then the name_type should be "smile", if it is an InChIKey then the name_type
should be "inchikey".

.. code-block:: python

    from fairmofsyncondition.read_write import cheminfo2iupac

    name_info = cheminfo2iupac.pubchem_to_inchikey('O', name='smile')
    print("IUPAC name from SMILES 'O':", name_info)

    name_info2 = cheminfo2iupac.pubchem_to_inchikey('ZNALFCQVQALKNH-UHFFFAOYSA-N', name='inchikey')
    print("IUPAC name from INCHIKEY 'ZNALFCQVQALKNH-UHFFFAOYSA-N':", name_info2)


struct2iupac
------------------------
This function extracts the IUPAC name and cheminformatic identifiers from a structure file.
It parses any ASE-readable file and computes the corresponding cheminformatic information and
iupac name.

.. code-block:: python

    from fairmofsyncondition.read_write import struct2iupac
    struct_info = cheminfo2iupac.pubchem_to_inchikey(filename)
    print("Cheminformatic info from structure file:", struct_info)


Command Line Usage
------------------
The cheminformatic data can also be executed directly from the command line. For example:

- To convert an IUPAC name to cheminformatic information:

  .. code-block:: bash

      iupac2cheminfor -n "ethanol"

- To determine the IUPAC name from a cheminformatic identifier:

  .. code-block:: bash

      cheminfo2iupac -n "O"

- To extract information from a structure file:

  .. code-block:: bash

      struct2iupac example_structure.xyz


