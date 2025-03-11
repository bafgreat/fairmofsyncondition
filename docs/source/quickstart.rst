Quick Start:
=================================

Useful Tools
------------

The module provides several command-line tools to work with chemical nomenclature and structures:

- **iupac2cheminfor**: Extracts cheminformatic identifiers (such as InChIKey and SMILES strings) directly from IUPAC or common names.

  .. code-block:: bash

      iupac2cheminfor 'water'

  Alternatively, specify an output file:

  .. code-block:: bash

      iupac2cheminfor -n 'water' -o filename

  By default, the output is written to ``cheminfor.csv`` if no output is provided.

- **cheminfo2iupac**: Converts a SMILES or InChIKey to its corresponding IUPAC name.

  .. code-block:: bash

      cheminfo2iupac -n 'O' -o filename

- **struct2iupac**: Extracts the IUPAC name and cheminformatic identifiers from a chemical structure file.

  .. code-block:: bash

      struct2iupac XOWJUR.xyz

Training
--------

To quickly train the model via the command line, use the ``train_bde`` CLI command. This command provides several helpful options to facilitate training:

.. code-block:: bash

    train_bde -h

For an optimal search of command-line parameters using Optuna, run:

.. code-block:: bash

    find_bde_parameters -h