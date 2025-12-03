# fairmofsyncondition

**fairmofsyncondition** is a Python module designed to predict the synthesis conditions for metal-organic frameworks (MOFs). This tool offers two main functionalities:

1. **Predict Synthesis Conditions from Crystal Structures Structure:** Given a crystal structure of a MOF, the model will predict the optimal set of conditions required to synthesize the specified structure.
2. **Predict MOF Structures from Synthesis Conditions:** If a set of reaction condition is provided, the model will predict the crystal structures of all possible MOF that can be formed under those conditions.

The model is trained on data extracted from the FAIR-MOF dataset, which is a comphrensive and carefully curated collection of MOF structures paired with their corresponding experimental synthesis conditions, building units and experimental synthetic conditions. This dataset serves as a robust foundation for accurate and reliable predictions.

## Features

- **Bidirectional Prediction:** Whether you have a MOF structure or reaction conditions, the module can provide the corresponding synthesis conditions or possible MOF structures, respectively.
- **FAIR-MOF Dataset:** Utilizes a comprehensive curated dataset of MOFs with verified experimental conditions.
- **User-Friendly:** Easy to install and use, with minimal setup required.

## Installation

The module can be installed directly from GitHub. Follow the steps below to get started:

### PyPi

Simply pip install.

'''bash
pip install fairmofsyncondition
'''

### GitHub Installation

To install fairmofsyncondition from GitHub, execute the following commands in your terminal:

```bash
# Clone the repository
git clone https://github.com/bafgreat/fairmofsyncondition.git

# Navigate into the project directory
cd fairmofsyncondition

# Install the package
pip install --upgrade pip setuptools wheel
pip install .
```

### PYPI Installation

To install fairmofsyncondition from PYPI, simply execute the following commands in your terminal:

```Python

pip install fairmofsyncondition

```

## Useful tool

`fairmofsyncondition_syncon`
is a command-line tool for predicting synthetic conditions of Metal–Organic Frameworks (MOFs) directly from CIF files.
It extracts **organic ligands**, **space group information**, and computes the **top-5 predicted metal salts**.

Quickly run command on any cif file

```bash
  fairmofsyncondition_syncon my_mof.cif
```

Or run and provide and outfile

```bash
  fairmofsyncondition_syncon my_mof.cif -o my_mof_report.txt
```

`iupac2cheminfor`
one of the most useful tool is to directly extract cheminonformatic identifiers such
as inchikey and smile strings directly from iupac names or common names. This can be
achieved using `iupac2cheminfor` CLI as follows:

```bash
  iupac2cheminfor 'water'
```

or

```bash
  iupac2cheminfor -n 'water' -o filename
```

The out will be written by default to cheminfor.csv if no output is provided
and if porvided it will be written to the name parsed.

`cheminfo2iupac`

Another useful tool is directly convert a `smile` or and `inchikey` their iupac name.
To achieve this simply run the following commandline tool

```bash
  cheminfo2iupac -n 'O' -o filename
```

`struct2iupac`
In other cases one may one to directly extract the iupac name and cheminformatic identifier of a chemical structure.
The quickest way to do this is by running the following commands.

```bash
  struct2iupac XOWJUR.xyz
```

`pg_graph from cif file or folder`

One can reliably create an `lmdb` dataset using the following code.

```bash
  pggraph_from_cifs -i ./CIFs/ -o mof_db.lmdb
```

## Training

To quickly train the model on the command line, simply use the
`train_bde` CLI command. It has several helpful options to facilated
training.

```bash
train_bde -h
```

The above command will provide all neccesarry information to train a model.

We also provide a commandline to to run optuna for searching optimal
command line arguments.

```bash
find_bde_parameters -h
```

## Machine Learning Folder

The folder **`machine_learning/`** contains the code and Jupyter notebooks to predict the **metal salt of a given MOF** using **Graph Neural Networks (GNNs)**.

- Each notebook (`Ex1.ipynb`, `Ex2.ipynb`, …, `Ex10.ipynb`) explores different combinations of input features such as:
  - **Scherrer** (grain size)
  - **Microstrain** (lattice distortion)
  - **OMS** (Open Metal Sites)
  - **Atomic Number**

- The notebooks share the same structure:
  1. **Load the Data** – import and prepare the dataset
  2. **Define the GNN Model** – specify the architecture
  3. **Train the Model** – train and save weights in `tmp/` (optional)
  4. **Load and Evaluate the Model** – load trained weights and test performance

**Note**:
If you only want to **test the model** with pre-trained weights, you can simply **skip step 3** (training).

## Documentation

Full documentation can be found [docs](https://bafgreat.github.io/fairmofsyncondition/).

## LICENSE

This project is licensed under the MIT
