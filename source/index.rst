fairmofsyncondition
===============================================

**fairmofsyncondition** is a Python module designed to predict the synthesis conditions for metal-organic frameworks (MOFs). This tool offers two main functionalities:

1. **Predict Synthesis Conditions from Crystal Structures:** Given a crystal structure of a MOF, the model will predict the optimal set of conditions required to synthesize the specified structure.
2. **Predict MOF Structures from Synthesis Conditions:** If a set of reaction conditions is provided, the model will predict the crystal structures of all possible MOFs that can be formed under those conditions.

The model is trained on data extracted from the FAIR-MOF dataset, which is a comprehensive and carefully curated collection of MOF structures paired with their corresponding experimental synthesis conditions, building units, and experimental synthetic conditions. This dataset serves as a robust foundation for accurate and reliable predictions.

Features
========

- **Bidirectional Prediction:** Whether you have a MOF structure or reaction conditions, the module can provide the corresponding synthesis conditions or possible MOF structures, respectively.
- **FAIR-MOF Dataset:** Utilizes a comprehensive curated dataset of MOFs with verified experimental conditions.
- **User-Friendly:** Easy to install and use, with minimal setup required.

Installation
============

The module can be installed directly from GitHub or PYPI. Follow the steps below to get started:

GitHub Installation
==================

To install `fairmofsyncondition` from GitHub, execute the following commands in your terminal:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/bafgreat/fairmofsyncondition.git

    # Navigate into the project directory
    cd fairmofsyncondition

    # Install the package
    pip install .

PYPI Installation
==================

To install `fairmofsyncondition` from PYPI, simply execute the following commands in your terminal:

.. code-block:: bash

    pip install fairmofsyncondition

License
=======

This project is licensed under the MIT License.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   module

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`