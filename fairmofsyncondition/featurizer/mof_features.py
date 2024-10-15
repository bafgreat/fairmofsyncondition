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

import mofstructure.mofdeconstructor as mof_decon


class FeatureFromAseAtom():
    """
    This class is used as a constructor to create MOF features from
    ase atom objects.
    **parameters:**
        ase_atom : ASE Atom object
    """
    def __init__(self, ase_atom):
        self.ase_atom = ase_atom

    def extract_sbu_building_units(self, ase_atom):
        """
        Extract building units from ase atom object
        **parameters:**
            ase_atom : ASE Atom object
        **returns:**
            metal_sbus : list of metal building
            organic_sbus : List of organic building units
            all_regions : List of all regions in the MOF
        """
        connected_components, atoms_indices_at_breaking_point, porpyrin_checker, all_regions =\
            mof_decon.secondary_building_units(ase_atom)

        metal_sbus, organic_sbus, _  =\
            mof_decon.find_unique_building_units(
                connected_components,
                atoms_indices_at_breaking_point,
                ase_atom, porpyrin_checker,
                all_regions,
                cheminfo=True
                )
        return metal_sbus, organic_sbus, all_regions

