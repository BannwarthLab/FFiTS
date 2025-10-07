from src.forcefield.fortran_energy.fortran_bindings import get_single_bond_gradient
from tests.component_tests.static_data import XYZ, WBO, ATOM_TYPES
from src.datatype.structure_data import StructuralInformation
from src.forcefield.fortran_energy.geometry_calc import bondlength
import scipy as sc

import numpy as np


def test_run():
    grad =  np.zeros(7*3)
    struc = StructuralInformation(7, XYZ, WBO, ATOM_TYPES)
    get_single_bond_gradient(struc.fortran_xyz, np.array([0,1]), 2.84821353, 2.0, grad)
    print(bondlength(struc.fortran_xyz, 0, 1))
    print(grad)
    assert False

def test_numerical_gradient():
    pass    
