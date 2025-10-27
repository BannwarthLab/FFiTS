from src.datatype.structure_data import ForceField, StructuralInformation, get_vander_matrix, angstrom2bohr
import numpy as np
import os
import tempfile
from src.ts_guess.define_starting_parameters import fill_ff
from src.forcefield.fortran_energy.geometry_calc import angle, bondlength, dihedral_angle
from src.forcefield.fortran_energy.ff_energy import energy_ff, complete_gradient, complete_hessian
import pandas as pd
# to not rely on other functions, data is given statically
XYZ = np.array([
    [-2.33287094, 3.31176687, 0.20110100],
    [-0.91630217, 2.85867268, -0.04327585],
    [0.06256276, 3.55862185, 0.08938727],
    [-2.92591176, 3.18375556, -0.71288068],
    [-2.79725946, 2.67965821, 0.96793335],
    [-0.81484498, 1.79716556, -0.36699673],
    [-2.35087245, 4.35655928, 0.51563164]
])
WBO = {
    (1, 2): 1.02668632226515, 
    (2, 3): 1.92755303185758, 
    (1, 4): 0.955689824153634,  
    (1, 5): 0.955863695522291,  
    (2, 6): 0.933812077856736,  
    (1, 7): 0.982636418257069  
}
ATOM_TYPES = ['C', 'C', 'O', 'H', 'H', 'H', 'H']

NAT = 7

def test_blanc_ff_object():
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1_new'
    )
    ff = ForceField(7, path, readff=True, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian) 

def test_construct_ff_object_from_file():
    path2ff = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/ff1_new')
    ff = ForceField(7, path2ff)
    # some random checks on the force field 
    np.testing.assert_equal(ff.bonds['atoms'].iloc[0], np.array([0, 1]))
    assert round(ff.dihedrals['parameter'].iloc[2], 6) == round(0.2630292, 6)
    assert round(ff.angles['reference_value'].iloc[1], 6) == round(2.003157810, 6)
    assert len(ff.bonds) == 6

def test_get_structural_information():
    expected_bo_matrix = np.array([
        [0,1.02668632226515,0,0.955689824153634,0.955863695522291,0,0.982636418257069],
        [1.02668632226515,0,1.92755303185758,0,0,0.933812077856736,0],
        [0,1.92755303185758,0,0,0,0,0],
        [0.955689824153634,0,0,0,0,0,0],
        [0.955863695522291,0,0,0,0,0,0],
        [0,0.933812077856736,0,0,0,0,0],        
        [0.982636418257069,0,0,0,0,0,0]      
    ])
    path2wbo = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/wbo1')
    path2xyz = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/struc1.xyz')

    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    assert info.molecule_count == 1
    np.testing.assert_allclose(info.bo_matrix, expected_bo_matrix, rtol=1e-7)

def test_get_vander_matrix():
    expected_vander_matrix = angstrom2bohr(np.array([
       [2.64, 2.64, 2.54, 2.23, 2.23, 2.23, 2.23],
       [2.64, 2.64, 2.54, 2.23, 2.23, 2.23, 2.23],
       [2.54, 2.54, 2.44, 2.13, 2.13, 2.13, 2.13],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82]]))
    vander_matrix = get_vander_matrix(NAT, ATOM_TYPES)
    print(vander_matrix)
    np.testing.assert_allclose(vander_matrix, expected_vander_matrix, rtol=1e-7)
    assert False


def test_fill_ff():
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    path = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule', 'ff1_new')
    ff_readin = ForceField(NAT, path, readff=True)
    ff = ForceField(NAT, path, readff=False)
    fill_ff(ff, info)
    
    pd.testing.assert_series_equal(ff_readin.bonds['reference_value'], ff.bonds['reference_value'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.bonds['atoms'], ff.bonds['atoms'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.bonds['type'], ff.bonds['type'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.angles['reference_value'], ff.angles['reference_value'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.angles['atoms'], ff.angles['atoms'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.angles['type'], ff.angles['type'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.dihedrals['reference_value'], ff.dihedrals['reference_value'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.dihedrals['atoms'], ff.dihedrals['atoms'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.dihedrals['type'], ff.dihedrals['type'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.repulsive['reference_value'], ff.repulsive['reference_value'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.repulsive['atoms'], ff.repulsive['atoms'], rtol=1e-5, atol=1e-8, check_index=False)
    pd.testing.assert_series_equal(ff_readin.repulsive['type'], ff.repulsive['type'], rtol=1e-5, atol=1e-8, check_index=False)
