
import os
import numpy as np
import pytest 
from src.interface.reader import readin_xyz, read_wbo_file
from tests.component_tests.utility import compare_dictionaries


def test_readin_xyz():
    expected_coordinates = np.array([
        [-2.33287094, 3.31176687, 0.20110100],
        [-0.91630217, 2.85867268, -0.04327585],
        [0.06256276, 3.55862185, 0.08938727],
        [-2.92591176, 3.18375556, -0.71288068],
        [-2.79725946, 2.67965821, 0.96793335],
        [-0.81484498, 1.79716556, -0.36699673],
        [-2.35087245, 4.35655928, 0.51563164]
    ])
    expected_atom_types = ['C', 'C', 'O', 'H', 'H', 'H', 'H']
    path2xyz = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/struc1.xyz')
    nat, comment, coordinates, atom_types = readin_xyz(path2xyz)
    assert nat == 7
    assert comment == 'This is a test case'
    np.testing.assert_allclose(coordinates, expected_coordinates, rtol=1e-7)
    assert atom_types == expected_atom_types

def test_readin_xyz_error_filenotfound():
    path2xyz = 'wrongpath'
    with pytest.raises(Exception):
        readin_xyz(path2xyz)

def test_readin_xyz_error_wrongnumberofatoms():
    path2xyz = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/struc1_wrong.xyz')
    with pytest.raises(Exception):
        nat, comment, coordinates, atom_types = readin_xyz(path2xyz) 

def test_read_wbo_file():
    expected_wbo_dict = {
           (1, 2): 1.02668632226515, 
           (2, 3): 1.92755303185758, 
           (1, 4): 0.955689824153634,  
           (1, 5): 0.955863695522291,  
           (2, 6): 0.933812077856736,  
           (1, 7): 0.982636418257069  
    }
    path2wbo = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/wbo1')
    wbo_dict = read_wbo_file(path2wbo)
    assert compare_dictionaries(wbo_dict, expected_wbo_dict)

def test_read_wbo_file_error_filenotfound():
    path2wbo = 'wrongpath'
    with pytest.raises(Exception):
        read_wbo_file(path2wbo)

