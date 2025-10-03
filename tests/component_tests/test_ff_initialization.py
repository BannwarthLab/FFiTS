from src.datatype.structure_data import ForceField, StructuralInformation
import pytest
import numpy as np
import os
from static_data import WBO, XYZ, ATOM_TYPES, NAT
import tempfile
from src.forcefield.setup.define_starting_parameters import define_relevant_bonds, get_vander_matrix, get_c_tables
# to not rely on other functions, data is given statically


# def test_blanc_ff_object():
    # ff = ForceField(7, os.path.join(os.getcwd))

def test_construct_ff_object_from_file():
    path2ff = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/ff1')
    ff = ForceField(7, path2ff)
    # some random checks on the force field 
    assert ff.bond_list[3] == [1, 7]
    assert round(ff.c_dihedral[(4, 1, 2, 6)],6) == round(0.19815370,6)
    # assert round(ff.c_dihedral[3],6) == round(0.19815370,6)
    assert round(ff.angles[4],6) == round(1.92842546,6)
    assert len(ff.c_angle) == len(ff.angle_list)
    assert len(ff.angles) == 9

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


def test_define_bonds():
    expected_bond_list = np.array([
        [1, 2],
        [1, 4],
        [1, 5],
        [1, 7],
        [2, 3],
        [2, 6]
    ])    
    expected_angle_list = np.array([
        [1, 2, 3],
        [1, 2, 6],
        [2, 1, 4],
        [2, 1, 5],
        [2, 1, 7],
        [3, 2, 6],
        [4, 1, 5],
        [4, 1, 7],
        [5, 1, 7]
    ])
    expected_dihedral_list = np.array([
        [3, 2, 1, 4],
        [3, 2, 1, 5],
        [3, 2, 1, 7],
        [4, 1, 2, 6],
        [5, 1, 2, 6],
        [6, 2, 1, 7]
    ])
    expected_lj_list = np.array([
        [1, 3],
        [1, 6],
        [2, 4],
        [2, 5],
        [2, 7],
        [3, 4],
        [3, 5],
        [3, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [4, 7],
        [5, 6],
        [5, 7],
        [6, 7]
    ])
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, 'ff')
        ff = ForceField(NAT, path, readff=False)
        define_relevant_bonds(ff, info)
        assert np.array_equal(ff.bond_list, expected_bond_list)
        assert np.array_equal(ff.angle_list, expected_angle_list)
        assert np.array_equal(ff.dihedral_list, expected_dihedral_list)
        assert np.array_equal(ff.lj_list, expected_lj_list)


def test_get_vander_matrix():
    expected_vander_matrix = np.array([
       [2.64, 2.64, 2.54, 2.23, 2.23, 2.23, 2.23],
       [2.64, 2.64, 2.54, 2.23, 2.23, 2.23, 2.23],
       [2.54, 2.54, 2.44, 2.13, 2.13, 2.13, 2.13],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
       [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82]])
    vander_matrix = get_vander_matrix(NAT, ATOM_TYPES)
    np.testing.assert_allclose(vander_matrix, expected_vander_matrix, rtol=1e-3)

def test_get_c_tables():
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, 'ff')
        ff = ForceField(NAT, path, readff=False)
        define_relevant_bonds(ff, info)
        get_c_tables(ff, info)
        assert len(ff.c_bond) == len(ff.bond_list) 
        assert len(ff.c_angle) == len(ff.angle_list) 
        assert len(ff.c_dihedral) == len(ff.dihedral_list) 
        assert len(ff.c_lj) == len(ff.lj_list)
        # check for False-like values
        assert all(ff.c_bond.values())
        assert all(ff.c_angle.values())
        assert all(ff.c_dihedral.values())
        assert all(ff.c_lj.values())
        # check specific values, may need to be changed with different first guesses
        assert round(ff.c_angle[(4,1,5)], 7) == round(0.19145774, 7)
        
def test_get_c_tables_DivisionByZero():
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, 'ff')
        ff = ForceField(NAT, path, readff=False)
        define_relevant_bonds(ff, info)
        info.bo_matrix[1,0] = 0
        with pytest.raises(Exception):
            get_c_tables(ff, info)