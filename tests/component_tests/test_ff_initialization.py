from src.datatype.structure_data import ForceField, StructuralInformation, get_vander_matrix
import numpy as np
import os
import tempfile
from src.forcefield.setup.define_starting_parameters import fill_ff
from src.forcefield.fortran_energy.geometry_calc import angle, bondlength, dihedral_angle
from src.forcefield.fortran_energy.ff_energy import * 
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
    ff = ForceField(7, os.path.join(os.getcwd))

# def test_construct_ff_object_from_file():
#     path2ff = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/ff1')
#     ff = ForceField(7, path2ff)
#     # some random checks on the force field 
#     assert ff.bonds == [1, 7]
#     assert round(ff.c_dihedral[(4, 1, 2, 6)],6) == round(0.19815370,6)
#     # assert round(ff.c_dihedral[3],6) == round(0.19815370,6)
#     assert round(ff.angles[4],6) == round(1.92842546,6)
#     assert len(ff.c_angle) == len(ff.angle_list)
#     assert len(ff.angles) == 9

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
    path = os.path.join('/home/dbabushkina/1_ts_search2024/pytsguess/tests/examples/small_single_molecule', 'ff1_new')
    ff = ForceField(NAT, path, readff=True)
    # fill_ff(ff, info)

    # ff.write()
    print(energy_ff(info.fortran_xyz, ff))    
    print(ff.bonds)
    print(ff.angles)
    print(ff.dihedrals)
    print(ff.repulsive)
    gradient = complete_gradient(info.fortran_xyz, ff)
    print(gradient)
    hessian = complete_hessian(info.fortran_xyz, ff)
    print(hessian)
    assert False


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
    print(vander_matrix)
    np.testing.assert_allclose(vander_matrix, expected_vander_matrix, rtol=1e-7)
    assert False

# def test_get_c_tables():
#     info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         path = os.path.join(tmpdirname, 'ff')
#         ff = ForceField(NAT, path, readff=False)
#         define_relevant_bonds(ff, info)
#         get_c_tables(ff, info)
#         assert len(ff.c_bond) == len(ff.bond_list) 
#         assert len(ff.c_angle) == len(ff.angle_list) 
#         assert len(ff.c_dihedral) == len(ff.dihedral_list) 
#         assert len(ff.c_lj) == len(ff.lj_list)
#         # check for False-like values
#         assert all(ff.c_bond.values())
#         assert all(ff.c_angle.values())
#         assert all(ff.c_dihedral.values())
#         assert all(ff.c_lj.values())
#         # check specific values, may need to be changed with different first guesses
#         # print(ff.c_angle)
#         # assert round(ff.c_angle[(4,1,5)], 7) == round(0.19145774, 7)
        
# def test_get_c_tables_DivisionByZero():
#     info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         path = os.path.join(tmpdirname, 'ff')
#         ff = ForceField(NAT, path, readff=False)
#         define_relevant_bonds(ff, info)
#         info.bo_matrix[1,0] = 0
#         with pytest.raises(Exception):
#             get_c_tables(ff, info)

# def test_get_ff_reference_values():
#     expected_bonds = np.array([
#         2.84821353,
#         2.07306194,
#         2.07289233,
#         2.06217920,
#         2.28782128,
#         2.10590956
#     ])
#     expected_angles = np.array([
#         2.17528567,
#         2.00315781,
#         1.91550051,
#         1.91559648,
#         1.92842546,
#         2.10474182,
#         1.86118849,
#         1.92084246,
#         1.92116183
#     ])
#     expected_dihedral_angles = np.array([
#         2.11933159,
#         -2.12383049,
#         -0.00201816,
#         -1.02233516,
#         1.01768806,
#         3.13950040
#     ])
#     expected_sigmas = np.array([
#         4.79990391,
#         4.21408887,
#         4.21408887,
#         4.21408887,
#         4.21408887,
#         4.02511627,
#         4.02511627,
#         4.02511627,
#         4.02511627,
#         3.43930123,
#         3.43930123,
#         3.43930123,
#         3.43930123,
#         3.43930123,
#         3.43930123
#     ])
#     info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         path = os.path.join(tmpdirname, 'ff')
#         ff = ForceField(NAT, path, readff=False)
#         define_relevant_bonds(ff, info)
#         get_c_tables(ff, info)
#         get_ff_reference_values(ff, info)
#         print(info.vander_matrix)
#         np.testing.assert_allclose(ff.bondlengths, expected_bonds, rtol=1e-5)
#         np.testing.assert_allclose(ff.angles, expected_angles, rtol=1e-5)
#         np.testing.assert_allclose(ff.dihedrals, expected_dihedral_angles, rtol=1e-5)
#         np.testing.assert_allclose(ff.sigmas, expected_sigmas, rtol=1e-5)
        

# def test_bondlengths():
#     assert np.isclose(get_bondlength(geometry, 0, 1), expected_bonds["1-2"], rtol=1e-6)
#     assert np.isclose(get_bondlength(geometry, 1, 2), expected_bonds["2-3"], rtol=1e-6)
#     assert np.isclose(get_bondlength(geometry, 2, 3), expected_bonds["3-4"], rtol=1e-6)

# def test_angles():
#     assert np.isclose(get_angle(geometry, 0, 1, 2), expected_angles["1-2-3"], rtol=1e-6)
#     assert np.isclose(get_angle(geometry, 1, 2, 3), expected_angles["2-3-4"], rtol=1e-6)

# def test_dihedral():
#     assert np.isclose(get_dihedral_angle(geometry, 0, 1, 2, 3), expected_dihedral, rtol=1e-6)
