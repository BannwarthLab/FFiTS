from ffits.forcefield.python_interface.fortran_bindings import get_single_bond_gradient
from ffits.forcefield.python_interface.ff_energy import *
# from tests.component_tests.static_data import XYZ, WBO, ATOM_TYPES
from ffits.datatype.structure_data import StructuralInformation, ForceField, angstrom2bohr, convert_xyz_to_fortranstyle
from ffits.utils.geometry_calc import bondlength
from ffits.io.reader import readin_xyz
import scipy as sc
import numpy as np
from ffits.ts_guess.parameterize_ff import calculate_hessian_rmsd
from ffits.forcefield.python_interface.ff_energy import * 
import os

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

def numerical_gradient(geometry_displ: np.ndarray, ff: ForceField, delta: float = 1e-5) -> np.ndarray:
    geometry_displ = np.asarray(geometry_displ, dtype=float)
    
    nat = geometry_displ.shape[1]
    gradient = np.zeros(nat * 3, order="F")
    for atom in range(nat):
        for coord in range(3):  
            displaced_plus  = geometry_displ.copy()
            displaced_minus = geometry_displ.copy()

            displaced_plus[coord, atom]  += delta
            displaced_minus[coord, atom] -= delta

            e_plus  = energy_ff(displaced_plus, ff)
            e_minus = energy_ff(displaced_minus, ff)

            grad_val = (e_plus - e_minus) / (2 * delta)
            gradient[3 * atom + coord] = grad_val

    return gradient

def numerical_hessian(nat, energy_func, coords, delta: float = 1e-4):
    n_atom = nat
    hessian = np.zeros((3 * n_atom, 3 * n_atom), dtype=float)

    xyz11   = coords.copy()
    xyz1m1  = coords.copy()
    xyzm11  = coords.copy()
    xyzm1m1 = coords.copy()

    for j in range(3 * n_atom):        
        for i in range(3 * n_atom):    
            coordinate1 = j % 3
            atom1 = j // 3
            coordinate2 = i % 3
            atom2 = i // 3

            xyz11[coordinate1, atom1]   += delta
            xyz11[coordinate2, atom2]   += delta
            xyz1m1[coordinate1, atom1]  += delta
            xyz1m1[coordinate2, atom2]  -= delta
            xyzm11[coordinate1, atom1]  -= delta
            xyzm11[coordinate2, atom2]  += delta
            xyzm1m1[coordinate1, atom1] -= delta
            xyzm1m1[coordinate2, atom2] -= delta

            energy11   = energy_func(xyz11)
            energy1m1  = energy_func(xyz1m1)
            energym11  = energy_func(xyzm11)
            energym1m1 = energy_func(xyzm1m1)

            hessian[j, i] = (energy11 - energy1m1 - energym11 + energym1m1) / (4.0 * delta**2)

            xyz11[:]   = coords
            xyz1m1[:]  = coords
            xyzm11[:]  = coords
            xyzm1m1[:] = coords

    return hessian



def hessian_rmsd(H1: np.ndarray, H2: np.ndarray) -> float:
    H1 = np.asarray(H1, dtype=float)
    H2 = np.asarray(H2, dtype=float)

    if H1.shape != H2.shape:
        raise ValueError(f"Hessian shapes differ: {H1.shape} vs {H2.shape}")

    diff = H1 - H2
    rmsd = np.sqrt(np.mean(diff**2))
    return rmsd

def test_numerical_gradient():
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1.csv'
    )    
    path2 = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'struc2.xyz'
    )
    ff = ForceField(NAT, path, readff=True, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
    _, _, xyz_start, _ = readin_xyz(path2) 
    analytical_gradient = ff.get_gradient(xyz_start)
    num_gradient = numerical_gradient(angstrom2bohr(convert_xyz_to_fortranstyle(ff.nat, xyz_start)), ff)
    np.testing.assert_allclose(analytical_gradient, num_gradient, rtol=1e-6)

def test_numerical_hessian():
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1.csv'
    )
    path2 = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'struc2.xyz'
    )
    ff = ForceField(NAT, path, readff=True, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
    _, _, xyz_start, _ = readin_xyz(path2) 
    xyz = angstrom2bohr(convert_xyz_to_fortranstyle(7, xyz_start))
    analytical_hessian = ff.get_hessian(xyz)
    num_hessian = numerical_hessian(7, ff.get_energy, xyz, delta=1e-4)
    for i in range(21):
        for j in range(21):
            print(f'{i}, {j}, {analytical_hessian[i,j]}, {num_hessian[i,j]}')
    assert calculate_hessian_rmsd(analytical_hessian, num_hessian, 3*ff.nat) < 0.006
    np.testing.assert_allclose(analytical_hessian, num_hessian,rtol=2, atol=1e-7, verbose=True) # TODO noch nicht so richtig tief ggf noch mal einzelte Teile checken
    

def test_energy_ff():
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1.csv'
    )
    ff = ForceField(NAT, path, readff=True)
    energy = energy_ff(info.fortran_xyz*1.01, ff)
    assert energy < 0.01
    