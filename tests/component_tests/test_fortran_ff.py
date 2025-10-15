from src.forcefield.fortran_energy.fortran_bindings import get_single_bond_gradient
from src.forcefield.fortran_energy.ff_energy import *
from tests.component_tests.static_data import XYZ, WBO, ATOM_TYPES
from src.datatype.structure_data import StructuralInformation, ForceField, angstrom2bohr, convert_xyz_to_fortranstyle
from src.forcefield.fortran_energy.geometry_calc import bondlength
from src.interface.reader import readin_xyz
import scipy as sc
import numpy as np
from src.forcefield.fortran_energy.ff_energy import * 
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


def numerical_hessian(geometry_displ: np.ndarray, ff: ForceField, delta: float = 1e-4) -> np.ndarray:
    geometry_displ = np.asarray(geometry_displ, dtype=float)

    nat = geometry_displ.shape[1]
    ncoords = nat * 3
    hessian = np.zeros((ncoords, ncoords), dtype=float)

    grad0 = numerical_gradient(geometry_displ, ff, delta=delta/10)

    for a in range(nat):
        for coord in range(3):
            idx = 3 * a + coord  
            geom_plus  = geometry_displ.copy()
            geom_minus = geometry_displ.copy()

            geom_plus[coord, a]  += delta
            geom_minus[coord, a] -= delta

            grad_plus  = numerical_gradient(geom_plus, ff, delta=delta/10)
            grad_minus = numerical_gradient(geom_minus, ff, delta=delta/10)

            hessian[:, idx] = (grad_plus - grad_minus) / (2 * delta)

    # Enforce symmetry numerically
    hessian = 0.5 * (hessian + hessian.T)
    return hessian

import numpy as np

def numerical_hessian(energy_func, coords, h=1e-4):
    """
    Compute a numerical Hessian (3n x 3n) using only energy evaluations.
    
    Parameters
    ----------
    energy_func : callable
        Function that takes a 1D numpy array of shape (3n,) and returns a scalar energy.
    coords : np.ndarray
        1D array of shape (3n,), the coordinates at which to compute the Hessian.
    h : float, optional
        Finite difference step size (default: 1e-4).
    
    Returns
    -------
    hessian : np.ndarray
        Symmetric Hessian matrix of shape (3n, 3n).
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.size
    hessian = np.zeros((n, n))
    
    E0 = energy_func(coords)

    # Diagonal second derivatives
    for i in range(n):
        d = np.zeros(n)
        d[i] = h
        E_plus = energy_func(coords + d)
        E_minus = energy_func(coords - d)
        hessian[i, i] = (E_plus - 2*E0 + E_minus) / (h**2)
    
    # Off-diagonal cross derivatives
    for i in range(n):
        for j in range(i+1, n):
            d_i = np.zeros(n)
            d_j = np.zeros(n)
            d_i[i] = h
            d_j[j] = h
            
            E_pp = energy_func(coords + d_i + d_j)
            E_pm = energy_func(coords + d_i - d_j)
            E_mp = energy_func(coords - d_i + d_j)
            E_mm = energy_func(coords - d_i - d_j)
            
            val = (E_pp - E_pm - E_mp + E_mm) / (4 * h**2)
            hessian[i, j] = hessian[j, i] = val  # enforce symmetry

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
        'ff1_new'
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
        'ff1_new'
    )
    path2 = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'struc2.xyz'
    )
    ff = ForceField(NAT, path, readff=True, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
    _, _, xyz_start, _ = readin_xyz(path2) 
    analytical_hessian = ff.get_hessian(xyz_start)
    # num_hessian = numerical_hessian(angstrom2bohr(convert_xyz_to_fortranstyle(ff.nat, xyz_start)), ff, delta=1e-7)
    num_hessian = numerical_hessian(ff.get_energy, xyz_start)

    np.testing.assert_allclose(analytical_hessian, num_hessian, rtol=1e-3)

def test_energy_ff():
    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1_new'
    )
    ff = ForceField(NAT, path, readff=True)
    energy = energy_ff(info.fortran_xyz*1.01, ff)
    assert energy < 0.01
    