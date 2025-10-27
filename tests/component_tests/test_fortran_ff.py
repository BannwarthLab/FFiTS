from src.forcefield.fortran_energy.fortran_bindings import get_single_bond_gradient
from src.forcefield.fortran_energy.ff_energy import *
# from tests.component_tests.static_data import XYZ, WBO, ATOM_TYPES
from src.datatype.structure_data import StructuralInformation, ForceField, angstrom2bohr, convert_xyz_to_fortranstyle
from src.forcefield.fortran_energy.geometry_calc import bondlength
from src.io.reader import readin_xyz
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

def numerical_hessian(energy_func, coords, h=1e-5):

    print(coords)
    x = np.asarray(coords, dtype=float).flatten()
    print(x)
    print(x.reshape(7,3).T)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    E0 = energy_func(x.copy())

    # diagonal second derivatives
    for i in range(n):
        xf, xb = x.copy(), x.copy()
        xf[i] += h; xb[i] -= h
        H[i, i] = (energy_func(xf) + energy_func(xb) - 2.0 * E0) / (h * h)

    # mixed second derivatives
    for i in range(n):
        for j in range(i+1, n):
            x_pp = x.copy(); x_pm = x.copy(); x_mp = x.copy(); x_mm = x.copy()
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            E_pp = energy_func(x_pp)
            E_pm = energy_func(x_pm)
            E_mp = energy_func(x_mp)
            E_mm = energy_func(x_mm)
            Hij = (E_pp + E_mm - E_pm - E_mp) / (4.0 * h * h)
            H[i, j] = Hij
            H[j, i] = Hij

    return H



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
        'struc1.xyz'
    )
    ff = ForceField(NAT, path, readff=True, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
    _, _, xyz_start, _ = readin_xyz(path2) 
    analytical_hessian = ff.get_hessian(xyz_start)
    # num_hessian = numerical_hessian(angstrom2bohr(convert_xyz_to_fortranstyle(ff.nat, xyz_start)), ff, delta=1e-7)
    num_hessian = numerical_hessian(ff.get_energy, xyz_start, h=1e-4)
    # print(num_hessian)
    # print(analytical_hessian)
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
    
if __name__ == "__main__":
    test_numerical_hessian()