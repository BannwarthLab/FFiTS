import src.forcefield.fortran_energy.fortran_forcefield as fortran_ff
import numpy as np

def get_single_bond_gradient(xyz: np.ndarray, bond_atoms: np.ndarray, bondlength: float, c: float, gradient: np.ndarray):    
    'bond_atoms: zero-based atom pair'
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms[0] + 1
    bond_atoms_onebased[1] = bond_atoms[1] + 1
    fortran_ff.bond_derivatives.get_single_bond_gradient(xyz, bond_atoms_onebased, bondlength, c, gradient)

def get_single_angle_gradient(xyz: np.ndarray, angle_atoms: np.ndarray, angle: float, c: float, gradient: np.ndarray):  
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1
    fortran_ff.angle_derivatives.get_single_angle_gradient(xyz, angle_atoms, angle, c, gradient)

def get_single_dihedral_gradient(xyz: np.ndarray, dihedral_atoms: np.ndarray, dihedral_angle: float, c: float, gradient: np.ndarray):  
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1
    fortran_ff.dihedral_derivatives.get_single_dihedral_gradient(xyz, dihedral_atoms, dihedral_angle, c, gradient)

def get_single_lj_gradient(xyz: np.ndarray, lj_atom_pair: np.ndarray, sigma: float, c: float, gradient: np.ndarray):  
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1
    fortran_ff.dihedral_derivatives.get_single_lj_gradient(xyz, lj_atom_pair, sigma, c, gradient)

def get_single_bond_hessian(xyz: np.ndarray, bond_atoms: np.ndarray, bondlength: float, c: float, hessian: np.ndarray): 
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1   
    fortran_ff.bond_derivatives.get_single_bond_hessian(xyz, bond_atoms, bondlength, c, hessian)

def get_single_angle_hessian(xyz: np.ndarray, angle_atoms: np.ndarray, angle: float, c: float, hessian: np.ndarray):  
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1
    fortran_ff.angle_derivatives.get_single_angle_hessian(xyz, angle_atoms, angle, c, hessian)

def get_single_dihedral_hessian(xyz: np.ndarray, dihedral_atoms: np.ndarray, dihedral_angle: float, c: float, hessian: np.ndarray):  
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1
    fortran_ff.dihedral_derivatives.get_single_dihedral_hessian(xyz, dihedral_atoms, dihedral_angle, c, hessian)

def get_single_lj_hessian(xyz: np.ndarray, lj_atom_pair: np.ndarray, sigma: float, c: float, hessian: np.ndarray):  
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms_zerobased[0] + 1
    bond_atoms_onebased[1] = bond_atoms_zerobased[1] + 1 
    fortran_ff.lj_derivatives.get_single_lj_hessian(xyz, lj_atom_pair, sigma, c, hessian)
