import src.fortran_forcefield.fortran_forcefield as fortran_ff
import numpy as np

def get_single_bond_gradient(xyz: np.ndarray, bond_atoms: np.ndarray, bondlength: float, c: float, gradient: np.ndarray):    
    fortran_ff.bond_derivatives.get_single_bond_gradient(xyz, bond_atoms, bondlength, c, gradient)

def get_single_angle_gradient(xyz: np.ndarray, angle_atoms: np.ndarray, angle: float, c: float, gradient: np.ndarray):  
    fortran_ff.angle_derivatives.get_single_angle_gradient(xyz, angle_atoms, angle, c, gradient)

def get_single_dihedral_gradient(xyz: np.ndarray, dihedral_atoms: np.ndarray, dihedral_angle: float, c: float, gradient: np.ndarray):  
    fortran_ff.dihedral_derivatives.get_single_dihedral_gradient(xyz, dihedral_atoms, dihedral_angle, c, gradient)


def get_single_bond_hessian(xyz: np.ndarray, bond_atoms: np.ndarray, bondlength: float, c: float, hessian: np.ndarray):    
    fortran_ff.bond_derivatives.get_single_bond_gradient(xyz, bond_atoms, bondlength, c, hessian)

def get_single_angle_hessian(xyz: np.ndarray, angle_atoms: np.ndarray, angle: float, c: float, hessian: np.ndarray):  
    fortran_ff.angle_derivatives.get_single_angle_gradient(xyz, angle_atoms, angle, c, hessian)

def get_single_dihedral_hessian(xyz: np.ndarray, dihedral_atoms: np.ndarray, dihedral_angle: float, c: float, hessian: np.ndarray):  
    fortran_ff.dihedral_derivatives.get_single_dihedral_gradient(xyz, dihedral_atoms, dihedral_angle, c, hessian)
