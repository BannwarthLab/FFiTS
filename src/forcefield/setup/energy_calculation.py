import src.forcefield_calculator.fortran_bindings as fb
from src.datatype.structure_data import Structure
import numpy as np


def energy(struc: Structure):
    pass

def complete_gradient(struc: Structure) -> np.ndarray:
    gradient = np.zeros((struc.ff.nat*3), order='F')

    bond_gradient(struc, gradient)
    angle_gradient(struc, gradient)
    dihedral_gradient(struc, gradient)
    repulsive_gradient(struc, gradient)

    return gradient

def complete_hessian(struc: Structure):
    hessian = np.zeros((struc.ff.nat*3,struc.ff.nat*3), order='F')

    bond_hessian(struc, hessian)
    angle_hessian(struc, hessian)
    dihedral_hessian(struc, hessian)
    repulsive_hessian(struc, hessian)

    return hessian

def bond_gradient(struc: Structure, gradient: np.ndarray):
    '''calculates gradient of all bonds of structure'''
    ff = struc.ff
    for i in range(len(ff.bond_list)):
        a1 = ff.bond_list[i][0]
        a2 = ff.bond_list[i][1]
        fb.get_single_bond_gradient(struc.info.fortran_xyz, ff.bond_list[i], ff.bondlengths[i], ff.c_bond[a1][a2]**2, gradient)

def angle_gradient(struc: Structure, gradient: np.ndarray):
    '''calculates gradient of all angles of structure'''
    ff = struc.ff
    for i in range(len(ff.bond_list)):
        a1 = ff.angle_list[i][0]
        a2 = ff.angle_list[i][1]
        a3 = ff.angle_list[i][2]
        fb.get_single_angle_gradient(struc.info.fortran_xyz, ff.angle_list[i], ff.angles[i], ff.c_angle[a1][a2][a3]**2, gradient)

def dihedral_gradient(struc: Structure, gradient: np.ndarray):
    '''calculates gradient of all dihedral angles of structure'''
    ff = struc.ff
    for i in range(len(ff.bond_list)):
        a1 = ff.dihedral_list[i][0]
        a2 = ff.dihedral_list[i][1]
        a3 = ff.dihedral_list[i][2]
        a4 = ff.dihedral_list[i][3]
        fb.get_single_dihedral_gradient(struc.info.fortran_xyz, ff.bond_list[i], ff.bondlengths[i], ff.c_dihedral[a1][a2][a3][a4]**2, gradient)

def repulsive_gradient(struc: Structure, gradient: np.ndarray):
    '''calculates gradient of all repulsive forces of structure'''
    ff = struc.ff #TODO fortran binding dafür schreiben
    # for i in range(len(ff.bond_list)):
    #     a1 = ff.bond_list[i][0]
    #     a2 = ff.bond_list[i][1]
    #     fb.get_single_bond_gradient(struc.info.fortran_xyz, ff.bond_list[i], ff.bondlengths[i], ff.c_bond[a1][a2]**2, gradient)

def bond_hessian(struc: Structure, hessian: np.ndarray):
    ff = struc.ff
    for i in range(len(ff.bond_list)):
        a1 = ff.bond_list[i][0]
        a2 = ff.bond_list[i][1]
        fb.get_single_bond_hessian(struc.info.fortran_xyz, ff.bond_list[i], ff.bondlengths[i], ff.c_bond[a1][a2]**2, hessian)

def angle_hessian(struc: Structure, hessian: np.ndarray):
    ff = struc.ff
    for i in range(len(ff.bond_list)):
        a1 = ff.angle_list[i][0]
        a2 = ff.angle_list[i][1]
        a3 = ff.angle_list[i][2]
        fb.get_single_angle_hessian(struc.info.fortran_xyz, ff.angle_list[i], ff.angles[i], ff.c_angle[a1][a2][a3]**2, hessian)

def dihedral_hessian(struc: Structure, hessian: np.ndarray):
    ff = struc.ff
    for i in range(len(ff.bond_list)):
        a1 = ff.dihedral_list[i][0]
        a2 = ff.dihedral_list[i][1]
        a3 = ff.dihedral_list[i][2]
        a4 = ff.dihedral_list[i][3]
        fb.get_single_dihedral_hessian(struc.info.fortran_xyz, ff.bond_list[i], ff.bondlengths[i], ff.c_dihedral[a1][a2][a3][a4]**2, hessian)

def repulsive_hessian(struc: Structure, hessian: np.ndarray):
    ff = struc.ff #TODO fortran binding dafür schreiben
    # for i in range(len(ff.bond_list)):
    #     a1 = ff.bond_list[i][0]
    #     a2 = ff.bond_list[i][1]
    #     fb.get_single_bond_gradient(struc.info.fortran_xyz, ff.bond_list[i], ff.bondlengths[i], ff.c_bond[a1][a2]**2, gradient)

