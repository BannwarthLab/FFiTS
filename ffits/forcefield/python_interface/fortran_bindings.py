'Fortran Bindings for FF Gradient and Hessian Calculation. This module provides Python functions that serve as interfaces to the Fortran implementations of gradient and hessian calculations for bonds, angles, dihedrals and repulsive interactions in the force field. These functions take care of converting atom indices from zero-based to one-based format as required by the Fortran code and call the appropriate Fortran subroutines to perform the calculations, adding the results to the provided gradient and hessian arrays in place.'

import ffits.forcefield.fortran.fortran_forcefield as fortran_ff
import numpy as np


def get_single_bond_gradient(
    xyz: np.ndarray,
    bond_atoms: np.ndarray,
    bondlength: float,
    c: float,
    gradient: np.ndarray,
):
    """Calculates the gradient for a single bond on a given displaced geometry and adds it to the provided gradient array in place. The bond is defined by the indices of the two atoms involved and the current bond length. The gradient is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        bond_atoms (np.ndarray): atoms involved in the bond (shape: [2], zero-based indices)
        bondlength (float): reference bondlength
        c (float): force constant
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms[0] + 1
    bond_atoms_onebased[1] = bond_atoms[1] + 1
    fortran_ff.bond_derivatives.get_single_bond_gradient(
        xyz, bond_atoms_onebased, bondlength, c, gradient
    )


def get_single_angle_gradient(
    xyz: np.ndarray,
    angle_atoms: np.ndarray,
    angle: float,
    c: float,
    gradient: np.ndarray,
):
    """Calculates the gradient for a single angle on a given displaced geometry and adds it to the provided gradient array in place. The angle is defined by the indices of the three atoms involved and the current angle. The gradient is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        angle_atoms (np.ndarray): atoms involved in the angle (shape: [3], zero-based indices)
        angle (float): reference angle
        c (float): force constant
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    angle_atoms_onebased = np.zeros(3)
    angle_atoms_onebased[0] = angle_atoms[0] + 1
    angle_atoms_onebased[1] = angle_atoms[1] + 1
    angle_atoms_onebased[2] = angle_atoms[2] + 1
    fortran_ff.angle_derivatives.get_single_angle_gradient(
        xyz, angle_atoms_onebased, angle, c, gradient
    )


def get_single_dihedral_gradient(
    xyz: np.ndarray,
    dihedral_atoms: np.ndarray,
    dihedral_angle: float,
    c: float,
    gradient: np.ndarray,
):
    """Calculates the gradient for a single dihedral on a given displaced geometry and adds it to the provided gradient array in place. The dihedral is defined by the indices of the four atoms involved and the current dihedral angle. The gradient is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        dihedral_atoms (np.ndarray): atoms involved in the dihedral (shape: [4], zero-based indices)
        dihedral_angle (float): reference dihedral angle
        c (float): force constant
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    dihedral_atoms_onebased = np.zeros(4)
    dihedral_atoms_onebased[0] = dihedral_atoms[0] + 1
    dihedral_atoms_onebased[1] = dihedral_atoms[1] + 1
    dihedral_atoms_onebased[2] = dihedral_atoms[2] + 1
    dihedral_atoms_onebased[3] = dihedral_atoms[3] + 1
    fortran_ff.dihedral_derivatives.get_single_dihedral_gradient(
        xyz, dihedral_atoms_onebased, dihedral_angle, c, gradient
    )


def get_single_repulsive_gradient(
    xyz: np.ndarray,
    repulsive_atom_pair: np.ndarray,
    sigma: float,
    c: float,
    gradient: np.ndarray,
):

    """Calculates the gradient for a single repulsive term on a given displaced geometry and adds it to the provided gradient array in place. The repulsive term is defined by the indices of the two atoms involved and the current distance. The gradient is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        repulsive_atom_pair (np.ndarray): atoms involved in the repulsive term (shape: [2], zero-based indices)
        sigma (float): reference distance
        c (float): force constant
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    lj_atom_pair_onebased = np.zeros(2)
    lj_atom_pair_onebased[0] = repulsive_atom_pair[0] + 1
    lj_atom_pair_onebased[1] = repulsive_atom_pair[1] + 1
    fortran_ff.lj_derivatives.get_single_lj_gradient(
        xyz, lj_atom_pair_onebased, sigma, c, gradient
    )


def get_single_bond_hessian(
    xyz: np.ndarray,
    bond_atoms: np.ndarray,
    bondlength: float,
    c: float,
    hessian: np.ndarray,
):
    """Calculates the hessian for a single bond on a given displaced geometry and adds it to the provided hessian array in place. The bond is defined by the indices of the two atoms involved and the current bond length. The hessian is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        bond_atoms (np.ndarray): atoms involved in the bond (shape: [2], zero-based indices)
        bondlength (float): reference bondlength
        c (float): force constant
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    
    bond_atoms_onebased = np.zeros(2)
    bond_atoms_onebased[0] = bond_atoms[0] + 1
    bond_atoms_onebased[1] = bond_atoms[1] + 1
    fortran_ff.bond_derivatives.get_single_bond_hessian(
        xyz, bond_atoms_onebased, bondlength, c, hessian
    )


def get_single_angle_hessian(
    xyz: np.ndarray,
    angle_atoms: np.ndarray,
    angle: float,
    c: float,
    hessian: np.ndarray,
):
    """Calculates the hessian for a single angle on a given displaced geometry and adds it to the provided hessian array in place. The angle is defined by the indices of the three atoms involved and the current angle. The hessian is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        angle_atoms (np.ndarray): atoms involved in the angle (shape: [3], zero-based indices)
        angle (float): reference angle
        c (float): force constant
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    angle_atoms_onebased = np.zeros(3)
    angle_atoms_onebased[0] = angle_atoms[0] + 1
    angle_atoms_onebased[1] = angle_atoms[1] + 1
    angle_atoms_onebased[2] = angle_atoms[2] + 1
    fortran_ff.angle_derivatives.get_single_angle_hessian(
        xyz, angle_atoms_onebased, angle, c, hessian
    )


def get_single_dihedral_hessian(
    xyz: np.ndarray,
    dihedral_atoms: np.ndarray,
    dihedral_angle: float,
    c: float,
    hessian: np.ndarray,
):
    """Calculates the hessian for a single dihedral on a given displaced geometry and adds it to the provided hessian array in place. The dihedral is defined by the indices of the four atoms involved and the current dihedral angle. The hessian is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        dihedral_atoms (np.ndarray): atoms involved in the dihedral (shape: [4], zero-based indices)
        dihedral_angle (float): reference dihedral angle
        c (float): force constant
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    dihedral_atoms_onebased = np.zeros(4)
    dihedral_atoms_onebased[0] = dihedral_atoms[0] + 1
    dihedral_atoms_onebased[1] = dihedral_atoms[1] + 1
    dihedral_atoms_onebased[2] = dihedral_atoms[2] + 1
    dihedral_atoms_onebased[3] = dihedral_atoms[3] + 1
    fortran_ff.dihedral_derivatives.get_single_dihedral_hessian(
        xyz, dihedral_atoms_onebased, dihedral_angle, c, hessian
    )


def get_single_repulsive_hessian(
    xyz: np.ndarray,
    repulsive_atom_pair: np.ndarray,
    sigma: float,
    c: float,
    hessian: np.ndarray,
):
    """Calculates the hessian for a single repulsive interaction on a given displaced geometry and adds it to the provided hessian array in place. The repulsive interaction is defined by the indices of the two atoms involved and the current distance. The hessian is calculated using the Fortran FF implementation.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        repulsive_atom_pair (np.ndarray): atoms involved in the repulsive interaction (shape: [2], zero-based indices)
        sigma (float): sigma parameter
        c (float): force constant
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    lj_atom_pair_onebased = np.zeros(2)
    lj_atom_pair_onebased[0] = repulsive_atom_pair[0] + 1
    lj_atom_pair_onebased[1] = repulsive_atom_pair[1] + 1
    fortran_ff.lj_derivatives.get_single_lj_hessian(
        xyz, lj_atom_pair_onebased, sigma, c, hessian
    )
