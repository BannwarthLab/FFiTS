import ffits.forcefield.fortran.fortran_forcefield as fortran_ff
import numpy as np


def get_single_bond_gradient(
    xyz: np.ndarray,
    bond_atoms: np.ndarray,
    bondlength: float,
    c: float,
    gradient: np.ndarray,
):
    "bond_atoms: zero-based atom pair"
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
    lj_atom_pair_onebased = np.zeros(2)
    lj_atom_pair_onebased[0] = repulsive_atom_pair[0] + 1
    lj_atom_pair_onebased[1] = repulsive_atom_pair[1] + 1
    fortran_ff.lj_derivatives.get_single_lj_hessian(
        xyz, lj_atom_pair_onebased, sigma, c, hessian
    )
