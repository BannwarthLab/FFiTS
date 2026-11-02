"""FF energy, gradient and hessian evaluation (per-term and combined) via the Fortran bindings.

The term data is read from the FF DataFrames into numpy arrays once per call
(see :func:`_term_arrays`), so no pandas row access happens inside the loops.
Repulsive terms (by far the most numerous, ~N_atoms^2) are evaluated in a single
call of the batched Fortran routines; bond, angle and dihedral terms use the
Fortran routines term by term.
"""

import ffits.forcefield.python_interface.fortran_bindings as fb
from ffits.datatype.forcefield_data import ForceField
from ffits.utils.geometry import angles, bondlengths, dihedral_angles
import numpy as np
import pandas as pd


def _create_term_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms FF DataFrame columns to np arrays for Atoms (0-based, shape [n_terms, n_atoms_per_term]), reference values and force constants (parameter**2)."""
    if df.empty:
        return np.zeros((0, 4), dtype=np.int32), np.zeros(0), np.zeros(0)
    atoms = np.array(df["atoms"].tolist(), dtype=np.int32).reshape(len(df), -1)
    reference = df["reference_value"].to_numpy(dtype=float)
    force_constant = df["parameter"].to_numpy(dtype=float) ** 2
    return atoms, reference, force_constant


def _check_geometry(xyz_displaced: np.ndarray, ff: ForceField) -> None:
    """Raise if the geometry is not column-major or its atom count doesn't match the force field."""
    if np.shape(xyz_displaced) == (ff.nat, 3):
        raise Exception("Please give xyz in column-major format.")

    if np.shape(xyz_displaced)[1] != ff.nat:
        raise Exception(
            f"Number of atoms in the geometry ({np.shape(xyz_displaced)[1]}) does not match the number of atoms in the force field ({ff.nat})."
        )


def energy_ff(xyz_displaced: np.ndarray, ff: ForceField) -> float:
    """Returns FF energy for given geometry.

    Args:
        xyz_displaced (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation

    Raises:
        Exception: If the geometry is not column-major or its atom count doesn't match the force field.

    Returns:
        float: The computed energy.
    """
    _check_geometry(xyz_displaced, ff)

    fact_bond = 1.0
    fact_ang = 1.0
    fact_dih = 1.0
    energy = 0.0

    # bonds
    atoms, reference, c = _create_term_arrays(ff.bonds)
    current = bondlengths(xyz_displaced, atoms)
    energy += np.sum(fact_bond * c * (current - reference) ** 2)

    # angles
    atoms, reference, c = _create_term_arrays(ff.angles)
    current = angles(xyz_displaced, atoms)
    energy += np.sum(fact_ang * c * (current - reference) ** 2)

    # dihedrals
    atoms, reference, c = _create_term_arrays(ff.dihedrals)
    current = dihedral_angles(xyz_displaced, atoms)
    energy += np.sum(
        fact_dih
        * c
        * (
            (np.cos(current) - np.cos(reference)) ** 2
            + (np.sin(current) - np.sin(reference)) ** 2
        )
    )

    # repulsive
    atoms, reference, c = _create_term_arrays(ff.repulsive)
    current = bondlengths(xyz_displaced, atoms)
    energy += np.sum(fact_bond * 4 * c * (reference / current) ** 12)

    return energy


def _fortran_terms(
    xyz: np.ndarray, df: pd.DataFrame, fortran_routine, output: np.ndarray
) -> None:
    """Calls a single-term routine of fortran_bindings (zero-based atoms) for every term in df, which adds its derivative to output in place."""
    atoms, reference, c = _create_term_arrays(df)
    for n in range(len(reference)):
        fortran_routine(xyz, atoms[n], reference[n], c[n], output)


# -------------------------------------------------------------------------
# --- Bond Gradient -------------------------------------------------------
# -------------------------------------------------------------------------
def bond_gradient(xyz: np.ndarray, ff: ForceField, gradient: np.ndarray) -> None:
    """Calculates FF bond gradient of given displaced geometry and adds it in place to the given gradient array.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    _fortran_terms(xyz, ff.bonds, fb.get_single_bond_gradient, gradient)


# -------------------------------------------------------------------------
# --- Angle Gradient ------------------------------------------------------
# -------------------------------------------------------------------------
def angle_gradient(xyz: np.ndarray, ff: ForceField, gradient: np.ndarray) -> None:
    """Calculates FF angle gradient of given displaced geometry and adds it in place to the given gradient array.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    _fortran_terms(
        xyz,
        ff.angles,
        fb.get_single_angle_gradient,
        gradient,
    )


# -------------------------------------------------------------------------
# --- Dihedral Gradient ---------------------------------------------------
# -------------------------------------------------------------------------
def dihedral_gradient(xyz: np.ndarray, ff: ForceField, gradient: np.ndarray) -> None:
    """Calculates FF dihedral gradient of given displaced geometry and adds it in place to the given gradient array.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    _fortran_terms(
        xyz,
        ff.dihedrals,
        fb.get_single_dihedral_gradient,
        gradient,
    )


# -------------------------------------------------------------------------
# --- Repulsive Gradient --------------------------------------------------
# -------------------------------------------------------------------------
def repulsive_gradient(xyz: np.array, ff: ForceField, gradient: np.ndarray) -> None:
    """Calculates FF repulsive gradient of given displaced geometry and adds it in place to the given gradient array.

    All pairs are evaluated in one call of the batched Fortran routine
    get_lj_gradient (lj_derivatives.f90), see :func:`fb.get_repulsive_gradient`.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        gradient (np.ndarray): FF gradient array to which the calculated gradient will be added in place (shape: [nat*3])
    """
    if ff.repulsive.empty:
        return
    atoms, sigma, c = _create_term_arrays(ff.repulsive)
    fb.get_repulsive_gradient(xyz, atoms, sigma, c, gradient)


def complete_gradient(xyz_displaced: np.ndarray, ff: ForceField) -> np.ndarray:
    """Calculates complete FF gradient of the given displaced geometry.

    Args:
        xyz_displaced (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation

    Raises:
        Exception: Raises an exception if the input geometry is not in the correct format or if the number of atoms does not match the force field.

    Returns:
        np.ndarray: calculated gradient array (shape: [nat*3])
    """
    _check_geometry(xyz_displaced, ff)

    gradient = np.zeros((ff.nat * 3), order="F")

    bond_gradient(xyz_displaced, ff, gradient)
    angle_gradient(xyz_displaced, ff, gradient)
    dihedral_gradient(xyz_displaced, ff, gradient)
    repulsive_gradient(xyz_displaced, ff, gradient)

    return gradient


# -------------------------------------------------------------------------
# --- Bond hessian -------------------------------------------------------
# -------------------------------------------------------------------------
def bond_hessian(xyz: np.ndarray, ff: ForceField, hessian: np.ndarray) -> None:
    """Calculates FF bond hessian of given displaced geometry and adds it in place to the given hessian array.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    _fortran_terms(xyz, ff.bonds, fb.get_single_bond_hessian, hessian)


# -------------------------------------------------------------------------
# --- Angle hessian ------------------------------------------------------
# -------------------------------------------------------------------------
def angle_hessian(xyz: np.ndarray, ff: ForceField, hessian: np.ndarray) -> None:
    """Calculates FF angle hessian of given displaced geometry and adds it in place to the given hessian array.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    _fortran_terms(xyz, ff.angles, fb.get_single_angle_hessian, hessian)


# -------------------------------------------------------------------------
# --- Dihedral hessian ---------------------------------------------------
# -------------------------------------------------------------------------
def dihedral_hessian(xyz: np.ndarray, ff: ForceField, hessian: np.ndarray) -> None:
    """Calculates FF dihedral hessian of given displaced geometry and adds it in place to the given hessian array.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    _fortran_terms(
        xyz,
        ff.dihedrals,
        fb.get_single_dihedral_hessian,
        hessian,
    )


# -------------------------------------------------------------------------
# --- Repulsive hessian --------------------------------------------------
# -------------------------------------------------------------------------
def repulsive_hessian(xyz: np.array, ff: ForceField, hessian: np.ndarray) -> None:
    """Calculates FF repulsive hessian of given displaced geometry and adds it in place to the given hessian array.

    All pairs are evaluated in one call of the batched Fortran routine
    get_lj_hessian (lj_derivatives.f90), see :func:`fb.get_repulsive_hessian`.

    Args:
        xyz (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation
        hessian (np.ndarray): FF hessian array to which the calculated hessian will be added in place (shape: [nat*3, nat*3])
    """
    if ff.repulsive.empty:
        return
    atoms, sigma, c = _create_term_arrays(ff.repulsive)
    fb.get_repulsive_hessian(xyz, atoms, sigma, c, hessian)


def complete_hessian(xyz_displaced: np.ndarray, ff: ForceField) -> np.ndarray:
    """Calculates complete FF hessian of the given displaced geometry.

    Args:
        xyz_displaced (np.ndarray): displaced geometry in column-major format (shape: [nat, 3])
        ff (ForceField): Force field object containing parameters and lists for energy calculation

    Raises:
        Exception: Raises an exception if the input geometry is not in the correct format or if the number of atoms does not match the force field.

    Returns:
        np.ndarray: calculated hessian array (shape: [nat*3, nat*3])
    """
    _check_geometry(xyz_displaced, ff)

    hessian = np.zeros((ff.nat * 3, ff.nat * 3), order="F")

    bond_hessian(xyz_displaced, ff, hessian)
    angle_hessian(xyz_displaced, ff, hessian)
    dihedral_hessian(xyz_displaced, ff, hessian)
    repulsive_hessian(xyz_displaced, ff, hessian)

    return hessian
