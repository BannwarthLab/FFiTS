"""Geometry helpers: unit conversion and bond length/angle/dihedral calculations."""

import numpy as np


def convert_xyz_to_fortranstyle(xyz: np.ndarray) -> np.ndarray:
    """
    Converts xyz to column-major.
    """
    return np.asarray(xyz, dtype=float, order="F").T


def angstrom2bohr(val: float | np.ndarray):
    """
    Convert a value or array from Angstroms to Bohr.
    """
    if type(val) == np.array:
        return np.multiply(val, 1.8897259)
    return val * 1.8897259


def _column_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise dot product of two arrays of shape (3, n)."""
    return (a * b).sum(axis=0)


def bondlengths(geometry: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Returns bond lengths for all atom pairs in atoms.

    Args:
        geometry (np.ndarray): Coordinates, shape (3, n_atoms).
        atoms (np.ndarray): Atom pairs (0-based), shape (n_terms, 2).

    Returns:
        np.ndarray: Bond lengths, shape (n_terms,).
    """
    atoms = np.asarray(atoms, dtype=int).reshape(-1, 2)
    v12 = geometry[:, atoms[:, 0]] - geometry[:, atoms[:, 1]]
    return np.linalg.norm(v12, axis=0)


def angles(geometry: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Returns angles for all atom triples (atom1 - atom2 - atom3) in atoms.

    Args:
        geometry (np.ndarray): Coordinates, shape (3, n_atoms).
        atoms (np.ndarray): Atom triples (0-based, vertex in the middle), shape (n_terms, 3).

    Returns:
        np.ndarray: Angles in radians, shape (n_terms,).
    """
    atoms = np.asarray(atoms, dtype=int).reshape(-1, 3)
    v12 = geometry[:, atoms[:, 0]] - geometry[:, atoms[:, 1]]
    v23 = geometry[:, atoms[:, 2]] - geometry[:, atoms[:, 1]]

    dot = _column_dot(v12, v23)
    norm = np.linalg.norm(v12, axis=0) * np.linalg.norm(v23, axis=0)
    cos_theta = np.clip(dot / norm, -1.0, 1.0)

    return np.arccos(cos_theta)


def dihedral_angles(geometry: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Returns signed dihedral angles for all atom quadruples (atom1 - atom2 - atom3 - atom4) in atoms.

    Args:
        geometry (np.ndarray): Coordinates, shape (3, n_atoms).
        atoms (np.ndarray): Atom quadruples (0-based), shape (n_terms, 4).

    Returns:
        np.ndarray: Dihedral angles in radians (-pi, pi], shape (n_terms,).
    """
    atoms = np.asarray(atoms, dtype=int).reshape(-1, 4)
    v12 = geometry[:, atoms[:, 1]] - geometry[:, atoms[:, 0]]
    v23 = geometry[:, atoms[:, 2]] - geometry[:, atoms[:, 1]]
    v34 = geometry[:, atoms[:, 3]] - geometry[:, atoms[:, 2]]

    # normal vectors
    normal1 = np.cross(v12, v23, axis=0)
    normal2 = np.cross(v23, v34, axis=0)
    length3 = np.linalg.norm(v23, axis=0)

    # atan2 version (preferred for signed dihedral)
    x = _column_dot(normal1, normal2)
    y = _column_dot(np.cross(normal1, normal2, axis=0), v23 / length3)

    return np.arctan2(y, x)


def bondlength(geometry: np.array, atom1: int, atom2: int):
    """Returns bond length between atom1 and atom2 (single-term version of :func:`bondlengths`).

    Args:
        geometry (np.ndarray): Coordinates, shape (3, n_atoms).
        atom1 (int): Index of the first atom (0-based).
        atom2 (int): Index of the second atom (0-based).

    Returns:
        float: Bond length.
    """
    return bondlengths(geometry, [atom1, atom2])[0]


def angle(geometry: np.array, atom1: int, atom2: int, atom3: int):
    """Returns angle formed by atoms (atom1 - atom2 - atom3) (single-term version of :func:`angles`).

    Args:
        geometry (np.ndarray): Coordinates, shape (3, n_atoms).
        atom1 (int): Index of the first atom (0-based).
        atom2 (int): Index of the vertex atom (0-based).
        atom3 (int): Index of the third atom (0-based).

    Returns:
        float: Angle in radians.
    """
    return angles(geometry, [atom1, atom2, atom3])[0]


def dihedral_angle(geometry: np.array, atom1: int, atom2: int, atom3: int, atom4: int):
    """Returns dihedral angle formed by atoms (atom1 - atom2 - atom3 - atom4) (single-term version of :func:`dihedral_angles`).

    Args:
        geometry (np.ndarray): Coordinates, shape (3, n_atoms).
        atom1 (int): Index of the first atom (0-based).
        atom2 (int): Index of the second atom (0-based).
        atom3 (int): Index of the third atom (0-based).
        atom4 (int): Index of the fourth atom (0-based).

    Returns:
        float: Dihedral angle in radians.
    """
    return dihedral_angles(geometry, [atom1, atom2, atom3, atom4])[0]
