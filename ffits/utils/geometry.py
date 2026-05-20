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


def bondlength(geometry: np.array, atom1: int, atom2: int):
    """
    Returns bond length between atom1 and atom2.

    Parameters
    ----------
    geometry : np.ndarray
        Shape (3, n_atoms). Coordinates of atoms.
    atom1, atom2 : int
        Indices of atoms (0-based).

    Returns
    -------
    float
        Bond length.
    """
    vector_12 = geometry[:, atom1] - geometry[:, atom2]
    return np.linalg.norm(vector_12)


def angle(geometry: np.array, atom1: int, atom2: int, atom3: int):
    """
    Returns angle formed by atoms (atom1 - atom2 - atom3).

    Parameters
    ----------
    geometry : np.ndarray
        Shape (3, n_atoms). Coordinates of atoms.
    atom1, atom2, atom3 : int
        Indices of atoms (0-based).

    Returns
    -------
    float
        Angle in radians.
    """
    xyz1 = geometry[:, atom1]
    xyz2 = geometry[:, atom2]
    xyz3 = geometry[:, atom3]

    v12 = xyz1 - xyz2
    v23 = xyz3 - xyz2

    dot = np.dot(v12, v23)
    norm = np.linalg.norm(v12) * np.linalg.norm(v23)
    cos_theta = np.clip(dot / norm, -1.0, 1.0)

    return np.arccos(cos_theta)


def dihedral_angle(geometry: np.array, atom1: int, atom2: int, atom3: int, atom4: int):
    """
    Returns dihedral angle formed by atoms (atom1 - atom2 - atom3 - atom4).

    Parameters
    ----------
    geometry : np.ndarray
        Shape (3, n_atoms). Coordinates of atoms.
    atom1, atom2, atom3, atom4 : int
        Indices of atoms (0-based).

    Returns
    -------
    float
        Dihedral angle in radians.
    """
    xyz1 = geometry[:, atom1]
    xyz2 = geometry[:, atom2]
    xyz3 = geometry[:, atom3]
    xyz4 = geometry[:, atom4]

    v12 = xyz2 - xyz1
    v23 = xyz3 - xyz2
    v34 = xyz4 - xyz3

    # normal vectors
    normal1 = np.cross(v12, v23)
    normal2 = np.cross(v23, v34)

    length1 = np.linalg.norm(normal1)
    length2 = np.linalg.norm(normal2)
    length3 = np.linalg.norm(v23)

    # atan2 version (preferred for signed dihedral)
    x = np.dot(normal1, normal2)
    y = np.dot(np.cross(normal1, normal2), v23 / length3)
    angle = np.arctan2(y, x)

    return angle
