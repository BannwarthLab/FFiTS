from __future__ import annotations
import os
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


def readin_xyz(xyz_path: str) -> Tuple[int, str, np.ndarray, List[str]]:
    """
    Reads an XYZ file and returns the number of atoms, comment line, coordinates, and atom types. It is assumed that the XYZ file follows the standard format where the first line contains the number of atoms, the second line is a comment, and the subsequent lines contain the atom type followed by x, y, z coordinates, given in Angström.

    Args:
        xyz_path (str): path to xyz file

    Raises:
        FileNotFoundError: xyz_path does not exist
        ValueError: xyz_path is invalid
        ValueError: xyz_path contains invalid data
        ValueError: xyz_path has mismatched atom count
        ValueError: xyz_path has malformed lines
        ValueError: xyz_path has invalid numeric values

    Returns:
        Tuple[int, str, np.ndarray, List[str]]: nat, comment, coordinates , atom types
    """
    if not os.path.isfile(xyz_path):
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    with open(xyz_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file]

    while lines and not lines[-1]:
        lines.pop()

    if len(lines) < 2:
        raise ValueError(
            f"Invalid XYZ file: {xyz_path}. Must contain at least 2 lines."
        )

    try:
        nat = int(lines[0])
    except ValueError:
        raise ValueError(
            f"First line of XYZ file must be an integer (number of atoms)."
        )
    comment = lines[1]
    atom_lines = lines[2:]
    if len(atom_lines) != nat:
        raise ValueError(
            f"Atom count mismatch in {xyz_path}: declared {nat}, found {len(atom_lines)}"
        )

    atom_types: List[str] = []
    coordinates: list[list[float]] = []

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed line in {xyz_path}: '{line}'")
        atom, *coords = parts[:4]
        atom_types.append(atom)
        try:
            coordinates.append([float(c) for c in coords])
        except ValueError:
            raise ValueError(f"Invalid numeric values in line: '{line}'")

    return nat, comment, np.array(coordinates, dtype=float), atom_types


def read_xyz_2dict(
    filename: str,
) -> dict:
    """Returns the structure data from an XYZ file as a dict. Wrapper for readin_xyz."""
    nat, comment, xyz, atom_types = readin_xyz(filename)
    return {"nat": nat, "comment": comment, "atom_types": atom_types, "xyz": xyz}


def read_wbo_file(wbo_path: str) -> Dict[Tuple[int, int], float]:
    """
    Reads xtb-style WBO (Wiberg Bond Order) data from a file and parses it into a dictionary.

    Returns a dictionary where keys are tuples of atom indices (sorted) and values are WBOs.

    Atoms are directly converted to 0-based indexing when stored in the dictionary.

    Example line in file:
        1 2 0.95
    """
    if not os.path.isfile(wbo_path):
        raise FileNotFoundError(f"WBO file not found: {wbo_path}")

    wbo_dict: Dict[Tuple[int, int], float] = {}

    with open(wbo_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue  # skip empty lines
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed line in WBO file: '{line.strip()}'")
            try:
                atom1, atom2 = int(parts[0]), int(parts[1])
                wbo_value = float(parts[2])
            except ValueError:
                raise ValueError(f"Invalid numeric values in line: '{line.strip()}'")
            bond = tuple(sorted((atom1 - 1, atom2 - 1)))
            wbo_dict[bond] = wbo_value

    return wbo_dict


def read_xtb_hessian(file_path):
    """
    Reads a Hessian matrix from the xtb output format
    """

    with open(file_path, "r") as f:
        lines = f.readlines()

    data_lines = [
        line.strip() for line in lines if not line.lower().startswith("$hessian")
    ]

    numbers = []
    for line in data_lines:
        if line:  # skip empty lines
            numbers.extend(map(float, line.split()))

    total_values = len(numbers)
    dim = int(np.sqrt(total_values))

    if dim * dim != total_values:
        raise ValueError(
            f"The number of Hessian elements ({total_values}) does not form a square matrix."
        )

    if dim % 3 != 0 or dim == 0:
        raise ValueError(f"The Hessian dimensions ({dim}x{dim}) is not divisible by 3.")

    hessian = np.array(numbers).reshape((dim, dim))
    return hessian
