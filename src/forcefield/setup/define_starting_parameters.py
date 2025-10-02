import numpy as np
from src.datatype.structure_data import StructuralInformation, ForceField

VANDER_VALUES = np.array([
0.91, 0.92, # H, He
0.75, 1.28, 1.35, 1.32, 1.27, 1.22, 1.17, 1.13, # Li-Ne
1.04, 1.24, 1.49, 1.56, 1.55, 1.53, 1.49, 1.45, # Na-Ar
1.35, 1.34, # K, Ca
1.42, 1.42, 1.42, 1.42, 1.42, # Sc-Zn
1.42, 1.42, 1.42, 1.42, 1.42,
1.50, 1.57, 1.60, 1.61, 1.59, 1.57, # Ga-Kr
1.48, 1.46, # Rb, Sr
1.49, 1.49, 1.49, 1.49, 1.49, # Y-Cd
1.49, 1.49, 1.49, 1.49, 1.49,
1.52, 1.64, 1.71, 1.72, 1.72, 1.71, # In-Xe
2.00, 2.00,
2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, # La-Yb
2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
2.00, 2.00, 2.00, 2.00, 2.00, # Lu-Hg
2.00, 2.00, 2.00, 2.00, 2.00,
2.00, 2.00, 2.00, 2.00, 2.00, 2.00 # Tl-Rn
])

PERIODIC_TABLE = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86
}

def atom_symbol_to_number(symbol: str) -> int:
    """Convert an element symbol (e.g. 'C') to its atomic number (e.g. 6)."""
    try:
        return PERIODIC_TABLE[symbol.capitalize()]
    except KeyError:
        raise ValueError(f"Unknown atom symbol: {symbol}")
    
def get_vander_matrix(nat, at, vander_values=VANDER_VALUES, factor=1.0):
    """
    Build van der Waals interaction matrix.

    Parameters
    ----------
    nat : int
        Number of atoms.
    at : array-like of str
    vander_values : np.ndarray
        Reference van der Waals radii (length 86).
    factor : float
        Scaling factor.

    Returns
    -------
    vander_matrix : np.ndarray (nat x nat)
    """
    # Convert atomic numbers to 0-based indices
    atom_numbers = [PERIODIC_TABLE[s] for s in at]
    radii = vander_values[np.array(atom_numbers) - 1] * factor
    # Broadcasting sum of pairwise radii
    return radii[:, None] + radii[None, :]

def get_bondlength(xyz, atom1, atom2):
    """Compute Euclidean bond length between two atoms."""
    return np.linalg.norm(xyz[atom1] - xyz[atom2])

def get_c_tables(ff: ForceField, repulsive_start_ex=None):
    """
    Compute c_bond, c_angle, c_dihedral, and c_lj tables for ForceField object.
    """
    for atom1, atom2 in ff.bond_list[:, :ff.count_bond].T:
        bondlength = get_bondlength(ff.xyz0, atom1, atom2)
        ff.c_bond[(atom1, atom2)] = ff.wbo[atom1, atom2] / bondlength

    # --- Angles ---
    for atom1, atom2, atom3 in ff.angle_list[:, :ff.count_angle].T:
        bl1 = get_bondlength(ff.xyz0, atom1, atom2)
        bl2 = get_bondlength(ff.xyz0, atom2, atom3)
        product = ff.wbo[atom1, atom2] * ff.wbo[atom2, atom3]
        ff.c_angle[(atom1, atom2, atom3)] = (product / (bl1 * bl2)) ** 0.5

    # --- Dihedrals ---
    for atom1, atom2, atom3, atom4 in ff.dihedral_list[:, :ff.count_dihedral].T:
        bl1 = get_bondlength(ff.xyz0, atom1, atom2)
        bl2 = get_bondlength(ff.xyz0, atom2, atom3)
        bl3 = get_bondlength(ff.xyz0, atom3, atom4)
        product = (ff.wbo[atom1-1, atom2-1] *
                   ff.wbo[atom2-1, atom3-1] *
                   ff.wbo[atom3-1, atom4-1])
        ff.c_dihedral[(atom1, atom2, atom3, atom4)] = (product / (bl1 * bl2 * bl3)) ** (1/3)

    # --- Lennard-Jones terms ---
    repulsive_start = 0.01 if repulsive_start_ex is None else repulsive_start_ex
    for atom1, atom2 in ff.lj_list[:, :len(ff.bond_list)].T:
        ff.c_lj[(atom1, atom2)] = repulsive_start


def canonical_dihedral(i, j, l, m):
    """
    Return a canonical ordering for a dihedral (i, j, l, m),
    so that (i,j,l,m) and (m,l,j,i) collapse to the same tuple.
    """
    forward = (i, j, l, m)
    reverse = (m, l, j, i)
    return min(forward, reverse)


def define_relevant_bonds(ff: ForceField, info: StructuralInformation, bo_threshold: float = 0.0):
    """
    build bond, angle, dihedral, and LJ lists from a bond-order matrix
    """
    n = ff.nat
    wbo = info.bo_matrix

    n = ff.nat
    wbo = info.bo_matrix

    # --- Step 1: adjacency matrix ---
    A = (wbo > bo_threshold).astype(int)
    np.fill_diagonal(A, 0)

    # --- Step 2: Bonds (edges) ---
    bond_i, bond_j = np.where(np.triu(A, 1))
    bonds = np.stack([bond_i + 1, bond_j + 1], axis=1)
    ff.bond_list = bonds[np.lexsort((bonds[:,1], bonds[:,0]))]  # sort by col0, then col1

    # --- Step 3: Angles ---
    angles = []
    for j in range(n):
        neighbors = np.where(A[j])[0]
        for i in neighbors:
            for l in neighbors:
                if i < l:
                    angles.append((i + 1, j + 1, l + 1))
    angles = np.array(angles, dtype=int)
    ff.angle_list = angles[np.lexsort((angles[:,2], angles[:,1], angles[:,0]))]

    # --- Step 4: Dihedrals ---
    dihedrals = []
    for j in range(n):
        for l in np.where(A[j])[0]:
            for i in np.where(A[j])[0]:
                if i == l:
                    continue
                for m in np.where(A[l])[0]:
                    if m in (i, j, l):
                        continue
                    dih = canonical_dihedral(i+1, j+1, l+1, m+1)
                    dihedrals.append(dih)

    # Remove duplicates and sort
    ff.dihedral_list = np.array(sorted(set(dihedrals)), dtype=int)

    # --- Step 5: LJ pairs ---
    all_i, all_j = np.triu_indices(n, 1)
    mask_nonbond = (A[all_i, all_j] == 0)
    lj = np.stack([all_i[mask_nonbond] + 1, all_j[mask_nonbond] + 1], axis=1)
    ff.lj_list = lj[np.lexsort((lj[:,1], lj[:,0]))]

# def count_bonds_angles_dihedrals(ff, bo_threshold=0.0):
#     """
#     Count bonds, angles, dihedrals, and LJ pairs in the ff object.

#     ff should expose:
#     - nat: int (number of atoms)
#     - wbo: np.ndarray (bond order matrix)
#     - vander_matrix: np.ndarray (optional, used for diagnostics)
#     - count_bond, count_angle, count_dihedral, count_lj: int (set in-place)
#     """
#     n_atom = ff.nat

#     count_bond = 0
#     count_angle = 0
#     count_dihedral = 0

#     for i in range(n_atom):
#         for j in range(n_atom):
#             if i == j:
#                 continue
#             if ff.wbo[i, j] > bo_threshold:
#                 count_bond += 1
#             for l in range(n_atom):
#                 if l in (i, j):
#                     continue
#                 if ff.wbo[i, j] * ff.wbo[j, l] > bo_threshold:
#                     count_angle += 1
#                 for m in range(n_atom):
#                     if m in (i, j, l):
#                         continue
#                     if ff.wbo[i, j] * ff.wbo[j, l] * ff.wbo[l, m] > bo_threshold:
#                         count_dihedral += 1

#     # Symmetry corrections
#     ff.count_bond = count_bond // 2
#     ff.count_angle = count_angle // 2
#     ff.count_dihedral = count_dihedral // 2

#     # LJ count = all pairs - bonds
#     ff.count_lj = n_atom * (n_atom - 1) - ff.count_bond
#     if ff.count_lj < 0:
#         ff.count_lj = 0
