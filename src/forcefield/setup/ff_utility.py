import numpy as np
from src.datatype.structure_data import Structure, ForceField

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

def get_vander_matrix(nat, at, vander_values=VANDER_VALUES, factor=1.0):
    """
    Build van der Waals interaction matrix.

    Parameters
    ----------
    nat : int
        Number of atoms.
    at : array-like of int
        Atomic numbers (1-based indexing, as in Fortran).
    vander_values : np.ndarray
        Reference van der Waals radii (length 86).
    factor : float
        Scaling factor.

    Returns
    -------
    vander_matrix : np.ndarray (nat x nat)
    """
    # Convert atomic numbers to 0-based indices
    radii = vander_values[np.array(at) - 1] * factor
    # Broadcasting sum of pairwise radii
    return radii[:, None] + radii[None, :]

def get_bondlength(xyz, atom1, atom2):
    """Compute Euclidean bond length between two atoms."""
    return np.linalg.norm(xyz[atom1] - xyz[atom2])

def get_c_tables(hopot, repulsive_start_ex=None):
    """
    Compute c_bond, c_angle, c_dihedral, and c_lj tables for hopot object.

    ```
    hopot should expose:
    - xyz0: np.ndarray with atomic positions
    - wbo: np.ndarray (bond orders)
    - bond_list, angle_list, dihedral_list, lj_list: np.ndarray with atom indices
    - count_bond, count_angle, count_dihedral, count_lj: int
    - c_bond, c_angle, c_dihedral, c_lj: np.ndarray (to be filled in-place)
    """
    # --- Bonds ---
    hopot.c_bond.fill(0.0)
    for atom1, atom2 in hopot.bond_list[:, :hopot.count_bond].T:
        bondlength = get_bondlength(hopot.xyz0, atom1, atom2)
        hopot.c_bond[atom1, atom2] = hopot.wbo[atom1, atom2] / bondlength

    # --- Angles ---
    hopot.c_angle.fill(0.0)
    for atom1, atom2, atom3 in hopot.angle_list[:, :hopot.count_angle].T:
        bl1 = get_bondlength(hopot.xyz0, atom1, atom2)
        bl2 = get_bondlength(hopot.xyz0, atom2, atom3)
        product = hopot.wbo[atom1, atom2] * hopot.wbo[atom2, atom3]
        hopot.c_angle[atom1, atom2, atom3] = (product / (bl1 * bl2)) ** 0.5

    # --- Dihedrals ---
    hopot.c_dihedral.fill(0.0)
    for atom1, atom2, atom3, atom4 in hopot.dihedral_list[:, :hopot.count_dihedral].T:
        bl1 = get_bondlength(hopot.xyz0, atom1, atom2)
        bl2 = get_bondlength(hopot.xyz0, atom2, atom3)
        bl3 = get_bondlength(hopot.xyz0, atom3, atom4)
        product = hopot.wbo[atom1, atom2] * hopot.wbo[atom2, atom3] * hopot.wbo[atom3, atom4]
        hopot.c_dihedral[atom1, atom2, atom3, atom4] = (product / (bl1 * bl2 * bl3)) ** (1/3)

    # --- Lennard-Jones terms ---
    repulsive_start = 0.01 if repulsive_start_ex is None else repulsive_start_ex
    hopot.c_lj.fill(0.0)
    for atom1, atom2 in hopot.lj_list[:, :hopot.count_lj].T:
        hopot.c_lj[atom1, atom2] = repulsive_start



def define_relevant_bonds(struc: Structure, bo_threshold=0.0):
    """
    build bond, angle, dihedral, and LJ lists from a bond-order matrix
    """

    n = struc.ff.nat
    wbo = struc.info.wbo


    A = (wbo > bo_threshold).astype(int) # adjacency matrix where A[i, j] = 1 if atom i is bonded to atom j (bond order > threshold).
    np.fill_diagonal(A, 0)  # no self-bonds

    # --- Step 2: Bonds (edges) ---
    bond_i, bond_j = np.where(np.triu(A, 1))
    bond_list = np.stack([bond_i, bond_j], axis=1)

    # --- Step 3: Angles (2-step paths) ---
    # Paths of length 2: i - j - l. Takes j as central atom and looks for 
    angles = []
    for j in range(n):
        neighbors = np.where(A[j])[0]
        for i in neighbors:
            for l in neighbors:
                if i < l:  # avoid duplicates, i-j-l same as l-j-i
                    angles.append((i, j, l))
    angle_list = np.array(angles, dtype=int)

    # --- Step 4: Dihedrals (3-step paths) ---
    dihedrals = []
    for j in range(n):
        for l in np.where(A[j])[0]:        # j-l bond
            for i in np.where(A[j])[0]:    # j-i bond
                if i == l: continue
                for m in np.where(A[l])[0]:  # l-m bond
                    if m in (i, j, l): continue
                    # enforce canonical ordering to avoid duplicates
                    dihedrals.append((i, j, l, m))
    dihedral_list = np.unique(np.array(dihedrals, dtype=int), axis=0)

    # --- Step 5: LJ pairs ---
    # All pairs i<j that are NOT bonded
    all_i, all_j = np.triu_indices(n, 1)
    mask_nonbond = (A[all_i, all_j] == 0)
    lj_list = np.stack([all_i[mask_nonbond], all_j[mask_nonbond]], axis=1)

    return {
        "bond_list": bond_list,
        "angle_list": angle_list,
        "dihedral_list": dihedral_list,
        "lj_list": lj_list,
    }

def count_bonds_angles_dihedrals(hopot, bo_threshold=0.0):
    """
    Count bonds, angles, dihedrals, and LJ pairs in the hopot object.

    hopot should expose:
    - nat: int (number of atoms)
    - wbo: np.ndarray (bond order matrix)
    - vander_matrix: np.ndarray (optional, used for diagnostics)
    - count_bond, count_angle, count_dihedral, count_lj: int (set in-place)
    """
    n_atom = hopot.nat

    count_bond = 0
    count_angle = 0
    count_dihedral = 0

    for i in range(n_atom):
        for j in range(n_atom):
            if i == j:
                continue
            if hopot.wbo[i, j] > bo_threshold:
                count_bond += 1
            for l in range(n_atom):
                if l in (i, j):
                    continue
                if hopot.wbo[i, j] * hopot.wbo[j, l] > bo_threshold:
                    count_angle += 1
                for m in range(n_atom):
                    if m in (i, j, l):
                        continue
                    if hopot.wbo[i, j] * hopot.wbo[j, l] * hopot.wbo[l, m] > bo_threshold:
                        count_dihedral += 1

    # Symmetry corrections
    hopot.count_bond = count_bond // 2
    hopot.count_angle = count_angle // 2
    hopot.count_dihedral = count_dihedral // 2

    # LJ count = all pairs - bonds
    hopot.count_lj = n_atom * (n_atom - 1) - hopot.count_bond
    if hopot.count_lj < 0:
        hopot.count_lj = 0
