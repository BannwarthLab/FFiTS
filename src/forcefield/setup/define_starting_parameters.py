import numpy as np
from src.datatype.structure_data import StructuralInformation, ForceField
from src.forcefield.fortran_energy.geometry_calc import angle, bondlength, dihedral_angle



def get_c_tables(ff: ForceField, info: StructuralInformation, repulsive_start_ex=None):
    """
    Compute c_bond, c_angle, c_dihedral, and c_lj tables for ForceField object.
    """
    for i in range(len(ff.bond_list)):
        atom1, atom2 = ff.bond_list[i, :]
        atom1 = atom1 - 1
        atom2 = atom2 - 1
        bl = bondlength(info.xyz, atom1, atom2)
        if info.bo_matrix[atom1, atom2] * bl == 0:
            raise Exception(ZeroDivisionError(f'Division by zero attempted for atoms {atom1, atom2}.'))
        ff.c_bond[(atom1+1, atom2+1)] = info.bo_matrix[atom1, atom2] / bl

    # --- Angles ---
    for i in range(len(ff.angle_list)):
        atom1, atom2, atom3 = ff.angle_list[i, :]
        atom1 = atom1 - 1
        atom2 = atom2 - 1
        atom3 = atom3 - 1
        bl1 = bondlength(info.xyz, atom1, atom2)
        bl2 = bondlength(info.xyz, atom2, atom3)
        product = info.bo_matrix[atom1, atom2] * info.bo_matrix[atom2, atom3]
        if bl1 * bl2 * product == 0:
            raise Exception(ZeroDivisionError(f'Division by zero attempted for atoms {atom1, atom2, atom3}.'))
        ff.c_angle[(atom1+1, atom2+1, atom3+1)] = (product / (bl1 * bl2)) ** 0.5

    # --- Dihedrals ---
    for i in range(len(ff.dihedral_list)):
        atom1, atom2, atom3, atom4 = ff.dihedral_list[i, :]
        atom1 = atom1 - 1
        atom2 = atom2 - 1
        atom3 = atom3 - 1
        atom4 = atom4 - 1
        bl1 = bondlength(info.xyz, atom1, atom2)
        bl2 = bondlength(info.xyz, atom2, atom3)
        bl3 = bondlength(info.xyz, atom3, atom4)
        product = (info.bo_matrix[atom1, atom2] *
                   info.bo_matrix[atom2, atom3] *
                   info.bo_matrix[atom3, atom4])
        if bl1 * bl2 * bl3 * product == 0:
            raise Exception(ZeroDivisionError(f'Division by zero attempted for atoms {atom1, atom2, atom3, atom4}.'))
        ff.c_dihedral[(atom1+1, atom2+1, atom3+1, atom4+1)] = (product / (bl1 * bl2 * bl3)) ** (1/3)

    # --- Lennard-Jones terms ---
    repulsive_start = 0.01 if repulsive_start_ex is None else repulsive_start_ex
    for i in range(len(ff.lj_list)):
        atom1, atom2 = ff.lj_list[i, :]
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

def get_ff_reference_values(ff: ForceField, info: StructuralInformation):
    for a in range(len(ff.bond_list)):
        i = ff.bond_list[0, a] - 1  
        j = ff.bond_list[1, a] - 1
        ff.bondlengths.append(bondlength(info.xyz, i, j))

    # Angles
    for a in range(len(ff.angle_list)):
        i = ff.angle_list[0, a] - 1
        j = ff.angle_list[1, a] - 1
        l = ff.angle_list[2, a] - 1
        ff.angles.append(angle(info.xyz, i, j, l))
    
    # Dihedrals
    for a in range(len(ff.dihedral_list)):
        i = ff.dihedral_list[0, a] - 1
        j = ff.dihedral_list[1, a] - 1
        l = ff.dihedral_list[2, a] - 1
        m = ff.dihedral_list[3, a] - 1
        ff.dihedrals.append(dihedral_angle(info.xyz, i, j, l, m))

    # Lennard-Jones terms
    for a in range(len(ff.lj_list)):
        i = ff.lj_list[0, a] - 1
        j = ff.lj_list[1, a] - 1
        ff.sigmas.append(info.vander_matrix[i, j] / (2**(1/6)))

def check_correct_ff_initialization(ff: ForceField):
    assert all(ff.c_bond.values())
    assert all(ff.c_angle.values())
    assert all(ff.c_dihedral.values())
    assert all(ff.c_lj.values())

def setup_unparameterized_forcefield(info: StructuralInformation, ff_filename: str, readff: bool = False) -> ForceField:
    ff = ForceField(info.nat, ff_filename, readff=readff)
    define_relevant_bonds(ff, info)
    get_c_tables(ff, info)
    return ff
