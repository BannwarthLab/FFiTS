import numpy as np
import pandas as pd
from src.datatype.structure_data import StructuralInformation, ForceField
from src.forcefield.fortran_energy.geometry_calc import angle, bondlength, dihedral_angle



def canonical_dihedral(i, j, l, m):
    """
    Return a canonical ordering for a dihedral (i, j, l, m),
    so that (i,j,l,m) and (m,l,j,i) collapse to the same tuple.
    """
    forward = (i, j, l, m)
    reverse = (m, l, j, i)
    return min(forward, reverse)


# def check_noNaN_ff_initialization(ff: ForceField):
#     assert all(ff.c_bond.values())
#     assert all(ff.c_angle.values())
#     assert all(ff.c_dihedral.values())
#     assert all(ff.c_lj.values())

# def setup_unparameterized_forcefield(info: StructuralInformation, ff_filename: str, readff: bool = False) -> ForceField:
#     ff = ForceField(info.nat, ff_filename, readff=readff)
#     define_relevant_bonds(ff, info)
#     get_c_tables(ff, info)
#     return ff

def fill_ff(ff: ForceField, info: StructuralInformation, bo_threshold: float = 0.0, repulsive_start_ex: float = None):
    """
    Build and fill ForceField DataFrames (bonds, angles, dihedrals, repulsive)
    with 0-based atom indices, reference values, and parameters.

    Uses imported geometry functions:
        bondlength(), angle(), dihedral_angle()
    """

    repulsive_start = 0.01 if repulsive_start_ex is None else repulsive_start_ex
    n = ff.nat
    wbo = info.bo_matrix

    # --- Step 1: adjacency matrix ---
    A = (wbo > bo_threshold).astype(int)
    np.fill_diagonal(A, 0)

    # ============================================================
    # Subfunctions
    # ============================================================

    # ---- Atom generation ----
    def get_bond_atoms(A):
        i, j = np.where(np.triu(A, 1))
        bonds = np.stack([i, j], axis=1)
        bonds = bonds[np.lexsort((bonds[:, 1], bonds[:, 0]))]
        return [np.array([a, b], dtype=int) for a, b in bonds]

    def get_angle_atoms(A):
        angles = []
        for j in range(n):
            neighbors = np.where(A[j])[0]
            for i in neighbors:
                for k in neighbors:
                    if i < k:
                        angles.append((i, j, k))
        if len(angles) == 0:
            return []
        angles = np.array(angles)[np.lexsort((np.array(angles)[:, 2],
                                              np.array(angles)[:, 1],
                                              np.array(angles)[:, 0]))]
        return [np.array([i, j, k], dtype=int) for i, j, k in angles]

    def get_dihedral_atoms(A):
        dihedrals = []
        for j in range(n):
            for l in np.where(A[j])[0]:
                for i in np.where(A[j])[0]:
                    if i == l:
                        continue
                    for m in np.where(A[l])[0]:
                        if m in (i, j, l):
                            continue
                        dihedrals.append(canonical_dihedral(i, j, l, m))
        dihedrals = sorted(set(dihedrals))
        return [np.array([i, j, k, l], dtype=int) for i, j, k, l in dihedrals]

    def get_repulsive_atoms(A):
        all_i, all_j = np.triu_indices(n, 1)
        mask = (A[all_i, all_j] == 0)
        pairs = np.stack([all_i[mask], all_j[mask]], axis=1)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        return [np.array([i, j], dtype=int) for i, j in pairs]

    # ---- Reference calculations ----
    def ref_bond(atoms):
        i, j = atoms
        return bondlength(info.fortran_xyz, i, j)

    def ref_angle(atoms):
        i, j, k = atoms
        return angle(info.fortran_xyz, i, j, k)

    def ref_dihedral(atoms):
        i, j, k, l = atoms
        return dihedral_angle(info.fortran_xyz, i, j, k, l)

    def ref_repulsive(atoms):
        i, j = atoms
        return info.vander_matrix[i, j] / (2 ** (1 / 6))

    # ---- Parameter calculations ----
    def param_bond(atoms):
        i, j = atoms
        bl = bondlength(info.fortran_xyz, i, j)
        bo = info.bo_matrix[i, j]
        if bo * bl == 0:
            raise ZeroDivisionError(f"Division by zero for bond {atoms.tolist()}.")
        return bo / bl

    def param_angle(atoms):
        i, j, k = atoms
        bl1 = bondlength(info.fortran_xyz, i, j)
        bl2 = bondlength(info.fortran_xyz, j, k)
        prod = info.bo_matrix[i, j] * info.bo_matrix[j, k]
        if bl1 * bl2 * prod == 0:
            raise ZeroDivisionError(f"Division by zero for angle {atoms.tolist()}.")
        return (prod / (bl1 * bl2)) ** 0.5

    def param_dihedral(atoms):
        i, j, k, l = atoms
        bl1 = bondlength(info.fortran_xyz, i, j)
        bl2 = bondlength(info.fortran_xyz, j, k)
        bl3 = bondlength(info.fortran_xyz, k, l)
        prod = info.bo_matrix[i, j] * info.bo_matrix[j, k] * info.bo_matrix[k, l]
        if bl1 * bl2 * bl3 * prod == 0:
            raise ZeroDivisionError(f"Division by zero for dihedral {atoms.tolist()}.")
        return (prod / (bl1 * bl2 * bl3)) ** (1 / 3)

    def param_repulsive(_atoms):
        return repulsive_start

    # ============================================================
    # Assemble ForceField DataFrames
    # ============================================================

    ff.bonds = pd.DataFrame({
        "type": "bonds",
        "atoms": get_bond_atoms(A)
    })
    ff.bonds["reference_value"] = ff.bonds["atoms"].apply(ref_bond)
    ff.bonds["parameter"] = ff.bonds["atoms"].apply(param_bond)

    ff.angles = pd.DataFrame({
        "type": "angles",
        "atoms": get_angle_atoms(A)
    })
    ff.angles["reference_value"] = ff.angles["atoms"].apply(ref_angle)
    ff.angles["parameter"] = ff.angles["atoms"].apply(param_angle)

    ff.dihedrals = pd.DataFrame({
        "type": "dihedrals",
        "atoms": get_dihedral_atoms(A)
    })
    ff.dihedrals["reference_value"] = ff.dihedrals["atoms"].apply(ref_dihedral)
    ff.dihedrals["parameter"] = ff.dihedrals["atoms"].apply(param_dihedral)

    ff.repulsive = pd.DataFrame({
        "type": "repulsive",
        "atoms": get_repulsive_atoms(A)
    })
    ff.repulsive["reference_value"] = ff.repulsive["atoms"].apply(ref_repulsive)
    ff.repulsive["parameter"] = ff.repulsive["atoms"].apply(param_repulsive)
