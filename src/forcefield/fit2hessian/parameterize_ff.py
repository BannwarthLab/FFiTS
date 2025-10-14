from src.datatype.structure_data import ForceField, StructuralInformation
from src.forcefield.fortran_energy.ff_energy import complete_hessian
from src.forcefield.fortran_energy.fortran_bindings import get_single_bond_hessian, get_single_angle_hessian, get_single_dihedral_hessian, get_single_repulsive_hessian
from typing import Optional
import numpy as np
import copy
import warnings
import molbar 


def calculate_hessian_rmsd(hessian_ff, hessian_ref, ndof, out_rmsd):
    """Compute RMSD between hessian_ff and hessian_ref and write scalar to out_rmsd (mutable).
    In Python we will just return the RMSD float."""
    diff = hessian_ff[:ndof, :ndof] - hessian_ref[:ndof, :ndof]
    # RMSD over matrix elements
    rmsd = np.sqrt(np.mean(diff**2))
    return rmsd


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def _atom_slice(atom_idx: int) -> slice:
    """Return the 3-row slice indices for a given atom (0-based)."""
    start = 3 * int(atom_idx)
    return slice(start, start + 3)

# --------------------------------------------------------------------
# Core functions (Python translation of Fortran module)
# --------------------------------------------------------------------

def fit_ff_to_hessian(struc: StructuralInformation,
                      ff: ForceField,
                      maxit_ex: Optional[int] = None,
                      stepsize_ex: Optional[float] = None,
                      threshold_ex: Optional[float] = None,
                      constant_repulsion_ex: Optional[bool] = None):
    """
    description
    """
    nat = int(ff.nat)
    3 * natndof = 3 * nat

    # allocate scratch arrays
    grd = np.zeros(3 * nat, dtype=np.float64)
    hessian_ff = np.zeros((3 * nat, 3 * nat), dtype=np.float64)

    # default parameters
    maxit = 1000 if maxit_ex is None else int(maxit_ex)
    stepsize = 0.05 if stepsize_ex is None else float(stepsize_ex)
    threshold = 0.001 if threshold_ex is None else float(threshold_ex)
    constant_repulsion = True if constant_repulsion_ex is None else bool(constant_repulsion_ex)

    counter = 0
    temp_old = 1.0
    temp = 1.0
    rmsd_gap = 0.5
    rmsdd = 1.0

    ff_new = copy.deepcopy(ff)
    
    print("--------------------- START OF FF FITTING ---------------------")
    print("Following parameters are used (maxit, stepsize, threshold):", maxit, stepsize, threshold)

    # Main iterative loop
    while (rmsd_gap >= threshold) and (counter < maxit):
        temp_old = temp
        temp = 0.0
        counter += 1
        hessian_ff.fill(0.0)

        # compute FF Hessian given current parameters
        hessian_ff = complete_hessian(struc.fortran_xyz, ff_new)

        # update bonds
        for f in range(int(hopot.count_bond)):
            update_bond(hopot, hessian_ff, f, stepsize)

        # update angles
        for f in range(int(hopot.count_angle)):
            update_angle(hopot, hessian_ff, f, stepsize)

        # update dihedrals
        for f in range(int(hopot.count_dihedral)):
            update_dihedral(hopot, hessian_ff, f, stepsize)

        # update LJ (repulsion) if not kept constant
        if not constant_repulsion:
            for f in range(int(hopot.count_lj)):
                update_repulsion(hopot, hessian_ff, f, stepsize)

        # compute RMSD between current FF Hessian and reference Hessian
        rmsdd = calculate_hessian_rmsd(hessian_ff, hopot.hessian, ndof, None)
        print("CYCLE", counter, "RMSD:", rmsdd)
        temp = rmsdd
        rmsd_gap = abs(temp_old - temp)

    return {"iterations": counter, "final_rmsd": rmsdd}


# -----------------------------
# single-parameter update steps
# -----------------------------
def update_bonds4fit(row, struc: StructuralInformation, hessian_ff: np.ndarray, stepsize: float):
    i = row['atoms'][0]
    j = row['atoms'][1]

    deriv1 = derivative_c_first_atomwise(struc.nat, struc.fortran_xyz,
                                         row['reference_value'],
                                         hessian_ff, struc.hessian,
                                         atom1=i, atom2=j, c=row['parameter'])
    deriv2 = derivative_c_second_atomwise(struc.nat, struc.fortran_xyz,
                                         row['reference_value'],
                                         hessian_ff, struc.hessian,
                                         atom1=i, atom2=j, c=row['parameter'])

    row['parameter'] = update_single_ffparam(row['parameter'], deriv1, deriv2, stepsize)


# -----------------------------
# single-parameter update steps
# -----------------------------
def update_bond(hopot, hessian_ff, position_in_list: int, stepsize: float):
    i = int(hopot.bond_list[0, position_in_list])
    j = int(hopot.bond_list[1, position_in_list])

    deriv1 = derivative_c_first_atomwise(hopot.nat, hopot.xyz0,
                                         hopot.bondlengths[position_in_list],
                                         hessian_ff, hopot.hessian,
                                         atom1=i, atom2=j, c=hopot.c_bond[i, j])
    deriv2 = derivative_c_second_atomwise(hopot.nat, hopot.xyz0,
                                          hopot.bondlengths[position_in_list],
                                          hessian_ff, hopot.hessian,
                                          atom1=i, atom2=j, c=hopot.c_bond[i, j])

    hopot.c_bond[i, j] = update_single_ffparam(hopot.c_bond[i, j], deriv1, deriv2, stepsize)


def update_angle(hopot, hessian_ff, position_in_list: int, stepsize: float):
    i = int(hopot.angle_list[0, position_in_list])
    j = int(hopot.angle_list[1, position_in_list])
    l = int(hopot.angle_list[2, position_in_list])

    deriv1 = derivative_c_first_atomwise(hopot.nat, hopot.xyz0,
                                         hopot.angles[position_in_list],
                                         hessian_ff, hopot.hessian,
                                         atom1=i, atom2=j, atom3=l, c=hopot.c_angle[i, j, l])
    deriv2 = derivative_c_second_atomwise(hopot.nat, hopot.xyz0,
                                          hopot.angles[position_in_list],
                                          hessian_ff, hopot.hessian,
                                          atom1=i, atom2=j, atom3=l, c=hopot.c_angle[i, j, l])

    hopot.c_angle[i, j, l] = update_single_ffparam(hopot.c_angle[i, j, l], deriv1, deriv2, stepsize)


def update_dihedral(hopot, hessian_ff, position_in_list: int, stepsize: float):
    i = int(hopot.dihedral_list[0, position_in_list])
    j = int(hopot.dihedral_list[1, position_in_list])
    l = int(hopot.dihedral_list[2, position_in_list])
    m = int(hopot.dihedral_list[3, position_in_list])

    deriv1 = derivative_c_first_atomwise(hopot.nat, hopot.xyz0,
                                         hopot.dihedrals[position_in_list],
                                         hessian_ff, hopot.hessian,
                                         atom1=i, atom2=j, atom3=l, atom4=m, c=hopot.c_dihedral[i, j, l, m])
    deriv2 = derivative_c_second_atomwise(hopot.nat, hopot.xyz0,
                                          hopot.dihedrals[position_in_list],
                                          hessian_ff, hopot.hessian,
                                          atom1=i, atom2=j, atom3=l, atom4=m, c=hopot.c_dihedral[i, j, l, m])

    hopot.c_dihedral[i, j, l, m] = update_single_ffparam(hopot.c_dihedral[i, j, l, m], deriv1, deriv2, stepsize)


def update_repulsion(hopot, hessian_ff, position_in_list: int, stepsize: float):
    i = int(hopot.lj_list[0, position_in_list])
    j = int(hopot.lj_list[1, position_in_list])

    # sigma = vander_matrix(i,j) / 2^(1/6)
    sigma = float(hopot.vander_matrix[i, j]) / (2.0 ** (1.0 / 6.0))

    deriv1 = lj_derivative_c_first_atomwise(hopot.nat, hopot.xyz0, hessian_ff, hopot.hessian,
                                            atom1=i, atom2=j, c=hopot.c_lj[i, j], sigma=sigma)
    deriv2 = lj_derivative_c_second_atomwise(hopot.nat, hopot.xyz0, hessian_ff, hopot.hessian,
                                             atom1=i, atom2=j, c=hopot.c_lj[i, j], sigma=sigma)

    hopot.c_lj[i, j] = update_single_ffparam(hopot.c_lj[i, j], deriv1, deriv2, stepsize)


def update_single_ffparam(val: float, deriv1: float, deriv2: float, stepsize: float) -> float:
    if deriv2 == 0.0:
        warnings.warn("Second derivative is zero; skipping parameter update (returning original value).")
        return val
    return val - deriv1 * (1.0 / deriv2) * stepsize


# -----------------------------
# derivative assembly wrappers
# -----------------------------
def derivative_c_second_atomwise(n_atom: int, geometry_ff: np.ndarray, val_ref: float,
                                 hess_ff: np.ndarray, hess_ref: np.ndarray,
                                 atom1: int, atom2: int, c: float,
                                 atom3: Optional[int] = None, atom4: Optional[int] = None) -> float:
    """
    Compute second derivative (scalar) wrt FF parameter c for specified atoms.
    Returns the scalar deriv value (already multiplied by required symmetry factor).
    """
    ndof = 3 * n_atom
    gradient = np.zeros(ndof, dtype=np.float64)
    hess_ff_single = np.zeros((ndof, ndof), dtype=np.float64)

    if (atom3 is not None) and (atom4 is not None):
        get_dihedral_hessian_four_atoms(geometry_ff, val_ref, atom1, atom2, atom3, atom4, c, gradient, hess_ff_single)
        deriv = 0.0
        deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom1, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom2, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom3, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv * 2.0  # hessian symmetry factor

    elif (atom3 is not None) and (atom4 is None):
        get_angle_hessian_three_atoms(geometry_ff, val_ref, atom1, atom2, atom3, c, gradient, hess_ff_single)
        deriv = 0.0
        deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv * 2.0

    else:
        # bond case
        get_bond_hessian_two_atoms(geometry_ff, val_ref, atom1, atom2, c, gradient, hess_ff_single)
        deriv = 0.0
        deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv * 2.0


def derivative_c_first_atomwise(n_atom: int, geometry_ff: np.ndarray, val_ref: float,
                                hess_ff: np.ndarray, hess_ref: np.ndarray,
                                atom1: int, atom2: int, c: float,
                                atom3: Optional[int] = None, atom4: Optional[int] = None) -> float:
    """
    Compute first derivative (scalar) wrt FF parameter c for specified atoms.
    """
    ndof = 3 * n_atom
    gradient = np.zeros(ndof, dtype=np.float64)
    hess_ff_single = np.zeros((ndof, ndof), dtype=np.float64)

    if (atom3 is not None) and (atom4 is not None):
        get_single_dihedral_hessian(geometry_ff, val_ref, atom1, atom2, atom3, atom4, c, gradient, hess_ff_single)
        s = 0.0
        s = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, s)
        s = get_sum_first_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, s)
        s = get_sum_first_c_deriv(c, atom1, atom4, hess_ff, hess_ref, hess_ff_single, s)
        s = get_sum_first_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, s)
        s = get_sum_first_c_deriv(c, atom2, atom4, hess_ff, hess_ref, hess_ff_single, s)
        s = get_sum_first_c_deriv(c, atom3, atom4, hess_ff, hess_ref, hess_ff_single, s)
        return s

    elif (atom3 is not None) and (atom4 is None):
        get_single_angle_hessian(geometry_ff, val_ref, atom1, atom2, atom3, c, gradient, hess_ff_single)
        s = 0.0
        s = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, s)
        s = get_sum_first_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, s)
        s = get_sum_first_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, s)
        return s

    else:
        get_single_bond_hessian(geometry_ff, val_ref, atom1, atom2, c, gradient, hess_ff_single)
        s = 0.0
        s = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, s)
        return s


# -----------------------------------------------------------
# Sum helpers — convert Fortran index ranges to Python slices
# -----------------------------------------------------------
def get_sum_first_c_deriv(c: float, atom1: int, atom2: int,
                          hess_ff: np.ndarray, hess_ref: np.ndarray, hess_ff_single: np.ndarray, acc: float) -> float:
    """
    Sum over 3x3 block between atom1 and atom2:
    acc += (hess_ff - hess_ref) * 4 * hess_ff_single / c
    """
    sl_i = _atom_slice(atom1)
    sl_j = _atom_slice(atom2)
    block_diff = hess_ff[sl_i, sl_j] - hess_ref[sl_i, sl_j]
    block_single = hess_ff_single[sl_i, sl_j]
    # element-wise sum
    acc += np.sum((block_diff * 4.0 * block_single) / c)
    return acc


def get_sum_second_c_deriv(c: float, atom1: int, atom2: int,
                           hess_ff: np.ndarray, hess_ref: np.ndarray, hess_ff_single: np.ndarray, acc: float) -> float:
    """
    acc += (2*hess_ff_single/c**2) * (2*hess_ff_single + hess_ff - hess_ref)
    """
    sl_i = _atom_slice(atom1)
    sl_j = _atom_slice(atom2)
    A = hess_ff_single[sl_i, sl_j]
    B = hess_ff[sl_i, sl_j] - hess_ref[sl_i, sl_j]
    acc += np.sum((2.0 * A / (c ** 2)) * (2.0 * A + B))
    return acc


# -----------------------------------------------------------
# LJ-specific wrappers
# -----------------------------------------------------------
def lj_derivative_c_second_atomwise(n_atom: int, geometry_ff: np.ndarray,
                                    hess_ff: np.ndarray, hess_ref: np.ndarray,
                                    atom1: int, atom2: int, c: float, sigma: float) -> float:
    ndof = 3 * n_atom
    gradient = np.zeros(ndof, dtype=np.float64)
    hess_ff_single = np.zeros((ndof, ndof), dtype=np.float64)

    get_lj_hessian_two_atoms(geometry_ff, atom1, atom2, c, sigma, gradient, hess_ff_single)
    deriv = 0.0
    deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
    return deriv * 2.0


def lj_derivative_c_first_atomwise(n_atom: int, geometry_ff: np.ndarray,
                                   hess_ff: np.ndarray, hess_ref: np.ndarray,
                                   atom1: int, atom2: int, c: float, sigma: float) -> float:
    ndof = 3 * n_atom
    gradient = np.zeros(ndof, dtype=np.float64)
    hess_ff_single = np.zeros((ndof, ndof), dtype=np.float64)

    get_lj_hessian_two_atoms(geometry_ff, atom1, atom2, c, sigma, gradient, hess_ff_single)
    deriv = 0.0
    deriv = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
    return deriv

