from src.datatype.structure_data import ForceField, StructuralInformation, Structure
from src.forcefield.python_interface.ff_energy import complete_hessian
import src.forcefield.python_interface.fortran_bindings as fb
from src.io.print.details import print_ff_fitting
from typing import Optional
import numpy as np
import warnings


def calculate_hessian_rmsd(hessian_ff, hessian_ref, dim):
    """rmsd"""
    diff = hessian_ff[:dim, :dim] - hessian_ref[:dim, :dim]
    rmsd = np.sqrt(np.mean(diff**2))
    return rmsd

def ff_fit_objective_function(hessian_ff: np.ndarray, hessian_ref: np.ndarray, dim: int):
    if np.shape(hessian_ff) != (dim, dim):
        raise Exception('Hessian has the wrong dimension')
    if np.shape(hessian_ref) != (dim, dim):
        raise Exception('Hessian has the wrong dimension')
    res = 0
    for i in range(dim):
        for j in range(dim):
            if i//3 != j//3:
                res += 0.5*(hessian_ff[i,j] - hessian_ref[i,j])**2
    return res


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def _atom_slice(atom_idx: int) -> slice:
    """Return the 3-row slice indices for a given atom (0-based)."""
    start = 3 * int(atom_idx)
    return slice(start, start + 3)


def fit_ff_to_hessian(struc: Structure, 
                      maxit: int = 1000,
                      stepsize: float = 0.15,
                      threshold: float = 0.0005,
                      constant_repulsion: bool = True):
    """
    description
    """
    ff = struc.ff
    info = struc.info
    nat = ff.nat

    hessian_ff = np.zeros((3 * nat, 3 * nat), dtype=np.float64, order='F')

    counter = 0
    temp_old = 1.0
    temp = 1.0
    rmsd_gap = 0.5
    rmsdd = 1.0

    
    print_ff_fitting(struc.path.xyz_filename)
    print("Following parameters are used (maxit, stepsize, threshold):", maxit, stepsize, threshold, '\n')

    # Main iterative loop
    while (rmsd_gap >= threshold) and (counter < maxit):
        temp_old = temp
        temp = 0.0
        counter += 1
        hessian_ff.fill(0.0)

        # compute FF Hessian given current parameters
        hessian_ff = ff.get_hessian(info.fortran_xyz)

        # update bonds
        new_values = []
        for row in ff.bonds.itertuples(): 
            new_param = update_bond(row, info, hessian_ff, stepsize)
            # print(row)
            # print(new_param, row.Index)
            new_values.append((row.Index, new_param))
        for idx, val in new_values:
            ff.bonds.at[idx, 'parameter'] = val
            # print(idx, val)

        # update angles
        new_values = []
        for row in ff.angles.itertuples(): 
            new_param = update_angle(row, info, hessian_ff, stepsize)
            new_values.append((row.Index, new_param))
        for idx, val in new_values:
            ff.angles.at[idx, 'parameter'] = val

        # update dihedrals
        new_values = []
        for row in ff.dihedrals.itertuples(): 
            new_param = update_dihedral(row, info, hessian_ff, stepsize)
            new_values.append((row.Index, new_param))
        for idx, val in new_values:
            ff.dihedrals.at[idx, 'parameter'] = val

        # update LJ (repulsion) if not kept constant
        if not constant_repulsion:
            new_values = []
            for row in ff.repulsive.itertuples(): 
                new_param = update_repulsive(row, info, hessian_ff, stepsize)
                new_values.append((row.Index, new_param))
            for idx, val in new_values:
                ff.repulsive.at[idx, 'parameter'] = val

        # compute RMSD between current FF Hessian and reference Hessian
        rmsdd = calculate_hessian_rmsd(hessian_ff, info.hessian, 3*nat)
        obj_fun = ff_fit_objective_function(hessian_ff, info.hessian, 3*nat)
        print(f"CYCLE {counter} RMSD: {round(rmsdd, 4):.4f}") #, "obj_fun", obj_fun) 
        temp = rmsdd
        rmsd_gap = abs(temp_old - temp)

    print(f'[INFO] Fitting finished after {counter} iterations with an RMSD of {round(rmsdd, 4)} and {round(obj_fun, 4)}.')

    ff.write()
    return {"iterations": counter, 
            "final_rmsd": rmsdd, 
            "final_objectiv_function": obj_fun}


# -----------------------------
# single-parameter update steps
# -----------------------------
def update_bond(row, info: StructuralInformation, hessian_ff: np.ndarray, stepsize: float):
    i = row.atoms[0]
    j = row.atoms[1]

    deriv1 = derivative_c_first_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        )
    deriv2 = derivative_c_second_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        )

    return update_single_ffparam(row.parameter, deriv1, deriv2, stepsize)

 
def update_angle(row, info: StructuralInformation, hessian_ff: np.ndarray, stepsize: float):
    i = row.atoms[0]
    j = row.atoms[1]
    l = row.atoms[2]

    deriv1 = derivative_c_first_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, c=row.parameter
        )
    deriv2 = derivative_c_second_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, c=row.parameter
        )

    return update_single_ffparam(row.parameter, deriv1, deriv2, stepsize)


def update_dihedral(row, info: StructuralInformation, hessian_ff: np.ndarray, stepsize: float):
    i = row.atoms[0]
    j = row.atoms[1]
    l = row.atoms[2]
    m = row.atoms[3]

    deriv1 = derivative_c_first_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, atom4=m, c=row.parameter
        )
    deriv2 = derivative_c_second_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, atom4=m, c=row.parameter
        )

    return update_single_ffparam(row.parameter, deriv1, deriv2, stepsize)

def update_repulsive(row, info: StructuralInformation, hessian_ff: np.ndarray, stepsize: float):
    i = row.atoms[0]
    j = row.atoms[1]

    deriv1 = repulsive_derivative_c_first_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        )
    deriv2 = repulsive_derivative_c_second_atomwise(
            info.nat, info.fortran_xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        )
    
    return update_single_ffparam(row.parameter, deriv1, deriv2, stepsize)


def update_single_ffparam(val: float, deriv1: float, deriv2: float, stepsize: float) -> float:
    if deriv2 == 0.0:
        warnings.warn("Second derivative is zero; skipping parameter update (returning original value).")
        return val
    return val - deriv1 * (1.0 / deriv2) * stepsize


def derivative_c_second_atomwise(nat: int, geometry_ff: np.ndarray, val_ref: float,
                                 hess_ff: np.ndarray, hess_ref: np.ndarray,
                                 atom1: int, atom2: int, c: float,
                                 atom3: Optional[int] = None, atom4: Optional[int] = None) -> float:
    """
    Compute second derivative (scalar) wrt FF parameter c for specified atoms.
    Returns the scalar deriv value (already multiplied by required symmetry factor).
    """
    
    hess_ff_single = np.zeros((3 * nat, 3 * nat), dtype=np.float64, order='F')
    deriv = 0.0

    if (atom3 is not None) and (atom4 is not None):
        fb.get_single_dihedral_hessian(geometry_ff, np.array([atom1, atom2, atom3, atom4]), val_ref, c**2, hess_ff_single)
        deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom1, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom2, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom3, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv * 2.0  # hessian symmetry factor

    elif (atom3 is not None) and (atom4 is None):
        fb.get_single_angle_hessian(geometry_ff, np.array([atom1, atom2, atom3]), val_ref, c**2, hess_ff_single)
        deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_second_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv * 2.0

    else:
        # bond case
        fb.get_single_bond_hessian(geometry_ff, np.array([atom1, atom2]), val_ref, c**2, hess_ff_single)
        deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv * 2.0


def derivative_c_first_atomwise(nat: int, geometry_ff: np.ndarray, val_ref: float,
                                hess_ff: np.ndarray, hess_ref: np.ndarray,
                                atom1: int, atom2: int, c: float,
                                atom3: Optional[int] = None, atom4: Optional[int] = None) -> float:
    """
    Compute first derivative (scalar) wrt FF parameter c for specified atoms.
    """
    hess_ff_single = np.zeros((3 * nat, 3 * nat), dtype=np.float64, order='F')
    deriv = 0.0
    if (atom3 is not None) and (atom4 is not None):
        fb.get_single_dihedral_hessian(geometry_ff, np.array([atom1, atom2, atom3, atom4]), val_ref, c**2, hess_ff_single)
        deriv = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_first_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_first_c_deriv(c, atom1, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_first_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_first_c_deriv(c, atom2, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_first_c_deriv(c, atom3, atom4, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv

    elif (atom3 is not None) and (atom4 is None):
        fb.get_single_angle_hessian(geometry_ff, np.array([atom1, atom2, atom3]), val_ref, c**2, hess_ff_single)
        deriv = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_first_c_deriv(c, atom1, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        deriv = get_sum_first_c_deriv(c, atom2, atom3, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv

    else:
        fb.get_single_bond_hessian(geometry_ff, [atom1, atom2], val_ref, c**2, hess_ff_single)
        deriv = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
        return deriv



def get_sum_first_c_deriv(c: float, atom1: int, atom2: int,
                          hess_ff: np.ndarray, hess_ref: np.ndarray, hess_ff_single: np.ndarray, acc: float) -> float:
    """
    Sum over 3x3 block between atom1 and atom2:
    acc += (hess_ff - hess_ref) * 4 * hess_ff_single / c
    """
    sl_i = _atom_slice(atom1)
    sl_j = _atom_slice(atom2)
    block_diff = hess_ff[sl_i, sl_j] - hess_ref[sl_i, sl_j]
    # print(block_diff)
    block_single = hess_ff_single[sl_i, sl_j]
    print(hess_ff_single[sl_i, sl_j])
    # element-wise sum
    acc += np.sum((block_diff * 4.0 * block_single) / c)
    return acc


def get_sum_second_c_deriv(c: float, atom1: int, atom2: int,
                           hess_ff: np.ndarray, hess_ref: np.ndarray, hess_ff_single: np.ndarray, acc: float) -> float:
    """
    acc += (2*hess_ff_single/c**2) * (2*hess_ff_single + hess_ff - hess_ref)
    """
    # print(hess_ff_single)
    sl_i = _atom_slice(atom1)
    sl_j = _atom_slice(atom2)
    A = hess_ff_single[sl_i, sl_j]
    B = hess_ff[sl_i, sl_j] - hess_ref[sl_i, sl_j]
    acc += np.sum((2.0 * A / (c ** 2)) * (2.0 * A + B))
    return acc


# -----------------------------------------------------------
# LJ-specific wrappers
# -----------------------------------------------------------

def repulsive_derivative_c_first_atomwise(nat: int, geometry_ff: np.ndarray, sigma: float,
                                   hess_ff: np.ndarray, hess_ref: np.ndarray,
                                   atom1: int, atom2: int, c: float) -> float:
    
    hess_ff_single = np.zeros((3 * nat, 3 * nat), dtype=np.float64, order='F')

    fb.get_single_repulsive_hessian(geometry_ff, np.array([atom1, atom2]), c**2, sigma, hess_ff_single)
    deriv = 0.0
    deriv = get_sum_first_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
    return deriv


def repulsive_derivative_c_second_atomwise(nat: int, geometry_ff: np.ndarray, sigma: float,
                                    hess_ff: np.ndarray, hess_ref: np.ndarray,
                                    atom1: int, atom2: int, c: float) -> float:
    
    hess_ff_single = np.zeros((3 * nat, 3 * nat), dtype=np.float64, order='F')

    fb.get_single_repulsive_hessian(geometry_ff, np.array([atom1, atom2]), c**2, sigma, hess_ff_single)
    deriv = 0.0
    deriv = get_sum_second_c_deriv(c, atom1, atom2, hess_ff, hess_ref, hess_ff_single, deriv)
    return deriv * 2
