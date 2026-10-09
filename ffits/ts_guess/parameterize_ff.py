"""Fits force field parameters to a reference Hessian via Newton-style updates."""
import logging
from ffits.ts_guess.define_starting_parameters import fill_ff
from ffits.datatype.calculation_data import CalculationOptions
from ffits.datatype.structure_data import StructuralInformation, Structure
import ffits.forcefield.python_interface.fortran_bindings as fb
from ffits.io.print.details import print_ff_fitting
from typing import Optional
import numpy as np
import scipy.sparse as sp
import warnings

logger = logging.getLogger(__name__)


def parameterize_ff(struc: Structure, calcopt: CalculationOptions, priorities):
    """
    Parameterizes the FF of the given structure by fitting to the reference Hessian. The fitting process iteratively updates the FF parameters (bonds, angles, dihedrals, and optionally LJ repulsion) based on the difference between the current FF Hessian and the reference Hessian. FF object is filled in place.

    Args:
        struc (Structure): The structure whose FF is to be parameterized. Must have a reference Hessian in struc.info.hessian.
        calcopt (CalculationOptions): Calculation options containing fitting parameters such as max iterations, stepsize, and threshold for convergence.
        priorities (list): Atom priority information used for determining which parameters to update first (not implemented in this simplified version).
    """
    fill_ff(
        struc.ff,
        struc.info,
        repulsive_start=calcopt.ff_parameter_repulsion,
        priorities=priorities,
        bo_threshold=calcopt.bo_treshold,
    )
    fit_ff_to_hessian(
        struc,
        maxit=calcopt.ff_parameterization_maxiteration,
        stepsize=calcopt.ff_parameterization_stepsize,
        threshold=calcopt.ff_parameterization_threshold,
        constant_repulsion=calcopt.ff_parameterization_constant_repulsion,
    )


def calculate_hessian_rmsd(hessian_ff: np.ndarray, hessian_ref: np.ndarray, dim: int):
    """RMSD between the FF and reference Hessians, restricted to the first ``dim`` rows/cols."""
    diff = hessian_ff[:dim, :dim] - hessian_ref[:dim, :dim]
    rmsd = np.sqrt(np.mean(diff**2))
    return rmsd


def ff_fit_objective_function(
    hessian_ff: np.ndarray, hessian_ref: np.ndarray, dim: int
):
    """Sum of squared off-diagonal-block differences between the FF and reference Hessians (the fitting objective)."""
    if np.shape(hessian_ff) != (dim, dim):
        raise Exception("Hessian has the wrong dimension")
    if np.shape(hessian_ref) != (dim, dim):
        raise Exception("Hessian has the wrong dimension")
    diff = (hessian_ff - hessian_ref)[_offdiagonal_block_mask(dim)]
    return 0.5 * np.sum(diff**2)


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------


def _offdiagonal_block_mask(dim: int) -> np.ndarray:
    """Boolean (dim, dim) mask, True for Hessian entries that couple two *different* atoms.

    These are the only entries the objective looks at (the 3x3 diagonal blocks
    are ignored). ``matrix[mask]`` flattens the selected entries into a 1D vector.
    """
    atom = np.arange(dim) // 3
    return atom[:, None] != atom[None, :]


# --------------------------------------------------------------------
# Newton update for all FF parameters at once
# --------------------------------------------------------------------
# THE IDEA IN PLAIN WORDS
#
# We fit one parameter c_p per FF term (bond, angle, dihedral, ...) so that the FF
# Hessian H_ff matches the reference Hessian H_ref. Every term p adds its own
# "subhessian" u_p to the FF Hessian, and u_p grows with the force constant c_p**2:
#
#     H_ff = u_1 + u_2 + ... + u_P          with   u_p = c_p**2 * U_p
#
# U_p is the subhessian the term would have for force constant 1. It does not depend
# on c at all, so we compute it only once (build_unit_hessians) and rescale it with
# the current c_p whenever we need u_p.
#
# The objective only counts Hessian entries between *different* atoms:
#
#     O = 1/2 * sum over these entries of (H_ff - H_ref)**2
#
# Newton's method needs the gradient of O (one number per parameter) and the second
# derivatives of O (a P x P matrix). Both are built from the "slope" of the FF Hessian,
# i.e. how much every Hessian entry changes if c_p changes:
#
#     slope_p = dH_ff/dc_p = 2 * c_p * U_p          ( = 2 * u_p / c_p )
#
#     gradient[p]     = sum over entries of (H_ff - H_ref) * slope_p
#     hessian[p, q]   = sum over entries of slope_p * slope_q                 (p != q)
#     hessian[p, p]   = sum over entries of slope_p**2
#                       + 2 * sum over entries of (H_ff - H_ref) * U_p
#
# (the last term appears only on the diagonal: the slope of term p itself changes with c_p)
#
# and the update is     c_new = c - stepsize * solve(hessian, gradient)
#
# HOW THE DATA IS STORED
#
# A full 3N x 3N Hessian is reduced to the entries between different atoms and
# flattened to a vector of length M. Stacking the vectors of all P terms as rows gives
# the matrix ``unit_hessians`` (P x M): row p is the subhessian U_p of term p.
# It is a scipy *sparse* matrix (it only stores its non-zero entries, which works well
# because a term only touches a few atoms). ``@``, ``.T`` and ``.toarray()`` behave as
# for a normal matrix.


def _fitted_term_groups(ff, constant_repulsion: bool) -> list:
    """The FF terms that are fitted, as (table of terms, function computing one term's subhessian).

    The order bonds -> angles -> dihedrals -> repulsive is also the order of the
    parameters in the vector ``c``.
    """
    groups = [
        (ff.bonds, fb.get_single_bond_hessian),
        (ff.angles, fb.get_single_angle_hessian),
        (ff.dihedrals, fb.get_single_dihedral_hessian),
    ]
    if not constant_repulsion:
        groups.append((ff.repulsive, fb.get_single_repulsive_hessian))
    return groups


def _read_parameters(groups: list) -> np.ndarray:
    """Stack the parameters of all fitted terms into one vector ``c`` (bonds first, ...)."""
    return np.concatenate([df["parameter"].to_numpy(dtype=float) for df, _ in groups])


def _write_parameters(groups: list, c: np.ndarray) -> None:
    """Split the vector ``c`` up again and store it in the tables of terms (inverse of _read_parameters)."""
    first = 0
    for df, _ in groups:
        df["parameter"] = c[first : first + len(df)]
        first += len(df)


def build_unit_hessians(
    xyz: np.ndarray, nat: int, groups: list, mask: np.ndarray
) -> sp.csr_matrix:
    """Subhessians U_p of all fitted terms for force constant 1, one row per term (P x M).

    Called once before the fit, because U_p does not change while fitting: the
    subhessian of a term is exactly proportional to its force constant, so the
    subhessian for the current parameter is simply ``c_p**2 * U_p``.

    Args:
        xyz: geometry in the format expected by the Fortran routines.
        nat: number of atoms.
        groups: fitted terms, see :func:`_fitted_term_groups`.
        mask: True for the Hessian entries between different atoms that the objective uses.
    """
    rows = []
    for terms, subhessian_of_one_term in groups:
        for atoms, reference_value in zip(terms["atoms"], terms["reference_value"]):
            # Fortran adds the subhessian of this one term into a 3N x 3N array ...
            full_subhessian = np.zeros((3 * nat, 3 * nat), dtype=np.float64, order="F")
            force_constant = 1.0
            subhessian_of_one_term(
                xyz, np.asarray(atoms), reference_value, force_constant, full_subhessian
            )
            # ... of which we keep only the entries between different atoms, as a flat vector.
            rows.append(sp.csr_matrix(full_subhessian[mask]))

    if not rows:  
        return sp.csr_matrix((0, int(mask.sum())))
    return sp.vstack(rows, format="csr")


def objective_hessian_offdiagonal(dH_dc: sp.csr_matrix) -> np.ndarray:
    """Second derivatives d2O/(dc_p dc_q) of the objective for p != q, as a P x P matrix (zero diagonal).

    For two different parameters the objective only depends on the product of their
    slopes (their subhessians do not influence each other):

        d2O/(dc_p dc_q) = sum over entries of  slope_p * slope_q
                        = 4 * sum over entries of (u_p / c_p) * (u_q / c_q)

    which matches the hand derivation with u = c**2 * U. Terms that share no atom pair
    have no entries in common, so their element is zero.

    The same thing without matrix notation (slow, for reading only)::

        for p in range(P):
            for q in range(P):
                if p != q:
                    result[p, q] = sum(dH_dc[p, m] * dH_dc[q, m] for m in range(M))

    Args:
        dH_dc: (P x M) slopes of the Hessian entries, row p = dH_ff/dc_p.
    """
    # entry [p, q] = sum over m of dH_dc[p, m] * dH_dc[q, m]
    slope_products = (dH_dc @ dH_dc.T).toarray()
    # the diagonal is handled separately in objective_hessian_diagonal
    np.fill_diagonal(slope_products, 0.0)
    return slope_products


def objective_hessian_diagonal(
    unit_hessians: sp.csr_matrix, dH_dc: sp.csr_matrix, hessian_difference: np.ndarray
) -> np.ndarray:
    """Second derivatives d2O/dc_p**2 of the objective, one number per parameter (length P).

    Two contributions: the slope squared (like the off-diagonal case with q = p),
    plus a term from the Hessian difference, because the slope of term p itself
    changes with c_p (d2H_ff/dc_p**2 = 2 * U_p).
    """
    # sum over entries of slope_p**2 (sum of the squares of each row)
    slope_squared = np.asarray(dH_dc.power(2).sum(axis=1)).ravel()
    # 2 * sum over entries of U_p * (H_ff - H_ref)
    curvature_term = 2.0 * (unit_hessians @ hessian_difference)
    return slope_squared + curvature_term


def objective_derivatives(
    c: np.ndarray, unit_hessians: sp.csr_matrix, hessian_difference: np.ndarray
) -> tuple:
    """Gradient (length P) and second-derivative matrix (P x P) of the objective w.r.t. the parameters ``c``.

    Args:
        c: current parameters (length P).
        unit_hessians: (P x M) subhessians for force constant 1, see :func:`build_unit_hessians`.
        hessian_difference: (length M) entries of H_ff - H_ref between different atoms.
    """
    # Slope of every Hessian entry w.r.t. every parameter: row p of unit_hessians times 2 * c_p.
    dH_dc = sp.diags(2.0 * c) @ unit_hessians

    gradient = dH_dc @ hessian_difference

    # off-diagonal part (its diagonal is still zero) ...
    hessian = objective_hessian_offdiagonal(dH_dc)
    # ... plus the diagonal
    diagonal = objective_hessian_diagonal(unit_hessians, dH_dc, hessian_difference)
    np.fill_diagonal(hessian, diagonal)
    return gradient, hessian


def newton_step(
    gradient: np.ndarray, hessian: np.ndarray, stepsize: float
) -> np.ndarray:
    """Change of the parameters, ``-stepsize * hessian^-1 @ gradient``.

     If the system is
    singular (e.g. a parameter without curvature) we fall back to the
    least-squares solution, which leaves such directions unchanged.
    """
    try:
        step = np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Hessian of the objective is singular; using a least-squares Newton step."
        )
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
    return -stepsize * step


def fit_ff_to_hessian(
    struc: Structure,
    maxit: int = 1000,
    stepsize: float = 0.15,
    threshold: float = 0.0005,
    constant_repulsion: bool = True,
):
    """Iteratively fits all FF parameters to the structure's reference Hessian.

    Each cycle recomputes the FF Hessian and does one damped Newton step
    ``c <- c - stepsize * H^-1 g`` for all bond/angle/dihedral (and repulsive,
    unless kept constant) parameters at once, where ``g`` and ``H`` are the
    gradient and Hessian of the objective w.r.t. the parameters (see
    :func:`objective_derivatives`). It stops once the RMSD to the reference
    Hessian stops changing by more than ``threshold`` or ``maxit`` is reached.

    Args:
        struc (Structure): Structure with an already-filled FF and reference Hessian.
        maxit (int, optional): Maximum number of fitting iterations. Defaults to 1000.
        stepsize (float, optional): Newton step scaling. Defaults to 0.15.
        threshold (float, optional): Convergence threshold on the RMSD change between iterations. Defaults to 0.0005.
        constant_repulsion (bool, optional): If True, repulsive parameters are not updated. Defaults to True.

    Returns:
        dict: ``{"iterations", "final_rmsd", "final_objectiv_function"}``.
    """
    ff = struc.ff
    info = struc.info
    nat = ff.nat

    # Only Hessian entries between different atoms enter the objective. Indexing a
    # 3N x 3N matrix with this mask returns exactly these entries as a flat vector.
    between_atoms = _offdiagonal_block_mask(3 * nat)
    reference_entries = info.hessian[between_atoms]

    # Subhessian of every fitted term for force constant 1. It does not change during
    # the fit, so it is computed only once here (see build_unit_hessians).
    groups = _fitted_term_groups(ff, constant_repulsion)
    unit_hessians = build_unit_hessians(info.fortran_xyz, nat, groups, between_atoms)

    counter = 0
    temp_old = 1.0
    temp = 1.0
    rmsd_gap = 0.5
    rmsdd = 1.0

    print_ff_fitting(struc.path.xyz_filename)
    logger.info(
        f"Following parameters are used (maxit, stepsize, threshold): {maxit}, {stepsize}, {threshold}"
    )

    # Main iterative loop
    while (rmsd_gap >= threshold) and (counter < maxit):
        temp_old = temp
        temp = 0.0
        counter += 1

        # 1. FF Hessian for the current parameters and its difference to the reference
        hessian_ff = ff.get_hessian(info.fortran_xyz)
        hessian_difference = hessian_ff[between_atoms] - reference_entries

        # 2. One Newton step for all parameters at once
        c = _read_parameters(groups)
        gradient, obj_hessian = objective_derivatives(
            c, unit_hessians, hessian_difference
        )
        c_new = c + newton_step(gradient, obj_hessian, stepsize)
        _write_parameters(groups, c_new)

        # 3. Quality of the fit (measured with the FF Hessian from before this step)
        rmsdd = calculate_hessian_rmsd(hessian_ff, info.hessian, 3 * nat)
        obj_fun = 0.5 * hessian_difference @ hessian_difference
        logger.debug(f"CYCLE {counter} RMSD: {round(rmsdd, 4):.4f}")
        temp = rmsdd
        rmsd_gap = abs(temp_old - temp)

    logger.info(
        f"Fitting finished after {counter} iterations with an RMSD of {round(rmsdd, 4)} and {round(obj_fun, 4)}."
    )

    ff.write()
    return {
        "iterations": counter,
        "final_rmsd": rmsdd,
        "final_objectiv_function": obj_fun,
    }

