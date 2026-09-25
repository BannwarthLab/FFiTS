import numpy as np
import os
from ffits.datatype.structure_data import (
    StructuralInformation,
)

from ffits.datatype.forcefield_data import ForceField
from ffits.ts_guess.parameterize_ff import (
    _fitted_term_groups,
    _offdiagonal_block_mask,
    _read_parameters,
    build_unit_hessians,
    objective_derivatives,
    ff_fit_objective_function,
    fit_ff_to_hessian,
)
from ffits.forcefield.python_interface.ff_energy import complete_hessian
from ffits.io.reader import readin_xyz, read_wbo_file, read_xtb_hessian
from ffits.ts_guess.define_starting_parameters import fill_ff
import copy



def num_first_derivative(
    ff: ForceField, info: StructuralInformation, delta: float = 1e-5
):
    xyz = info.fortran_xyz
    derivatives = []
    for i in range(len(ff.bonds)):
        ff_m = copy.deepcopy(ff)
        ff_m.bonds.iloc[i, ff_m.bonds.columns.get_loc("parameter")] -= delta
        hess_m = complete_hessian(xyz, ff_m)
        res_m = ff_fit_objective_function(hess_m, info.hessian, 3 * info.nat)

        ff_p = copy.deepcopy(ff)
        ff_p.bonds.iloc[i, ff_m.bonds.columns.get_loc("parameter")] += delta
        hess_p = complete_hessian(xyz, ff_p)
        res_p = ff_fit_objective_function(hess_p, info.hessian, 3 * info.nat)

        derivatives.append((res_p - res_m) / (delta * 2))

    for i in range(len(ff.angles)):
        ff_m = copy.deepcopy(ff)
        ff_m.angles.iloc[i, ff_m.angles.columns.get_loc("parameter")] -= delta
        hess_m = complete_hessian(xyz, ff_m)
        res_m = ff_fit_objective_function(hess_m, info.hessian, 3 * info.nat)

        ff_p = copy.deepcopy(ff)
        ff_p.angles.iloc[i, ff_m.angles.columns.get_loc("parameter")] += delta
        hess_p = complete_hessian(xyz, ff_p)
        res_p = ff_fit_objective_function(hess_p, info.hessian, 3 * info.nat)

        derivatives.append((res_p - res_m) / (delta * 2))

    for i in range(len(ff.dihedrals)):
        ff_m = copy.deepcopy(ff)
        ff_m.dihedrals.iloc[i, ff_m.dihedrals.columns.get_loc("parameter")] -= delta
        hess_m = complete_hessian(xyz, ff_m)
        res_m = ff_fit_objective_function(hess_m, info.hessian, 3 * info.nat)

        ff_p = copy.deepcopy(ff)
        ff_p.dihedrals.iloc[i, ff_m.dihedrals.columns.get_loc("parameter")] += delta
        hess_p = complete_hessian(xyz, ff_p)
        res_p = ff_fit_objective_function(hess_p, info.hessian, 3 * info.nat)

        derivatives.append((res_p - res_m) / (delta * 2))

    for i in range(len(ff.repulsive)):
        ff_m = copy.deepcopy(ff)
        ff_m.repulsive.iloc[i, ff_m.repulsive.columns.get_loc("parameter")] -= delta
        hess_m = complete_hessian(xyz, ff_m)
        res_m = ff_fit_objective_function(hess_m, info.hessian, 3 * info.nat)

        ff_p = copy.deepcopy(ff)
        ff_p.repulsive.iloc[i, ff_m.repulsive.columns.get_loc("parameter")] += delta
        hess_p = complete_hessian(xyz, ff_p)
        res_p = ff_fit_objective_function(hess_p, info.hessian, 3 * info.nat)

        derivatives.append((res_p - res_m) / (delta * 2))
    return derivatives



def analy_full_second_derivative(ff: ForceField, info: StructuralInformation):
    """Analytic gradient and full (diagonal + off-diagonal) Hessian of the objective
    w.r.t. all FF parameters, from the matrix form used in the Newton update."""
    groups = _fitted_term_groups(ff, constant_repulsion=False)
    mask = _offdiagonal_block_mask(3 * info.nat)
    unit_hessians = build_unit_hessians(info.fortran_xyz, info.nat, groups, mask)
    residual = complete_hessian(info.fortran_xyz, ff)[mask] - info.hessian[mask]
    return objective_derivatives(_read_parameters(groups), unit_hessians, residual)


def num_second_derivative(
    ff: ForceField, info: StructuralInformation, delta: float = 1e-4
):
    xyz = info.fortran_xyz

    # Calculate total number of parameters
    n_params = len(ff.bonds) + len(ff.angles) + len(ff.dihedrals) + len(ff.repulsive)

    # Initialize Hessian matrix
    hessian = np.zeros((n_params, n_params))

    # Helper function to get parameter lists in order
    def get_all_params():
        params = []
        for idx, row in enumerate(ff.bonds.itertuples()):
            params.append(("bond", idx, row))
        for idx, row in enumerate(ff.angles.itertuples()):
            params.append(("angle", idx, row))
        for idx, row in enumerate(ff.dihedrals.itertuples()):
            params.append(("dihedral", idx, row))
        for idx, row in enumerate(ff.repulsive.itertuples()):
            params.append(("repulsive", idx, row))
        return params

    params = get_all_params()

    # Helper function to perturb a parameter
    def perturb_ff(ff_copy, param_type, param_idx, delta_val):
        if param_type == "bond":
            ff_copy.bonds.iloc[
                param_idx, ff_copy.bonds.columns.get_loc("parameter")
            ] += delta_val
        elif param_type == "angle":
            ff_copy.angles.iloc[
                param_idx, ff_copy.angles.columns.get_loc("parameter")
            ] += delta_val
        elif param_type == "dihedral":
            ff_copy.dihedrals.iloc[
                param_idx, ff_copy.dihedrals.columns.get_loc("parameter")
            ] += delta_val
        elif param_type == "repulsive":
            ff_copy.repulsive.iloc[
                param_idx, ff_copy.repulsive.columns.get_loc("parameter")
            ] += delta_val

    # Calculate diagonal and off-diagonal elements
    for i in range(n_params):
        for j in range(i, n_params):
            if i == j:
                # Diagonal: second derivative with respect to same parameter
                param_type_i, idx_i, _ = params[i]

                hess_center = complete_hessian(xyz, ff)
                energy_center = ff_fit_objective_function(
                    hess_center, info.hessian, 3 * info.nat
                )

                ff_m = copy.deepcopy(ff)
                perturb_ff(ff_m, param_type_i, idx_i, -2 * delta)
                hess_m = complete_hessian(xyz, ff_m)
                energy_m = ff_fit_objective_function(hess_m, info.hessian, 3 * info.nat)

                ff_p = copy.deepcopy(ff)
                perturb_ff(ff_p, param_type_i, idx_i, 2 * delta)
                hess_p = complete_hessian(xyz, ff_p)
                energy_p = ff_fit_objective_function(hess_p, info.hessian, 3 * info.nat)

                hessian[i, j] = (energy_m + energy_p - 2 * energy_center) / (
                    4.0 * delta**2
                )

            else:
                # Off-diagonal: mixed partial derivative
                param_type_i, idx_i, _ = params[i]
                param_type_j, idx_j, _ = params[j]

                # Energy at (0, 0)
                hess_00 = complete_hessian(xyz, ff)
                energy_00 = ff_fit_objective_function(
                    hess_00, info.hessian, 3 * info.nat
                )

                # Energy at (+delta_i, +delta_j)
                ff_pp = copy.deepcopy(ff)
                perturb_ff(ff_pp, param_type_i, idx_i, delta)
                perturb_ff(ff_pp, param_type_j, idx_j, delta)
                hess_pp = complete_hessian(xyz, ff_pp)
                energy_pp = ff_fit_objective_function(
                    hess_pp, info.hessian, 3 * info.nat
                )

                # Energy at (+delta_i, -delta_j)
                ff_pm = copy.deepcopy(ff)
                perturb_ff(ff_pm, param_type_i, idx_i, delta)
                perturb_ff(ff_pm, param_type_j, idx_j, -delta)
                hess_pm = complete_hessian(xyz, ff_pm)
                energy_pm = ff_fit_objective_function(
                    hess_pm, info.hessian, 3 * info.nat
                )

                # Energy at (-delta_i, +delta_j)
                ff_mp = copy.deepcopy(ff)
                perturb_ff(ff_mp, param_type_i, idx_i, -delta)
                perturb_ff(ff_mp, param_type_j, idx_j, delta)
                hess_mp = complete_hessian(xyz, ff_mp)
                energy_mp = ff_fit_objective_function(
                    hess_mp, info.hessian, 3 * info.nat
                )

                # Energy at (-delta_i, -delta_j)
                ff_mm = copy.deepcopy(ff)
                perturb_ff(ff_mm, param_type_i, idx_i, -delta)
                perturb_ff(ff_mm, param_type_j, idx_j, -delta)
                hess_mm = complete_hessian(xyz, ff_mm)
                energy_mm = ff_fit_objective_function(
                    hess_mm, info.hessian, 3 * info.nat
                )

                # Mixed partial derivative
                mixed_deriv = (energy_pp - energy_pm - energy_mp + energy_mm) / (
                    4.0 * delta**2
                )
                hessian[i, j] = mixed_deriv
                hessian[j, i] = mixed_deriv  # Symmetric matrix

    return hessian

