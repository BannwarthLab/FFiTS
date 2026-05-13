import logging
from collections.abc import Callable
import numpy as np
from ffits.datatype.structure_data import ForceField, StructuralInformation
from ffits.external.molbar import (
    anc_optimizer,
    failed_anc_opt,
    write_last_valid_xyz,
    scipy_optimizer,
)
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)
from ffits.io.print.details import print_optimization_start, print_optimization_end
from ffits.io.reader import readin_xyz
from ffits.datatype.calculation_data import TSCalculationOptions
from ffits.utils.temp_dir_manager import TempDirManager
from typing import Optional
import sys

logger = logging.getLogger(__name__)


def optimize_with_forcefield(
    info: StructuralInformation,
    ff: ForceField,
    optimizer: Callable,
    calcoptions: TSCalculationOptions = None,
    trajectory_filename: str = "trajectory.xyz",
    final_geometry_filename: str = "optimized.xyz",
    opt_stdout_filename: str = "ts_optimization.out",
):
    """Optimizes a structure using a given force field and optimizer. Currently supports the anc_optimizer from MolBar, but can be extended to other optimizers as needed. If no calculation options are provided, default values will be used.

    Args:
        info (StructuralInformation): Structural information of the molecule to be optimized, including coordinates, atom types, etc.
        ff (ForceField): Force field object containing the parameters and methods for energy, gradient, and hessian calculations which is the potential for the optimization
        optimizer (Callable): The optimizer to be used for the optimization.
        calcoptions (TSCalculationOptions, optional): The calculation options for the optimization. Defaults to None.
        trajectory_filename (str, optional): Filename of the trajectory file. Defaults to "trajectory.xyz".
        final_geometry_filename (str, optional): Filename of the final geometry file. Defaults to "optimized.xyz".
        opt_stdout_filename (str, optional): Filename of the optimization stdout file. Defaults to "ts_optimization.out".

    Raises:
        Exception: If the specified optimizer is not recognized or if required calculator functions are not defined in the force field object when using the anc_optimizer.

    Returns:
        converged (bool): Whether the optimization converged or not
        energy (float): Final energy of the optimized structure
        final_geom (np.ndarray): Final geometry of the optimized structure in the shape (nat, 3)
    """
    if calcoptions is None:
        logger.warning(
            "No calculation options provided for optimization. Using default values."
        )
        calcoptions = TSCalculationOptions()

    orig_stdout = sys.stdout
    f = open(opt_stdout_filename, "w")
    sys.stdout = f

    if optimizer == anc_optimizer:
        converged, energy, final_geom, steps, time, message = optimizer(
            info.xyz,
            ff,
            info.atom_types,
            x_tol=calcoptions.molbar_optimizer_x_tol,
            e_tol=calcoptions.molbar_optimizer_e_tol,
            trajectory_filename=trajectory_filename,
            final_geometry_filename=final_geometry_filename,
            max_micro_steps=calcoptions.molbar_optimizer_max_micro_steps,
        )
        sys.stdout = orig_stdout
        f.close()

        if failed_anc_opt(opt_stdout_filename):
            logger.warning(
                f"The last valid structure of the optimization trajectory is written to {final_geometry_filename}."
            )
            write_last_valid_xyz()

    elif optimizer == scipy_optimizer:
        raise Exception(
            "Scipy optimizer is currently not supported in the optimization interface. Please use the anc_optimizer from MolBar."
        )
        result = scipy_optimizer(info.xyz, ff, info)
        sys.stdout = orig_stdout
        f.close()
        return result

    else:
        sys.stdout = orig_stdout
        f.close()
        raise Exception(f"Optimizer {optimizer} not recognized.")

    print_optimization_end(converged, energy, final_geom, steps, time, message)

    return converged, energy, final_geom
    # final_hessian = tsff.get_hessian(final_geom)
    # geom_with_masses = []
    # for i in range(tsff.nat):
    #     element = struc1.info.atom_types[i]
    #     mass = element_to_weight(element)
    #     x, y, z = final_geom[i]
    #     geom_with_masses.append(f"{element} {mass} {x} {y} {z}")

    # write_hessian_to_orcahessfile(tsff.nat, final_hessian, geom_with_masses, "ts.hess")


def optimize_with_forcefield_from_file(
    xyz_file: str,
    ff_file: str,
    optimizer: Callable = anc_optimizer,
    calcoptions: TSCalculationOptions = None,
    trajectory_filename: str = "trajectory.xyz",
    final_geometry_filename: str = "optimized.xyz",
    opt_stdout_filename: str = "ts_optimization.out",
):
    """Wrapper for optimize_with_forcefield, so that it is applyable directly starting at an xyz.

    Args:
        xyz_file (str): file name of the starting geometry in xyz format
        ff_file (str): file name of the force field file in ffits format
        optimizer (Callable, optional): The optimizer to be used for the optimization. Defaults to anc_optimizer.
        calcoptions (TSCalculationOptions, optional): The calculation options for the optimization. Defaults to None.
        trajectory_filename (str, optional): Filename of the trajectory file. Defaults to "trajectory.xyz".
        final_geometry_filename (str, optional): Filename of the final geometry file. Defaults to "optimized.xyz".
        opt_stdout_filename (str, optional): Filename of the optimization stdout file. Defaults to "ts_optimization.out".

    Returns:
        converged (bool): Whether the optimization converged or not
        energy (float): Final energy of the optimized structure
        final_geom (np.ndarray): Final geometry of the optimized structure in the shape (nat, 3)
    """
    print_optimization_start()
    if calcoptions is None:
        logger.warning(
            "No calculation options provided for optimization. Using default values."
        )
        calcoptions = TSCalculationOptions()

    logger.info(
        f"Optimization is performed with force field from file {ff_file}, starting from structure {xyz_file}."
    )
    nat, _, xyz, atom_types = readin_xyz(xyz_file)
    info = StructuralInformation(nat, xyz, {}, atom_types)
    ff = ForceField(
        nat,
        ff_filename=ff_file,
        readff=True,
        energy_calculator=energy_ff,
        gradient_calculator=complete_gradient,
        hessian_calculator=complete_hessian,
    )

    converged, energy, final_geom = optimize_with_forcefield(
        info,
        ff,
        optimizer=optimizer,
        calcoptions=calcoptions,
        trajectory_filename=trajectory_filename,
        final_geometry_filename=final_geometry_filename,
        opt_stdout_filename=opt_stdout_filename,
    )
    return converged, energy, final_geom
