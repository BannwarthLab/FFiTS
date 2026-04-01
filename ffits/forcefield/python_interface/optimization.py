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


def optimize_xyz_with_forcefield(
    xyz_file: str,
    ff_file: str,
    calcoptions: TSCalculationOptions,
    optimizer: Callable = anc_optimizer,
    temp_dir_manager: Optional[TempDirManager] = None,
):
    print_optimization_start()
    logger.info(
        f"Optimization is performed with force field from file {ff_file}, starting from structure {xyz_file}."
    )
    nat, comment, xyz, atom_types = readin_xyz(xyz_file)
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
        info, ff, optimizer, calcoptions
    )
    return converged, energy, final_geom


def optimize_with_forcefield(
    info: StructuralInformation,
    ff: ForceField,
    optimizer: Callable,
    calcoptions: TSCalculationOptions,
    trajectory_filename: str = "trajectory.xyz",
    final_geometry_filename: str = "optimized.xyz",
    opt_stdout_filename: str = "ts_optimization.out",
):

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

    elif optimizer == scipy_optimizer:  # TODO does not work yet i think
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
