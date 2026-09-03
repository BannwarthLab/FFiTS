"""TS guess generation: builds a TS force field from reactant/product FFs and optimizes on it."""
import logging
import numpy as np
from collections.abc import Callable
from ffits.io.print.config import print_calculation_data, print_header_setup
from ffits.ts_guess.mix_ff import create_tsff
from ffits.datatype.structure_data import Structure
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.calculation_data import CalculationData
from ffits.external.molbar import anc_optimizer, get_combinded_priorities
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)
from ffits.io.print.details import print_ts_optimization_start
from ffits.forcefield.python_interface.optimization import optimize_with_forcefield

from ffits.ts_guess.parameterize_ff import parameterize_ff

logger = logging.getLogger(__name__)


def get_ts_guess(
    struc1: Structure,
    struc2: Structure,
    calcdata: CalculationData | None = None,
    optimizer: Callable = anc_optimizer,
) -> tuple[ForceField, bool, float, np.ndarray]:
    """Generates a TS guess from reactant and product structures by first constructing the TS FF and the using it as the potential for the geometry optimization or either reactant or product structure.

    Args:
        struc1 (Structure): Structure object containing information about the reactant structure
        struc2 (Structure): Structure object containing information about the product structure
        calcdata (CalculationData, optional): Calculation data for the TS guess generation. Defaults to a fresh ``CalculationData.from_default()``. This should contain the default options for the TS FF creation and optimization.
        optimizer (Callable, optional): Optimization function (eg optimizer algorithm) to use. Defaults to anc_optimizer.

    Returns:
        tuple[ForceField, bool, float, np.ndarray]: TS ForceField object, convergence status of the optimization, final energy of the optimized structure, and final geometry of the optimized structure in the shape (nat, 3)
    """
    if calcdata is None:
        calcdata = CalculationData.from_default()

    trajectory_filename: str = "trj_ts_guess.xyz"
    final_geometry_filename: str = "ts_guess.xyz"
    print_ts_optimization_start()

    #### ------- Create TS Force Field by mixing reactant and product FFs ------- ####
    # factor reactant and product will be overwritten if weight bonds with hessian is set to True
    res = create_tsff(
        ff1=struc1.ff,
        info1=struc1.info,
        ff2=struc2.ff,
        info2=struc2.info,
        calcdata=calcdata,
    )
    tsff: ForceField = res["tsff"]
    tsff.energy_calculator = energy_ff
    tsff.gradient_calculator = complete_gradient
    tsff.hessian_calculator = complete_hessian

    ### ---------- optimization ------------ ###
    if tsff.start_from_reactant:
        logger.info(
            f"Starting TS guess optimization from reactant geometry {struc1.path.xyz_filename}."
        )
        initial_struc = struc1.info
    else:
        logger.info(
            f"Starting TS guess optimization from product geometry {struc2.path.xyz_filename}."
        )
        initial_struc = struc2.info

    converged, energy, final_geom = optimize_with_forcefield(
        initial_struc,
        tsff,
        optimizer,
        calcdata.ts_calc,
        trajectory_filename,
        final_geometry_filename,
        opt_stdout_filename="ts_optimization.out"
    )
    # TODO add time and also return it

    return tsff, converged, energy, final_geom


def get_ts_guess_from_xyz(
    reactant_filename: str,
    product_filename: str,
    calcdata: CalculationData | None = None,
    optimizer: Callable = anc_optimizer,
) -> tuple[ForceField, bool, float, np.ndarray]:
    """Wrapper for get_ts_guess, so that it is applyable directly starting at xyz files.

    Args:
        reactant_filename (str): file name of the reactant structure in xyz
        product_filename (str): file name of the product structure in xyz
        calcdata (CalculationData, optional): Calculation data for the TS guess generation. Defaults to a fresh ``CalculationData.from_default()``. This should contain the default options for the TS FF creation and optimization.
        optimizer (Callable, optional): Optimization function (eg optimizer algorithm) to use. Defaults to anc_optimizer.

    Returns:
        tuple[ForceField, bool, float, np.ndarray]: TS ForceField object, convergence status of the optimization, final energy of the optimized structure, and final geometry of the optimized structure in the shape (nat, 3)
    """
    if calcdata is None:
        calcdata = CalculationData.from_default()
    calcdata.reactant_path.xyz_filename = reactant_filename
    calcdata.product_path.xyz_filename = product_filename

    print_calculation_data(calcdata)

    print_header_setup()

    struc1 = Structure.from_config(
        cd=calcdata,
        calcopt=calcdata.reactant_calc,
        pathdata=calcdata.reactant_path,
    )
    struc2 = Structure.from_config(
        cd=calcdata,
        calcopt=calcdata.product_calc,
        pathdata=calcdata.product_path,
    )

    priorities = None
    if (
        not calcdata.reactant_calc.only_proper_dihedrals
        or not calcdata.product_calc.only_proper_dihedrals
    ):
        logger.info(
            "The option only_proper_dihedrals is set to False for either the reactant or product structure. This means that improper dihedrals will be included in the TS FF creation and optimization."
        )
        priorities = get_combinded_priorities(struc1.info, struc2.info)

    if calcdata.reactant_calc.ff_parameterization:
        parameterize_ff(struc1, calcdata.reactant_calc, priorities)

    if calcdata.product_calc.ff_parameterization:
        parameterize_ff(struc2, calcdata.product_calc, priorities)

    tsff, converged, energy, final_geom = get_ts_guess(
        struc1, struc2, calcdata=calcdata
    )

    return tsff, converged, energy, final_geom
