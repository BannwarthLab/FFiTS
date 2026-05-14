import logging
import os
import pandas as pd
import numpy as np
import sys
from collections.abc import Callable
from ffits.ts_guess.mix_ff import create_tsff
from ffits.datatype.structure_data import Structure
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.calculation_data import CalculationData
from ffits.external.molbar import anc_optimizer
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)
from ffits.io.print.details import print_ts_optimization_start
from ffits.io.file_writer import write_hessian_to_orcahessfile
from ffits.data.elements import element_to_weight
from ffits.forcefield.python_interface.optimization import optimize_with_forcefield
from ffits.utils.temp_dir_manager import TempDirManager
from typing import Optional

logger = logging.getLogger(__name__)


def get_ts_guess(
    struc1: Structure,
    struc2: Structure,
    calcdata: CalculationData = None,
    optimizer: Callable = anc_optimizer,
) -> tuple[ForceField, bool, float, np.ndarray]:
    if calcdata is None:
        logger.warning(
            "No calculation data provided for TS guess generation. Using default values."
        )
        calcdata = CalculationData()
        calcdata.ts_path.ff_filename = "tsff.csv"

    trajectory_filename: str = "trajectory.xyz"
    final_geometry_filename: str = "optimized.xyz"
    print_ts_optimization_start()
    #### ------- Create TS Force Field by mixing reactant and product FFs ------- ####
    res = create_tsff(
        ff1=struc1.ff,
        info1=struc1.info,
        ff2=struc2.ff,
        info2=struc2.info,
        fact1=calcdata.ts_calc.factor_reactant,
        fact2=calcdata.ts_calc.factor_product,
        calcdata=calcdata,
    )
    tsff: ForceField = res["tsff"]
    tsff.energy_calculator = energy_ff
    tsff.gradient_calculator = complete_gradient
    tsff.hessian_calculator = complete_hessian

    ### ---------- optimization ------------ ###
    if tsff.start_from_reactant:
        logger.info("Starting TS guess optimization from reactant geometry.")
        initial_struc = struc1.info
    else:
        logger.info("Starting TS guess optimization from product geometry.")
        initial_struc = struc2.info
    converged, energy, final_geom = optimize_with_forcefield(
        initial_struc,
        tsff,
        optimizer,
        calcdata.ts_calc,
        trajectory_filename,
        final_geometry_filename,
    )
    # TODO add time and also return it

    return tsff, converged, energy, final_geom
