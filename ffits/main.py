#!/bin/python
from collections.abc import Callable
import logging
import time
from ffits.datatype.structure_data import Structure
from ffits.io.logging_config import setup_logger
from ffits.utils.temp_dir_manager import TempDirManager
from ffits.datatype.calculation_data import CalculationData
from ffits.io.toml_parser import overwrite_from_commandline
from ffits.io.commandline_parser import parse_args
from ffits.ts_guess.parameterize_ff import parameterize_ff
from ffits.ts_guess.guess import get_ts_guess
from ffits.io.print.config import print_calculation_data, print_header_setup
from ffits.io.print.header import print_program_header
from ffits.io.print.summary import print_run_summary
from ffits.forcefield.python_interface.optimization import (
    optimize_with_forcefield_from_file,
)
from ffits.external.molbar import anc_optimizer, get_combinded_priorities
import sys

logger = logging.getLogger(__name__)


def run_optimizer_mode(
    structure_filename: str,
    ff_filename: str,
    optimizer: Callable = anc_optimizer,
    calcdata: CalculationData = CalculationData.from_default(),
):
    """
    Run in optimizer mode: takes a single structure and optimizes it using the specified optimizer.
    Usage: ffits structure.xyz --opt ff
    """
    converged, energy, final_geom = optimize_with_forcefield_from_file(
        structure_filename,
        ff_filename,
        optimizer=optimizer,
        calcoptions=calcdata.ts_calc,
    )

    return converged, energy, final_geom
    # read in FF and define energy terms
    # run optimizer as with calculation of TS guess through TS FF
    # return structure


def run_parameterization_mode(
    calcdata: CalculationData = CalculationData.from_default(),
):
    pass


def run_tsguess_mode(
    reactant_filename: str,
    product_filename: str,
    calcdata: CalculationData = CalculationData.from_default(),
):
    """
    Normal mode: processes reactant (calcdata.reactant_path.xyz_filename) and product coordinate files (calcdata.product_path.xyz_filename) to generate a TS guess.

    Args:
        calcdata (CalculationData, optional): Calculation data containing paths to reactant and product structures, as well as calculation options. Defaults to CalculationData.from_default().
    """
    # should be the same as in calcdata, but ensures that it definitely is the same.
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

    # Compute combined priorities for dihedral classification
    priorities = get_combinded_priorities(struc1.info, struc2.info)

    if calcdata.reactant_calc.ff_parameterization:
        parameterize_ff(struc1, calcdata.reactant_calc, priorities)

    if calcdata.product_calc.ff_parameterization:
        parameterize_ff(struc2, calcdata.product_calc, priorities)

    tsff, converged, energy, final_geom = get_ts_guess(
        struc1, struc2, calcdata=calcdata
    )

    return tsff, converged, energy, final_geom


def main():
    print_program_header()
    start_time = time.time()

    args = parse_args()

    # Initialize logging - DEBUG level if --debug flag, otherwise INFO
    log_level = "DEBUG" if args.get("debug") else "INFO"
    setup_logger(log_level)
    calcdata = (
        CalculationData.from_config(args.get("config_file"))
        if args.get("config_file")
        else CalculationData.from_default()
    )
    print(calcdata)
    overwrite_from_commandline(
        calcdata,
        multiplicity=args["multiplicity"],
        charge=args["charge"],
        structures=args["structures"],
    )

    try:
        if args["optff"] is not None:
            run_optimizer_mode(
                args.get("structures")[0], args.get("optff"), calcdata=calcdata
            )
        else:
            run_tsguess_mode(
                reactant_filename=calcdata.reactant_path.xyz_filename,
                product_filename=calcdata.product_path.xyz_filename,
                calcdata=calcdata,
            )

        end_time = time.time()
        print_run_summary(start_time, end_time, success=True)

    except Exception as e:
        end_time = time.time()
        print_run_summary(start_time, end_time, success=False, message=str(e))
        raise e


if __name__ == "__main__":
    main()
