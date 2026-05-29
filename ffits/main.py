#!/bin/python
from collections.abc import Callable
import logging
import time
from ffits.datatype.structure_data import Structure
from ffits.io.logging_config import setup_logger
from ffits.datatype.calculation_data import CalculationData
from ffits.io.toml_parser import overwrite_from_commandline
from ffits.io.commandline_parser import parse_args
from ffits.ts_guess.parameterize_ff import parameterize_ff
from ffits.ts_guess.guess import get_ts_guess, get_ts_guess_from_xyz
from ffits.io.print.config import print_calculation_data, print_header_setup
from ffits.io.print.header import print_program_header
from ffits.io.print.summary import print_run_summary
from ffits.ts_guess.rct_path import create_path
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
    structure_filename: str,
    calcdata: CalculationData = CalculationData.from_default(),
):
    """
    Important: If you want to use custom stucture information by inputting it through calcdata, please be aware, that calcdata.reactant_* is used to save information about the structure.

    Args:
        structure_filename (str): filename of structure, which is used for the FF parameterization
        calcdata (CalculationData, optional): calculation data containing configuration options. Only system and reactant information is used. Defaults to CalculationData.from_default().
    """
    logger.warning(
        "In parameterization mode the dihedral values may be different then when the parameterization is performed in a full TS guess calculation as no comparison of reactant and product structures is available."
    )
    calcdata.reactant_path.xyz_filename = structure_filename
    struc1 = Structure.from_config(
        cd=calcdata,
        calcopt=calcdata.reactant_calc,
        pathdata=calcdata.reactant_path,
    )
    if not calcdata.reactant_calc.ff_parameterization:
        logger.error(
            "FF parameterization mode was selected but ff_parameterization is not set to True in the config. Please rethink your choices and either set ff_parameterization to True or do not select --parameterize mode."
        )
        sys.exit(1)

    parameterize_ff(struc1, calcdata.reactant_calc, priorities=None)


def run_reaction_path_mode(
    reactant_filename: str,
    product_filename: str,
    steps: int = 10,
    calcdata: CalculationData = CalculationData.from_default(),
):
    """
    Run in reaction path mode: takes two structures and generates a specified number of frames along the reaction path between them. Currently not implemented.
    Usage: ffits reactant.xyz product.xyz --rctpath 10
    step number include reactant and product.
    """
    calcdata.reactant_path.xyz_filename = reactant_filename
    calcdata.product_path.xyz_filename = product_filename
    trajectory = create_path(
        calcdata, steps=steps-2, minfact1=0 + 1 / steps, maxfact1=1 - 1 / steps
    )

    return trajectory


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
    calcdata.reactant_path.xyz_filename = reactant_filename
    calcdata.product_path.xyz_filename = product_filename
    tsff, converged, energy, final_geom = get_ts_guess_from_xyz(
        reactant_filename=reactant_filename,
        product_filename=product_filename,
        calcdata=calcdata,
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
        elif args["parameterize"]:
            run_parameterization_mode(args.get("structures")[0], calcdata=calcdata)
        elif args["reaction_path"]:
            run_reaction_path_mode(
                reactant_filename=args.get("structures")[0],
                product_filename=args.get("structures")[1],
                steps=args["reaction_path"],
                calcdata=calcdata,
            )
        else:
            run_tsguess_mode(
                reactant_filename=args.get("structures")[0],
                product_filename=args.get("structures")[1],
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
