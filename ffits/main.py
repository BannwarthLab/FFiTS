#!/bin/python
"""Entry point and top-level run modes for the ffits command-line program."""

from collections.abc import Callable
import logging
import time
from time import perf_counter
from ffits.datatype.structure_data import Structure
from ffits.io.logging_config import setup_logger, _write_timing_report
from ffits.datatype.calculation_data import CalculationData
from ffits.io.toml_parser import overwrite_from_commandline
from ffits.io.commandline_parser import parse_args
from ffits.ts_guess.parameterize_ff import parameterize_ff
from ffits.ts_guess.guess import get_ts_guess_from_xyz
from ffits.io.print.header import print_program_header
from ffits.io.print.summary import print_run_summary
from ffits.ts_guess.rct_path import create_path
from ffits.forcefield.python_interface.optimization import (
    optimize_with_forcefield_from_file,
)
from ffits.external.molbar import anc_optimizer
import sys

logger = logging.getLogger(__name__)


def run_optimizer_mode(
    structure_filename: str,
    ff_filename: str,
    optimizer: Callable = anc_optimizer,
    calcdata: CalculationData | None = None,
):
    """Run in optimizer mode: optimize a single structure with a force field.

    Usage: ffits structure.xyz --opt ff

    Args:
        structure_filename (str): Structure file (xyz) to optimize.
        ff_filename (str): Force field file used for the optimization.
        optimizer (Callable, optional): Optimizer function to use.
        calcdata (CalculationData, optional): Calculation options to use.
            Defaults to a fresh ``CalculationData.from_default()``.

    Returns:
        tuple: ``(converged, energy, final_geom)``.
    """
    if calcdata is None:
        calcdata = CalculationData.from_default()
    converged, energy, final_geom = optimize_with_forcefield_from_file(
        structure_filename,
        ff_filename,
        optimizer=optimizer,
        calcoptions=calcdata.ts_calc,
    )

    return converged, energy, final_geom


def run_parameterization_mode(
    structure_filename: str,
    calcdata: CalculationData | None = None,
):
    """
    Important: If you want to use custom stucture information by inputting it through calcdata, please be aware, that calcdata.reactant_* is used to save information about the structure.

    Args:
        structure_filename (str): filename of structure, which is used for the FF parameterization
        calcdata (CalculationData, optional): calculation data containing configuration options. Only system and reactant information is used. Defaults to a fresh ``CalculationData.from_default()``.
    """
    if calcdata is None:
        calcdata = CalculationData.from_default()
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
    calcdata: CalculationData | None = None,
):
    """Run in reaction path mode: interpolate frames between two structures.

    Usage: ffits reactant.xyz product.xyz --rctpath 10

    Args:
        reactant_filename (str): Reactant structure file (xyz).
        product_filename (str): Product structure file (xyz).
        steps (int, optional): Total number of frames, including reactant
            and product. Defaults to 10.
        calcdata (CalculationData, optional): Calculation options to use.
            Defaults to a fresh ``CalculationData.from_default()``.

    Returns:
        The generated reaction-path trajectory.
    """
    if calcdata is None:
        calcdata = CalculationData.from_default()
    calcdata.reactant_path.xyz_filename = reactant_filename
    calcdata.product_path.xyz_filename = product_filename
    trajectory = create_path(
        calcdata, steps=steps - 2, minfact1=0 + 1 / steps, maxfact1=1 - 1 / steps
    )

    return trajectory


def run_tsguess_mode(
    reactant_filename: str,
    product_filename: str,
    calcdata: CalculationData | None = None,
    timing_logger: logging.Logger | None = None,
    timing_entries: list[str] | None = None,
):
    """Run in default (TS guess) mode: generate a transition-state guess.

    Args:
        reactant_filename (str): Reactant structure file (xyz).
        product_filename (str): Product structure file (xyz).
        calcdata (CalculationData, optional): Calculation options to use.
            Defaults to a fresh ``CalculationData.from_default()``.

    Returns:
        tuple: ``(tsff, converged, energy, final_geom)``.
    """
    if calcdata is None:
        calcdata = CalculationData.from_default()
    calcdata.reactant_path.xyz_filename = reactant_filename
    calcdata.product_path.xyz_filename = product_filename
    tsff, converged, energy, final_geom = get_ts_guess_from_xyz(
        reactant_filename=reactant_filename,
        product_filename=product_filename,
        calcdata=calcdata,
        timing_logger=timing_logger,
        timing_entries=timing_entries,
    )
    return tsff, converged, energy, final_geom


def main():
    """Run the ffits command-line program: parse args, dispatch to the
    selected run mode, and print a run summary."""
    print_program_header()
    start_time = time.time()

    args = parse_args()
    is_tsguess_mode = (
        args["optff"] is None and not args["parameterize"] and not args["reaction_path"]
    )
    timing_enabled = args.get("timing", False) and is_tsguess_mode
    timing_entries: list[str] = []
    timing_mode_start = None

    # Initialize logging - TIMING level if --timing flag, DEBUG if --debug, otherwise INFO
    log_level = "TIMING" if timing_enabled else "DEBUG" if args.get("debug") else "INFO"
    timing_logger = setup_logger(log_level)

    if args.get("timing", False) and not is_tsguess_mode:
        timing_logger.warning("--timing is currently supported only for TS guess mode.")

    success = False
    try:
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
            if timing_enabled:
                timing_mode_start = perf_counter()
            run_tsguess_mode(
                reactant_filename=args.get("structures")[0],
                product_filename=args.get("structures")[1],
                calcdata=calcdata,
                timing_logger=timing_logger if timing_enabled else None,
                timing_entries=timing_entries if timing_enabled else None,
            )
        success = True

        end_time = time.time()
        print_run_summary(start_time, end_time, success=True)

    except Exception as e:
        end_time = time.time()
        print_run_summary(start_time, end_time, success=False, message=str(e))
        raise e
    finally:
        if timing_enabled and timing_mode_start is not None:
            _write_timing_report(
                timing_entries=timing_entries,
                total_elapsed=perf_counter() - timing_mode_start,
                success=success,
            )


if __name__ == "__main__":
    main()
