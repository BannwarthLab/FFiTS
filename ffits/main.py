#!/bin/python
import logging
import time
import os
import copy
from ffits.io.logging_config import setup_logger
from ffits.utils.temp_dir_manager import TempDirManager
from ffits.datatype.calculation_data import CalculationData
from ffits.io.toml_parser import load_calculation_data, overwrite_from_commandline
from ffits.io.commandline_parser import parse_args
from ffits.setup.structure_preparatation import get_preliminary_information
from ffits.ts_guess.define_starting_parameters import fill_ff
from ffits.ts_guess.parameterize_ff import fit_ff_to_hessian
from ffits.ts_guess.guess import get_ts_guess
from ffits.io.print.config import print_calculation_data, print_header_setup
from ffits.io.print.header import print_program_header
from ffits.io.print.summary import print_run_summary
from ffits.forcefield.python_interface.optimization import optimize_xyz_with_forcefield
from ffits.io.file_writer import write_hessian_to_orcahessfile
from ffits.external.molbar import anc_optimizer, get_combinded_priorities

logger = logging.getLogger(__name__)

def run_optimizer_mode(args, temp_dir_manager):
    """
    Run in optimizer mode: takes a single structure and optimizes it using the specified optimizer.
    Usage: ffits structure.xyz --opt ff
    """
    logger.info("Running in optimizer mode")

    calcdata: CalculationData = load_calculation_data(args["config_file"])
    overwrite_from_commandline(calcdata, args["multiplicity"], args["charge"], args["structures"])
    # print_calculation_data(calcdata)

    optimize_xyz_with_forcefield(args['structures'][0], args['optff'], calcdata.ts_calc, anc_optimizer, temp_dir_manager)
    # read in FF and define energy terms
    # run optimizer as with calculation of TS guess through TS FF 
    # return structure 


def run_tsguess_mode(args, temp_dir_manager):
    """
    Normal mode: processes reactant and product structures through the TS guess pipeline.
    """
    calcdata = load_calculation_data(args["config_file"])
    overwrite_from_commandline(calcdata, args["multiplicity"], args["charge"], args["structures"])
    print_calculation_data(calcdata)

    print_header_setup()
    struc1 = get_preliminary_information(calcdata.reactant_calc, calcdata.reactant_path, 1, calcdata.system, temp_dir_manager)
    struc2 = get_preliminary_information(calcdata.product_calc, calcdata.product_path, 2, calcdata.system, temp_dir_manager)
    #    TODO add bo threshold in calcdata
    
    # Compute combined priorities for dihedral classification
    priorities = get_combinded_priorities(struc1.info, struc2.info)
    
    if calcdata.reactant_calc.ff_parameterization:
        fill_ff(struc1.ff, struc1.info, repulsive_start=calcdata.reactant_calc.ff_parameter_repulsion, priorities=priorities) 
        fit_ff_to_hessian(struc1,
                        maxit=calcdata.reactant_calc.ff_parameterization_maxiteration,
                        stepsize=calcdata.reactant_calc.ff_parameterization_stepsize,
                        threshold=calcdata.reactant_calc.ff_parameterization_threshold,
                        constant_repulsion=calcdata.reactant_calc.ff_parameterization_constant_repulsion)
        
    if calcdata.product_calc.ff_parameterization:
        fill_ff(struc2.ff, struc2.info, repulsive_start=calcdata.product_calc.ff_parameter_repulsion, priorities=priorities) 
        fit_ff_to_hessian(struc2,
                        maxit=calcdata.product_calc.ff_parameterization_maxiteration,
                        stepsize=calcdata.product_calc.ff_parameterization_stepsize,
                        threshold=calcdata.product_calc.ff_parameterization_threshold,
                        constant_repulsion=calcdata.product_calc.ff_parameterization_constant_repulsion)

    tsff, converged, energy, final_geom = get_ts_guess(struc1, struc2, calcdata=calcdata, temp_dir_manager=temp_dir_manager)
    


def main():
    start_time = time.time()
    
    try:
        args = parse_args()
        
        # Initialize logging - DEBUG level if --debug flag, otherwise INFO
        log_level = "DEBUG" if args.get("debug") else "INFO"
        setup_logger(log_level)
        
        # Initialize temporary directory manager
        temp_dir_manager = TempDirManager()
        
        print_program_header()
        
        if args["optff"] is not None:
            run_optimizer_mode(args, temp_dir_manager)
        else:
            run_tsguess_mode(args, temp_dir_manager)
        
        end_time = time.time()
        print_run_summary(start_time, end_time, success=True)
        
    except Exception as e:
        end_time = time.time()
        print_run_summary(start_time, end_time, success=False, message=str(e))
        raise




if __name__ == "__main__":
    main()
