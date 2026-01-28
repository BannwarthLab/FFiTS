#!/bin/python
import time
import os
import copy
from ffits.datatype.calculation_data import CalculationData
from ffits.io.toml_parser import load_calculation_data, overwrite_from_commandline
from ffits.io.commandline_parser import parse_args
from ffits.setup.structure_preparatation import get_preliminary_information
from ffits.ts_guess.define_starting_parameters import fill_ff
from ffits.ts_guess.parameterize_ff import fit_ff_to_hessian
from ffits.ts_guess.guess import get_ts_guess
from ffits.io.print.config import print_calculation_data, print_header_setup
from ffits.io.print.header import print_program_header
from ffits.io.file_writer import write_hessian_to_orcahessfile

def run_optimizer_mode(args):
    """
    Run in optimizer mode: takes a single structure and optimizes it using the specified optimizer.
    Usage: ffits structure.xyz --opt ff
    """
    print("[INFO] Running in optimizer mode")
    # read in FF and define energy terms
    # run optimizer as with calculation of TS guess through TS FF 
    # return structure 


def run_normal_mode(args):
    """
    Normal mode: processes reactant and product structures through the TS guess pipeline.
    """
    calcdata = load_calculation_data(args["input_file"])
    overwrite_from_commandline(calcdata, args["multiplicity"], args["charge"], args["structures"])
    print_calculation_data(calcdata)

    print_header_setup()
    struc1 = get_preliminary_information(calcdata.reactant_calc, calcdata.reactant_path, 1, calcdata.system.charge, calcdata.system.multiplicity)
    struc2 = get_preliminary_information(calcdata.product_calc, calcdata.product_path, 2, calcdata.system.charge, calcdata.system.multiplicity)
    #    TODO add bo threshold in calcdata
    
    fill_ff(struc1.ff, struc1.info, repulsive_start=calcdata.reactant_calc.ff_parameter_repulsion) 
    fill_ff(struc2.ff, struc2.info, repulsive_start=calcdata.product_calc.ff_parameter_repulsion) 

    fit_ff_to_hessian(struc1,
                      maxit=calcdata.reactant_calc.ff_parameterization_maxiteration,
                      stepsize=calcdata.reactant_calc.ff_parameterization_stepsize,
                      threshold=calcdata.reactant_calc.ff_parameterization_threshold,
                      constant_repulsion=calcdata.reactant_calc.ff_parameterization_constant_repulsion)
    fit_ff_to_hessian(struc2,
                      maxit=calcdata.product_calc.ff_parameterization_maxiteration,
                      stepsize=calcdata.product_calc.ff_parameterization_stepsize,
                      threshold=calcdata.product_calc.ff_parameterization_threshold,
                      constant_repulsion=calcdata.product_calc.ff_parameterization_constant_repulsion)

    tsff, converged, energy, final_geom = get_ts_guess(struc1, struc2, calcoptions=calcdata.ts_calc)#, optimizer=calcdata.ts_calc.optimizer)


def main():
    print_program_header()
    args = parse_args()
    
    if args["opt_mode"] is not None:
        run_optimizer_mode(args)
    else:
        run_normal_mode(args)




if __name__ == "__main__":
    main()
