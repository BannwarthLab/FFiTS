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

def main():
    # read commandline arguments and get calculation data from given config file
    
    print_program_header()
    args = parse_args()
    calcdata = load_calculation_data(args["input_file"])
    overwrite_from_commandline(calcdata, args["multiplicity"], args["charge"])
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

    get_ts_guess(struc1, struc2, calcoptions=calcdata.ts_calc)#, optimizer=calcdata.ts_calc.optimizer)



if __name__ == "__main__":
    main()
