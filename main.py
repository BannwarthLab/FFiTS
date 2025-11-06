#!/bin/python
import time
import os
import copy
from src.datatype.calculation_data import CalculationData
from src.io.toml_parser import load_calculation_data, overwrite_from_commandline
from src.io.commandline_parser import parse_args
from src.setup.structure_preparatation import get_preliminary_information
from src.ts_guess.define_starting_parameters import fill_ff
from src.ts_guess.parameterize_ff import fit_ff_to_hessian
from src.ts_guess.guess import get_ts_guess
from src.io.print.config import print_calculation_data, print_setup
from src.io.print.header import print_program_header

def main():
    # read commandline arguments and get calculation data from given config file
    # a normal call would python main.py struc1.xyz struc2.xyz -i input.toml -m 3 -c 1
    print_program_header()
    args = parse_args()
    calcdata = load_calculation_data(args["input_file"])
    overwrite_from_commandline(calcdata, args["multiplicity"], args["charge"])
    print_calculation_data(calcdata)


    print_setup()
    struc1 = get_preliminary_information(calcdata.reactant_calc, calcdata.reactant_path, 1, calcdata.system.charge, calcdata.system.multiplicity)
    struc2 = get_preliminary_information(calcdata.product_calc, calcdata.product_path, 2, calcdata.system.charge, calcdata.system.multiplicity)
    #    TODO add repulsive start and bo threshold in calcdata, and all the parameters in ff fit 
    
    fill_ff(struc1.ff, struc1.info) 
    fill_ff(struc2.ff, struc2.info) 

    fit_ff_to_hessian(struc1)
    fit_ff_to_hessian(struc2)

    get_ts_guess(struc1, struc2)
    # cwd = os.getcwd() 
    
    # calcdata = load_calculation_data()


# def main(calc_params):
#     # TODO ich muss noch fälle festlegen wenn man Sachen nicht berechnen will, dass sie dann immer noch eingelesen werden .
#     # calc_params = initialization('/home/dbabushkina/1_ts_search2024/pytsguess/config.json')

#     struc1_info = generate_data_for_calculation(calc_params.struc1, calc_params)
#     struc1_ff = generate_ff(calc_params.struc1, struc1_info)
#     struc1 = Structure(calc_params.struc1, struc1_ff, struc1_info)

#     struc2_info = generate_data_for_calculation(calc_params.struc2, calc_params)
#     struc2_ff = generate_ff(calc_params.struc2, struc2_info)
#     struc2 = Structure(calc_params.struc2, struc2_ff, struc2_info)

#     reac = Reaction(struc1, struc2)
    
#     #struc1, struc2 = align(struc1, struc2, reac, calc_params)
    
#     tic = time.time()
#     ts_ff = search_ts(struc1, struc2, calc_params)
#     tac = time.time()
#     print(f'TS Guess calculation took {tac - tic}')

#     constrained_optimization(reac, calc_params)

    

# if __name__ == "__main__":
#     calc_params = initialization('/home/dbabushkina/1_ts_search2024/pytsguess/config.json')
#     try:
#         main(calc_params)
#     except Exception as error: 
#         shutil.copy(Name.original_xyz(calc_params.struc1.xyz_filename), calc_params.struc1.xyz_filename)
#         shutil.copy(Name.original_xyz(calc_params.struc2.xyz_filename), calc_params.struc2.xyz_filename)
#         print(error)
if __name__ == "__main__":
    main()
