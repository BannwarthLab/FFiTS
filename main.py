#!/bin/python
import time
import shutil
import copy
from src.datatype.structure_data import *
from src.datatype.calculation_data import *
from src.initialization import *
from src.calculation import *
from src.alignment import align
from src.interface.reader import *
from src.interface.xtb import *
from src.tssearch import search_ts, constrained_optimization


def main(calc_params):
    # TODO ich muss noch fälle festlegen wenn man Sachen nicht berechnen will, dass sie dann immer noch eingelesen werden .
    # calc_params = initialization('/home/dbabushkina/1_ts_search2024/pytsguess/config.json')

    struc1_info = generate_data_for_calculation(calc_params.struc1, calc_params)
    struc1_ff = generate_ff(calc_params.struc1, struc1_info)
    struc1 = Structure(calc_params.struc1, struc1_ff, struc1_info)

    struc2_info = generate_data_for_calculation(calc_params.struc2, calc_params)
    struc2_ff = generate_ff(calc_params.struc2, struc2_info)
    struc2 = Structure(calc_params.struc2, struc2_ff, struc2_info)

    reac = Reaction.with_unique_bonds(struc1, struc2)
    
    struc1, struc2 = align(struc1, struc2, reac, calc_params)
    
    tic = time.time()
    ts_ff = search_ts(struc1, struc2, calc_params)
    tac = time.time()
    print(f'TS Guess calculation took {tac - tic}')

    constrained_optimization(reac, calc_params)

    

if __name__ == "__main__":
    calc_params = initialization('/home/dbabushkina/1_ts_search2024/pytsguess/config.json')
    try:
        main(calc_params)
    except Exception as error: 
        shutil.copy(Name.original_xyz(calc_params.struc1.xyz_filename), calc_params.struc1.xyz_filename)
        shutil.copy(Name.original_xyz(calc_params.struc2.xyz_filename), calc_params.struc2.xyz_filename)
        print(error)
        