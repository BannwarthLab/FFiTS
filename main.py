#!/bin/python
from interface.readin import *
from datatype.calculation_data import *
from datatype.structure_data import *
import os
import shutil
from calculation import *

def original_name(name):
    return 'original' + name

def copy_original_structure(temp_name):
    if not os.path.exists(temp_name):
        raise FileExistsError(temp_name, 'does not exist in current working directory: ', os.getcwd())
    shutil.copy(temp_name, original_name(temp_name))
    print(f'> {temp_name} was copied to {original_name(temp_name)}. {temp_name} may change in further calculations.')

def initialization(config_path):
    # TODO check wether files are present or perform calc is set to true. if both no, then we need an exception
    calc_params = readin_config(config_path)
    copy_original_structure(calc_params.struc1.xyz_filename)
    copy_original_structure(calc_params.struc2.xyz_filename)
    return calc_params

def generate_data_for_calculation(input_structure: StructurePath, calc_params: CalculationParameters):
    energy = GeometryOptimization.with_xtb(input_structure, calc_params)
    HessianCalculation.with_xtb(input_structure, calc_params)
    wbo_dict = WboCalculation.with_xtb(input_structure, calc_params)
    print(wbo_dict)

# def main_alignment(calc_param: CalculationParameters): 
#     pass


# def main_ts_search(calc_param: CalculationParameters): 
#     pass


def main():
    calc_params = initialization('/home/guests/dbabushkina/1_ts_search2024/pytsguess/config.json')
    generate_data_for_calculation(calc_params.struc1, calc_params)
    

    





if __name__ == "__main__":
    main()
    # main_alignment(calc_param)
    # main_ts_search(calc_param)