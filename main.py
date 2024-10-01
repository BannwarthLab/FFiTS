#!/bin/python
from interface.readin import *
from datatype.calculation_data import *
import os
import shutil
from calculation import GeometryOptimization

def original_name(name):
    return 'original' + name

def copy_original_structure(temp_name):
    if not os.path.exists(temp_name):
        raise FileExistsError(temp_name, 'does not exist in current working directory: ', os.getcwd())
    shutil.copy(temp_name, original_name(temp_name))
    print(f'> {temp_name} was copied to {original_name(temp_name)}. {temp_name} may change in further calculations.')

def initialization(config_path):
    calc_params = readin_config(config_path)
    copy_original_structure(calc_params.struc1.xyz_filename)
    copy_original_structure(calc_params.struc2.xyz_filename)
    return calc_params

def generate_data_for_calculation(input_xyz: str, calc_params: CalculationParameters):
    GeometryOptimization.with_xtb(input_xyz=input_xyz, calc=calc_params)

# def main_alignment(calc_param: CalculationParameters): 
#     pass


# def main_ts_search(calc_param: CalculationParameters): 
#     pass


def main():
    calc_params = initialization('/home/guests/dbabushkina/1_ts_search2024/pytsguess/config.json')
    generate_data_for_calculation('struc1.xyz', calc_params)
    

    





if __name__ == "__main__":
    main()
    # main_alignment(calc_param)
    # main_ts_search(calc_param)