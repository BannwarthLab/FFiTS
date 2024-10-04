#!/bin/python

import shutil
import os
from src.datatype.structure_data import StructurePath, Name, StructuralInformation
from src.datatype.calculation_data import CalculationParameters
from src.interface.reader import readin_config
from src.calculation import GeometryOptimization, WboCalculation, HessianCalculation, FittedFfGeneration

def copy_original_structure(temp_name):
    shutil.copy(temp_name, Name.original_xyz(temp_name))
    print(f'> {temp_name} was copied to {Name.original_xyz(temp_name)}. {temp_name} may change in further calculations.')

def initial_check(calc_params: CalculationParameters, struc: StructurePath):
    print(f'> Calculations will be performed in {os.getcwd()}')
    if not os.path.exists(struc.xyz_filename):
        raise FileExistsError(f'Please provide {struc.xyz_filename} to perform calculations.') 
    if not calc_params.perform_hesscalculation and not os.path.exists(struc.hess_filename): 
        raise FileExistsError(f'No Hessian file present. Please provide a hessian file for {struc.xyz_filename} or set perform_hesscalculation to True.')
    if not calc_params.perform_wbocalculation and not os.path.exists(struc.wbo_filename): 
        raise FileExistsError(f'No WBO file present. Please provide a wbo file for {struc.xyz_filename} or set perform_wbocalculation to True.')


def initialization(config_path):
    # TODO check wether files are present or perform calc is set to true. if both no, then we need an exception
    calc_params = readin_config(config_path)
    initial_check(calc_params, calc_params.struc1)
    initial_check(calc_params, calc_params.struc2)
    copy_original_structure(calc_params.struc1.xyz_filename)
    copy_original_structure(calc_params.struc2.xyz_filename)
    return calc_params



def generate_data_for_calculation(input_structure: StructurePath, calc_params: CalculationParameters):
    if calc_params.perform_geometryoptimization:
        nat, energy, xyz = GeometryOptimization.with_xtb(input_structure)
    else:
        nat, energy, xyz = readin_xyz(input_structure.xyz_filename)
    if calc_params.perform_hesscalculation:
        HessianCalculation.with_xtb(input_structure)
    if calc_params.perform_wbocalculation:
        wbo = WboCalculation.with_xtb(input_structure)
    else:   
        wbo = read_wbo_file(input_structure.wbo_filename)
    return StructuralInformation(nat, energy, xyz, wbo)
    

def generate_ff(struc_path: StructurePath, struc_info: StructuralInformation):
    return FittedFfGeneration.with_crest(struc_info.nat, struc_path.xyz_filename, struc_path)