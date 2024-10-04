#!/bin/python
from interface.readin import *
from datatype.calculation_data import *
from datatype.structure_data import *
import os
import shutil
from calculation import *
from forcefield import increase_atompair_relevance
from align import get_ffatom_pair
import copy

def original_name(name):
    return 'original' + name

def copy_original_structure(temp_name):
    shutil.copy(temp_name, original_name(temp_name))
    print(f'> {temp_name} was copied to {original_name(temp_name)}. {temp_name} may change in further calculations.')

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
    nat, energy, xyz = GeometryOptimization.with_xtb(input_structure)
    HessianCalculation.with_xtb(input_structure)
    wbo = WboCalculation.with_xtb(input_structure)
    return StructuralInformation(nat, energy, xyz, wbo)
    

def generate_ff(struc_path: StructurePath, struc_info: StructuralInformation):
    return FittedFfGeneration.with_crest(struc_info.nat, struc_path.xyz_filename, struc_path)

def modify_ff(struc: Structure, reac: Reaction):
    forbidden_pairs = [[]]
    atompairs_left = True
    while atompairs_left:
        at1, at2 = get_ffatom_pair(reac, struc, forbidden_pairs)
        print(at1, at2)  
        forbidden_pairs.append([at1, at2])
        print(forbidden_pairs)
        if at1 == 0 and at2 == 0:
            atompairs_left = False
            break 
        increase_atompair_relevance(struc.ff, at1, at2)
    np.multiply(struc.ff.c_bond, 2)
    np.multiply(struc.ff.c_angle, 2)
    np.multiply(struc.ff.c_dihedral, 2)
    np.multiply(struc.ff.c_lj, 0.5)
    os.remove('crestopt.xyz')

    struc.ff.write_force_field(struc.path.ff_filename)
    print(f'> FF for alignment written to {struc.path.ff_filename}')
    return struc.ff




def align(struc1: Structure, struc2: Structure, reac: Reaction, calc: CalculationParameters):
    if not calc.perform_alignment:
        return None
    struc1.ff = modify_ff(struc1, reac)
    GeometryOptimization.with_ff_potential_crest(struc1.path, struc1.ff)

    struc2.ff = modify_ff(struc2, reac)
    GeometryOptimization.with_ff_potential_crest(struc2.path, struc2.ff)

    return struc1, struc2

# def main_alignment(calc_param: CalculationParameters): 
#     pass


# def main_ts_search(calc_param: CalculationParameters): 
#     pass


def main():
    # TODO ich muss noch fälle festlegen wenn man Sachen nicht berechnen will, dass sie dann immer noch eingelesen werden .
    calc_params = initialization('/home/guests/dbabushkina/1_ts_search2024/pytsguess/config.json')

    struc1_info = generate_data_for_calculation(calc_params.struc1, calc_params)
    struc1_ff = generate_ff(calc_params.struc1, struc1_info)
    struc1 = Structure(calc_params.struc1, struc1_ff, struc1_info)

    struc2_info = generate_data_for_calculation(calc_params.struc2, calc_params)
    struc2_ff = generate_ff(calc_params.struc2, struc2_info)
    struc2 = Structure(calc_params.struc2, struc2_ff, struc2_info)

    reac = Reaction.with_unique_bonds(struc1, struc2)
    print(reac)
    
    new_struc1, new_struc2 = align(struc1, struc2, reac, calc_params)

    # TODO als nächstes: Atom Pair relevance erhöhen und neue Optimierung machen 
    

    





if __name__ == "__main__":
    main()
    # main_alignment(calc_param)
    # main_ts_search(calc_param)