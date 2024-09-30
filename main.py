#!/bin/python

from dataclasses import dataclass
import copy
import re
import shutil
import subprocess
import time
from time import sleep
import numpy as np
import pandas as pd
import os
import networkx as nx
from wbo_processort import Structure
from wbo_processort import Reaction
from call_xtb import Xtb
from input_library import input_ff_calc, input_ff_opt1, input_ff_opt2
from forcefield import Hopot
from call_crest import Crest
import time

import sys
import threading

# @dataclass
# class CalculationParameters:
#     xyz1: str 
#     hess1: str 
#     wbo1: str 
#     ff1: str
#     xyz2: str 
#     hess2: str 
#     wbo2: str 
#     ff2: str

def get_atom_pair(reaction: Reaction, struc: Structure):
    at1 = 0
    at2 = 0
    for val in reaction.unique_bonds:
        G_temp = copy.deepcopy(struc.G)
        G_temp.add_edge(val[0], val[1])
        if len([G_temp.subgraph(c).copy() for c in nx.connected_components(G_temp)]) == 1:
            print(val, 'is a connecting bond.')
            if struc.sep_molecules[0].has_node(val[0]):
                at1 = nx.get_node_attributes(struc.sep_molecules[0], 'id_in_subgraph')[val[0]] - 1
                at2 = nx.get_node_attributes(struc.sep_molecules[1], 'id_in_subgraph')[val[1]] - 1
            else:
                at2 = nx.get_node_attributes(struc.sep_molecules[1], 'id_in_subgraph')[val[0]] - 1
                at1 = nx.get_node_attributes(struc.sep_molecules[0], 'id_in_subgraph')[val[1]] - 1
            return at1, at2
        print(val, 'is not a connecting bond.')
    return None
    # raise Exception('dfsaf')

def get_ffatom_pair(reaction: Reaction, struc: Structure, forbidden_pair=[[-1,-1]]):
    at1 = 0
    at2 = 0
    for val in reaction.unique_bonds:
        G_temp = copy.deepcopy(struc.G)
        G_temp.add_edge(val[0], val[1])
        if len([G_temp.subgraph(c).copy() for c in nx.connected_components(G_temp)]) == 1:
            print(val, 'is a connecting bond.')
            if struc.sep_molecules[0].has_node(val[0]):
                at1 = val[0]
                at2 = val[1]
            else:
                at2 = val[0]
                at1 = val[1]
            pair = [at1, at2]
            if pair in forbidden_pair or reversed(pair) in forbidden_pair:
                continue
            return at1, at2
        print(val, 'is not a connecting bond.')
    return 0, 0

def create_infrastructure(wd: str, xyz: str) -> None:
    if not os.path.isdir(wd):
        print("Directory", wd, "is created.")
        os.makedirs(wd)
    else:
        print("Directory", wd, "is already present, will not be created.")

    shutil.copyfile(xyz, wd + '/original.xyz')
    # shutil.copyfile(path, xyz)
    print("File", xyz, "copied to", wd+"/", "as original.xyz.")

def first_ts_search(path_to_crest, nr_atoms):
    crestcaller = Crest()
    rc = 100

    rc1 = crestcaller.run_ts_search(startstruc=1,path_to_crest=path_to_crest,input_filename="input1.toml")
    os.rename('crestopt.log', 'crestopt1.log')
    print(rc1)
    rc2 = crestcaller.run_ts_search(startstruc=2,path_to_crest=path_to_crest,input_filename="input2.toml")
    os.rename('crestopt.log', 'crestopt2.log')
    print(rc2)
    if rc1 == 0 and rc2 == 0:
        print('Both TS searches worked.')
        print(os.path.exists('crestopt.xyz'))
        os.rename('crestopt.xyz', 'start1.xyz')
        return 0
    elif (rc1 == 0 and rc2 != 0) or (rc1 != 0 and rc2 == 0):
        print('One of the TS searches did not work with rc', rc1, rc2)
        return 1
    elif rc1 != 0 and rc2 != 0:
        p = subprocess.Popen('tail crestopt1.log -n '+ str(nr_atoms+2) + ' > start1.xyz', shell=True)
        p.wait()
        print('both processes did not work, last str of crestopt.log is used.')
        # p = subprocess.Popen('tail crestopt.xyz -n '+ str(nr_atoms)) + ' > final.xyz'
        # raise Exception('TS was not generated.')

    return rc

def print_box(name: str, width=80):
    print('┏' +    "━"*width       + "┓")
    print('┃' + name.center(width) + '┃')
    print('┗' +    "━"*width       + "┛")



def main_align(struc_id: str, xyz: str, wbo: str, hess: str):
    wbo1 = 'wbo1'
    wbo2 = 'wbo2'
    xyz1 = 'struc1.xyz'
    xyz2 = 'struc2.xyz'
    cwd = os.getcwd()
    print()
    print("We are in directory", os.getcwd())
    print_box(struc_id + ' Alignment')

    
    #-#-# INFRASTRUCTURE #-#-#-#-#
    wd = struc_id
    create_infrastructure(wd, xyz)

    struc = Structure(xyz, wbo)
    hopot = Hopot(struc.nat)
    reaction = Reaction(Structure(xyz1, wbo1), Structure(xyz2, wbo2))

    if struc.amount_of_molecules != 2:
        raise ValueError(xyz, 'does not consist of two molecules.')
    
    # struc.write_all_molecules_to_file(struc_id + '/coord')

    #-#-# MY ALIGNMENT #-#-#
    xtbcaller.hesscalc(hess,xyz)
    input_filename = 'ff_fit_input.toml'
    crestcaller.write_input2file(input_ff_calc, input_filename)
    crestcaller.calc_fitted_ff(input_filename)

    ff_name = 'force_field_' + struc_id
    print(ff_name)
    hopot.readin_forcefield(ff_name)
    
    forbidden_pairs = [[]]
    atompairs_left = True
    while atompairs_left:
        at1, at2 = get_ffatom_pair(reaction, struc, forbidden_pairs)
        print(at1, at2)  
        forbidden_pairs.append([at1, at2])
        print(forbidden_pairs)
        if at1 == 0 and at2 == 0:
            atompairs_left = False
            break 
        hopot.increase_atompair_relevance(at1, at2)
    np.multiply(hopot.c_bond, 2)
    np.multiply(hopot.c_angle, 2)
    np.multiply(hopot.c_dihedral, 2)
    np.multiply(hopot.c_lj, 0.5)
    os.remove('crestopt.xyz')

    if struc_id == 'struc1':
        ff_name = 'force_field1_mod'
        hopot.write_force_field(ff_name)
        input_filename = 'ff_opt.toml'
        crestcaller.write_input2file(input_ff_opt1, input_filename)
        crestcaller.calc_fitted_ff(input_filename)
    elif struc_id == 'struc2':
        ff_name = 'force_field2_mod'
        hopot.write_force_field(ff_name)
        input_filename = 'ff_opt.toml'
        crestcaller.write_input2file(input_ff_opt2, input_filename)
        crestcaller.calc_fitted_ff(input_filename)

    # copying
    path = cwd + '/crestopt.xyz'
    print(path)
    if os.path.exists(path):
        shutil.copyfile(xyz, struc_id + '/secmod.xyz')
        shutil.copyfile(path, xyz)
        print(path, ' copied to ', xyz)


    xtbcaller.hesscalc(hess,xyz)


    # readin_ff
    # modify_ff
    # opt_struc
    # recalc_hess

    # perform_avff

def main_run():
    print_box('TS Search')
    path_to_crest = "/home/guests/dbabushkina/master_thesis/crest/_build/crest"
    struc = Structure('struc1.xyz', 'wbo1')
    
    rc = first_ts_search(path_to_crest, struc.nat)

    rc = crestcaller.run_ts_search(startstruc=0,path_to_crest=path_to_crest, input_filename="avopt.toml")
    print('Further Optimiziation ends with rc', rc)
    if rc == 0:
        print('crestopt.xyz renamed to final.xyz')
        os.rename('crestopt.xyz', 'final.xyz')
    else:
        print('start1.xyz renamed to final.xyz')
        os.rename('start1.xyz', 'final.xyz')

def run_geometryoptimization(xtbcaller: Xtb, filename_in: str, filename_out: str):
    # TODO add a wbo calc and check to check whether topology changes and issue a warning
    xtbcaller = Xtb()
    xtbcaller.geomopt(filename_out, filename_in)
    print(f'.', end=" ")

if __name__ == "__main__":
    #### Parameters
    perform_geometryoptimization = True
    perform_alignment = False 
    perform_wbocalculation = True # ONLY set to false if wbo files already present 
    perform_hesscalculation = True # ONLY set to false if hess files already present

    xtb = Xtb()
    crestcaller = Crest()
    xyz0_1 = 'reac.xyz'
    xyz0_2= 'prod.xyz'
    wbo1 = 'wbo1'
    wbo2 = 'wbo2'
    hess1 = 'hess1'
    hess2 = 'hess2'
    xyz1 = 'struc1.xyz'
    xyz2 = 'struc2.xyz'
    struc1_id = 'struc1'
    struc2_id = 'struc2'

    # TODO add printing of all input values

    cwd = os.getcwd()
    if perform_geometryoptimization:
        try: 
            print(f'Geometry optimizations', end=" ")
            tic = time.perf_counter()
            run_geometryoptimization(xtb, xyz0_1, xyz1)
            run_geometryoptimization(xtb, xyz0_2, xyz2)
            toc = time.perf_counter()
            print(f"finished in {toc - tic:0.4f} seconds to files {xyz1} and {xyz2}")
        except FileExistsError as e:
            print(e) 
            print('If you desire a geometry optimization before the TS calculation, make sure to specify the correct filenames.')
        
    if perform_wbocalculation:
        try: 
            print(f'WBO calulations', end=" ")
            tic = time.perf_counter()
            xtb.wbocalc(wbo1, xyz1)
            xtb.wbocalc(wbo2, xyz2)
            toc = time.perf_counter()
            print(f"finished in {toc - tic:0.4f} seconds to files {wbo1} and {wbo2}")
        except FileExistsError as e:
            print(e) 
            print('If you desire a WBO calculation, please provide the xyz files.')

    if perform_hesscalculation:
        try: 
            print(f'Hessian calulations', end=" ")
            tic = time.perf_counter()
            xtb.hesscalc(hess1, xyz1)
            xtb.hesscalc(hess2, xyz2)
            toc = time.perf_counter()
            print(f"finished in {toc - tic:0.4f} seconds to files {hess1} and {hess2}")
        except FileExistsError as e:
            print(e) 
            print('If you desire a Hessian calculation, please provide the xyz files.')
        
    

    if perform_alignment:
        print('FF alignment will be performed as "perform_alignment" is set to', perform_alignment)
        try:    
            tic = time.perf_counter()
            main_align(struc1_id, xyz1, wbo1, 'hess1')
            toc = time.perf_counter()
            print(f"First alignment took: {toc - tic:0.4f} seconds")
        except:
            os.chdir(cwd)
            shutil.copyfile(struc1_id + "/original.xyz", 'struc1.xyz')
            print('Original structure will be used.')
        try:    
            tic = time.perf_counter()
            main_align(struc2_id, xyz2, wbo2, 'hess2')
            toc = time.perf_counter()
            print(f"Second alignment took: {toc - tic:0.4f} seconds")
        except:
            os.chdir(cwd)
            shutil.copyfile(struc2_id + "/original.xyz", 'struc2.xyz')
            print('Original structure will be used.')
        toc = time.perf_counter()
    else:
        print('FF alignment is not performed as "perform_alignment" is set to', perform_alignment)
        os.rename(xyz0_1, xyz1)
        os.rename(xyz0_2, xyz2)




    if os.path.exists('crestopt.xyz'):
        print(os.path.exists('crestopt.xyz'))
        os.remove('crestopt.xyz')

    tic = time.perf_counter()
    main_run()
    toc = time.perf_counter()
    print(f"TS search timing: {toc - tic:0.4f} seconds")
