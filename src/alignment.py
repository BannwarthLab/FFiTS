#!/bin/python

import networkx as nx
import copy
import numpy as np
import os
from src.datatype.structure_data import Structure, ForceField
from src.datatype.calculation_data import Reaction, CalculationParameters
from src.calculation import GeometryOptimization
from src.interface.exceptions import ConvergenceError


def get_ffatom_pair(reaction: Reaction, struc: Structure, forbidden_pair=[[-1,-1]]):
    at1 = 0
    at2 = 0
    
    for val in reaction.unique_bonds:
        G_temp = copy.deepcopy(struc.info.complete_graph)
        G_temp.add_edge(val[0], val[1])
        if len([G_temp.subgraph(c).copy() for c in nx.connected_components(G_temp)]) == 1:
            if struc.info.seperate_molecule_list[0].has_node(val[0]):
                at1 = val[0]
                at2 = val[1]
            else:
                at2 = val[0]
                at1 = val[1]
            pair = [at1, at2]
            if pair in forbidden_pair or reversed(pair) in forbidden_pair:
                continue
            print(f'   {val} is a connecting bond.')
            return at1, at2
        # print(val, 'is not a connecting bond.')
    return 0, 0


def increase_atompair_relevance(ff: ForceField, at1: int, at2: int, factor=10):
    if [at1, at2] not in ff.lj_list and [at2, at1] not in ff.lj_list and [at1,at2] not in [[0,0]]:
        print(ff.lj_list)
        raise Exception('Value does not make sense.')
    if at1 > at2:
        temp = at1
        at1 = at2
        at2 = temp
    ff.c_bond[at1-1, at2-1] = 0.35
    ff.c_bond[at2-1, at1-1] = 0.35 #self.c_bond[at1-1,at2-1] * factor
    ff.bond_list.append([at1, at2])
    ff.bondlengths.append(ff.sigmas[ff.lj_list.index([at1,at2])] * 0.9)
    ff.c_lj[at1-1, at2-1] = 0.0
    ff.c_lj[at2-1, at1-1] = 0.0


def modify_ff(struc: Structure, reac: Reaction):
    if len(struc.info.seperate_molecule_list) == 1:
        print(f'>! Only one molecule present in {struc.path.xyz_filename}, no alignment needed.')
        return struc.ff
    print(f'> Alignment of structures in {struc.path.xyz_filename} starts.')
    forbidden_pairs = [[]]
    atompairs_left = True
    while atompairs_left:
        at1, at2 = get_ffatom_pair(reac, struc, forbidden_pairs)
        forbidden_pairs.append([at1, at2])
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
    try:
        GeometryOptimization.with_ff_potential_crest(struc.path, struc.ff)
    except ConvergenceError:
        pass
    return struc.ff




def align(struc1: Structure, struc2: Structure, reac: Reaction, calc: CalculationParameters):
    if not calc.perform_alignment:
        return None 

    struc1.ff = modify_ff(struc1, reac)
    struc2.ff = modify_ff(struc2, reac)

    return struc1, struc2