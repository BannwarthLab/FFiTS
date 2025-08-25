#!/bin/python

import networkx as nx
import copy
import numpy as np
import os
from src.datatype.structure_data import Structure, ForceField
from src.datatype.calculation_data import Reaction, CalculationParameters
from src.calculation import GeometryOptimization, HessianCalculation
from src.interface.exceptions import ConvergenceError


def get_connecting_pair(reaction: Reaction, struc: Structure, forbidden_pair=[[-1,-1]]):
    '''
    Returns an atom pair, which connects two molecules in the structure 'struc' through a bond present in the other structure of Reaction 'reaction'. If the found atom pair is present in the list 'forbidden_pair', the search continues.
    '''
    
    at1 = 0
    at2 = 0
    
    for val in reaction.unique_bonds:
        G_temp = copy.deepcopy(struc.info.complete_graph)
        G_temp.add_edge(val[0], val[1])
        if len([G_temp.subgraph(c).copy() for c in nx.connected_components(G_temp)]) == 1: 
            #TODO this does not work like that for structures with more then 2 molecules
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
    return 0, 0


def add_atompair_to_bondlist(ff: ForceField, at1: int, at2: int, factor=0.35):
    '''
    Adds atompair (at1, at2) to bondlist of force field 'ff' and sets corresponding FF parameters to value 'factor'.
    '''
    if [at1, at2] in ff.bond_list and [at2, at1] in ff.bond_list and [at1,at2] not in [[0,0]]:
        print(ff.bond_list)
        raise Exception(f'Atompair {at1, at2} is already present in bond list of force field {ff}')
    if at1 > at2:
        temp = at1
        at1 = at2
        at2 = temp
    ff.c_bond[at1-1, at2-1] = factor
    ff.c_bond[at2-1, at1-1] = factor #self.c_bond[at1-1,at2-1] * factor
    ff.bond_list.append([at1, at2])
    ff.bondlengths.append(ff.sigmas[ff.lj_list.index([at1,at2])] * 0.9)
    ff.c_lj[at1-1, at2-1] = 0.0
    ff.c_lj[at2-1, at1-1] = 0.0


def modify_ff(struc: Structure, reac: Reaction, value=2):
    '''
    Modifies FF for Alignment and performs alignment for Structure 'struc' according to the differences of the WBOs of the structures in Reaction 'reac'. Value corresponds to the increase of the stiffness in the molecule by increasing the FF parameters of bonds, angles and dihedral angles. 
    '''
    if len(struc.info.seperate_molecule_list) == 1:
        print(f'>! Only one molecule present in {struc.path.xyz_filename}, no alignment needed.')
        return struc.ff
    print(f'> Alignment of structures in {struc.path.xyz_filename} starts.')
    forbidden_pairs = [[]]
    atompairs_left = True
    while atompairs_left:
        at1, at2 = get_connecting_pair(reac, struc, forbidden_pairs)
        forbidden_pairs.append([at1, at2])
        if at1 == 0 and at2 == 0:
            atompairs_left = False
            break 
        if [at1, at2] in struc.ff.bond_list:
            break
        add_atompair_to_bondlist(struc.ff, at1, at2)
    np.multiply(struc.ff.c_bond, value)
    np.multiply(struc.ff.c_angle, value)
    np.multiply(struc.ff.c_dihedral, value)
    np.multiply(struc.ff.c_lj, 1/value)
    os.remove('crestopt.xyz')

    struc.ff.write_force_field(struc.path.ff_filename)
    print(f'> FF for alignment written to {struc.path.ff_filename}')
    try: 
    # TODO das ist schon nicht schöner code. das macht keinen sinn, dass bei nicht konvergenz immer die letzte struktur genommen wird.
        GeometryOptimization.with_ff_potential_crest(struc.path, struc.ff)
    except ConvergenceError:
        pass
    return struc.ff




def align(struc1: Structure, struc2: Structure, reac: Reaction, calc: CalculationParameters):
    '''
    if perform_alignment is set to True: Performs alignment of Structures 'struc1' and 'struc2' by rotating the molecules in the seperate Structures corresponding to the differences between the Structures. Performs corresponding Hessian Calculations afterwards.
    '''
    if not calc.perform_alignment:
        return struc1, struc2 

    struc1.ff = modify_ff(struc1, reac)
    # HessianCalculation.with_xtb(struc1.path)
    struc2.ff = modify_ff(struc2, reac)
    # HessianCalculation.with_xtb(struc2.path)

    return struc1, struc2