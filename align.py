#!/bin/python

import networkx as nx
from datatype.structure_data import Structure 
from datatype.calculation_data import Reaction
import copy



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
