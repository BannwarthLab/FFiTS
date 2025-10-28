
from dataclasses import dataclass, field
from src.datatype.structure_data import Structure 
from src.datatype.structure_data import StructurePath 
import os
from typing import List
import json


class Reaction:
    '''
    Complete information about a given Reaction.
    '''
    structure1: Structure 
    structure2: Structure
    # transition_structure: Structure
    unique_bonds: List[int]
    unique_atoms: List[int]
    
    def __init__(self, struc1: Structure, struc2: Structure, threshold=0.99):
        """
        Compares two WBO data files and returns Reaction with unique atom indices, 
        including those with significant WBO differences (larger then threshold).
        """
        wbo_dict1 = struc1.info.wbo 
        wbo_dict2 = struc2.info.wbo 
        all_bonds = set(wbo_dict1.keys()).union(set(wbo_dict2.keys()))
        
        unique_bonds = set()
        for bond in all_bonds:
            # print(bond)
            wbo1 = wbo_dict1.get(bond, 0)
            wbo2 = wbo_dict2.get(bond, 0)
            if bond not in wbo_dict1 or bond not in wbo_dict2: # or abs(wbo1 - wbo2) > threshold:
                unique_bonds.add(bond)
        print('Unique bonds for the reaction are:')
        print(unique_bonds)
        # return list(unique_bonds)
        unique_atoms = set()
        for bond in unique_bonds:
            unique_atoms.update(bond)

        sorted_unique_atoms = sorted(unique_atoms)
        print('Unique atoms', unique_atoms, type(unique_atoms))
        self.structure1= struc1
        self.structure2 = struc2
        self.unique_atoms = unique_atoms
        self.unique_bonds = unique_bonds
