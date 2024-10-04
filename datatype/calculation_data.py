#!/bin/python

from dataclasses import dataclass, field
from datatype.structure_data import Structure 
from datatype.structure_data import StructurePath 
import os
from typing import List
import json


@dataclass
class Reaction:
    structure1: Structure 
    structure2: Structure
    # transition_structure: Structure
    unique_bonds: List[int]
    
    @staticmethod
    def with_unique_bonds(struc1: Structure, struc2: Structure, threshold=0.8):
        """Compares two WBO data files and returns unique atom indices, 
        including those with significant WBO differences."""
        wbo_dict1 = struc1.info.wbo 
        wbo_dict2 = struc2.info.wbo 
        all_bonds = set(wbo_dict1.keys()).union(set(wbo_dict2.keys()))
        
        unique_bonds = set()
        for bond in all_bonds:
            # print(bond)
            wbo1 = wbo_dict1.get(bond, 0)
            wbo2 = wbo_dict2.get(bond, 0)
            if bond not in wbo_dict1 or bond not in wbo_dict2 or abs(wbo1 - wbo2) > threshold:
                unique_bonds.add(bond)
        print('')
        print(unique_bonds)
        # return list(unique_bonds)
        unique_atoms = set()
        for bond in unique_bonds:
            unique_atoms.update(bond)

        sorted_unique_atoms = sorted(unique_atoms)
        return Reaction(structure1=struc1, structure2=struc2, unique_bonds=unique_bonds)


@dataclass
class CalculationParameters:
    perform_geometryoptimization: bool 
    perform_hesscalculation: bool 
    perform_wbocalculation: bool
    perform_alignment: bool
    perform_ts_search: bool
    struc1: StructurePath
    struc2: StructurePath
    ts: StructurePath
    working_dir: str = field(default_factory=os.getcwd)

    @staticmethod
    def from_json(config):
        return CalculationParameters(perform_geometryoptimization=config["calculation_details"]["perform_geometryoptimization"],
                                     perform_hesscalculation=config["calculation_details"]["perform_hesscalculation"],
                                     perform_wbocalculation=config["calculation_details"]["perform_wbocalculation"],
                                     perform_alignment=config["calculation_details"]["perform_alignment"],
                                     perform_ts_search=config["calculation_details"]["perform_ts_search"],
                                     struc1=StructurePath.from_json(1, config["filepaths_structure1"]),
                                     struc2=StructurePath.from_json(2, config["filepaths_structure2"]),
                                     ts=StructurePath.from_json(3, config["filepaths_transition_state"]))


## test
# with open('/home/guests/dbabushkina/1_ts_search2024/pytsguess/config.json', 'r') as f:
#     conf = json.load(f)
# a = CalculationParameters.from_json(conf)
# print(a.ts.xyz_filename)