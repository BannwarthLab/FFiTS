#!/bin/python

from dataclasses import dataclass, field
from structure_data import Structure 
from structure_data import StructurePath 
import os
from typing import List
import json


@dataclass
class Reaction:
    structure1: Structure 
    structure2: Structure
    # transition_structure: Structure
    unique_bonds: List[int]

@dataclass
class CalculationParameters:
    perform_alignment: bool
    perform_ts_search: bool
    struc1: StructurePath
    struc2: StructurePath
    ts: StructurePath
    working_dir: str = field(default_factory=os.getcwd)

    @staticmethod
    def from_json(config):
        return CalculationParameters(perform_alignment=config["calculation_details"]["perform_alignment"],
                                     perform_ts_search=config["calculation_details"]["perform_ts_search"],
                                     struc1=StructurePath.from_json(1, config["filepaths_structure1"]),
                                     struc2=StructurePath.from_json(2, config["filepaths_structure2"]),
                                     ts=StructurePath.from_json(3, config["filepaths_transition_state"]))


## test
# with open('/home/guests/dbabushkina/1_ts_search2024/pytsguess/config.json', 'r') as f:
#     conf = json.load(f)
# a = CalculationParameters.from_json(conf)
# print(a.ts.xyz_filename)