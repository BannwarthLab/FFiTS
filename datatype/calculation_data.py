#!/bin/python

from dataclasses import dataclass
from datatype.structure_data import Structure 
from datatype.structure_data import StructurePath 

from typing import List


@dataclass
class Reaction:
    structure1: Structure 
    structure2: Structure
    # transition_structure: Structure
    unique_bonds: List[int]

@dataclass
class CalculationParameters:
    working_dir: str
    perform_alignment: bool
    perform_ts_search: bool
    struc1: StructurePath
    struc2: StructurePath
    ts: StructurePath