#!/bin/python

from dataclasses import dataclass
import structure_data
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
    struc1: StructurePath
    struc2: StructurePath
    ts: StructurePath