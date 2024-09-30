#!/bin/python

from dataclasses import dataclass
from structure_data import Structure
from typing import List


@dataclass
class Reaction:
    structure1: Structure 
    structure2: Structure
    transition_structure: Structure
    unique_bonds: List[int]

@dataclass
class CalculationParameters:
    struc1: StructurePath
    struc2: StructurePath