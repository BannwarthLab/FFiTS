#!/bin/python

from dataclasses import dataclass
from typing import List
import networkx as nx
import numpy as np

@dataclass
class StructurePath:
    """
    Paths/Filenames and ID for a structure
    """
    id: int
    xyz_filename: str 
    hess_filename: str 
    wbo_filename: str 
    ff_filename: str

@dataclass
class ForceField:
    """ 
    The force field (FF) is defined through FF parameters (matrices starting with c_) as well 
    as the reference values (bondlengths, angles, dihedral angles and repulsion references) 
    and corresponging atom combinations (between which two atoms is the bond). 
    """
    nat: int
    c_bond = np.array
    c_angle = np.array
    c_dihedral = np.array
    c_lj = np.array
    bond_list: List[int]
    angle_list: List[int]
    dihedral_list: List[int]
    lj_list: List[int]
    bondlengths: List[float]
    angles: List[float]
    dihedrals: List[float]
    sigmas: List[float]

@dataclass
class StructuralInformation:
    nat: int
    xyz: List[float]
    wbo_list: List[float]
    molecule_count: int
    complete_graph: nx.Graph
    seperate_molecule_list: List[nx.Graph]



@dataclass
class Structure:
    path: StructurePath 
    ff: ForceField
    info: StructuralInformation
