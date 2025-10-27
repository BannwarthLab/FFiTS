#!/bin/python

from src.datatype.calculation_data import *
from src.datatype.structure_data import *
from typing import Tuple, Dict, List
import numpy as np
import json 

def readin_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return CalculationParameters.from_json(config)


def readin_xyz(xyz_path: str) -> Tuple[int, str, np.array, List[str]]:
    if not os.path.exists(xyz_path):
        raise Exception(FileNotFoundError(f'{xyz_path} not found.'))
    with open(xyz_path, 'r') as file:
        lines = file.readlines()
    nat = int(lines[0].strip())
    comment = lines[1].strip()
    coordinates = []
    atom_types = []
    for i in range(2, len(lines)):
        line = lines[i]
        if line.strip():  # skip empty lines
            atom, x, y, z = line.split()
            atom_types.append(atom)
            coordinate = [float(x), float(y), float(z)]
            coordinates.append(coordinate)
    if len(coordinates) == nat:
        return nat, comment, np.array(coordinates), atom_types
    raise Exception(f'Number of atoms was given incorrectly in file {xyz_path}. Given number is {nat}, while counted number is {int(len(coordinates))}.')
    
def read_wbo_file(wbo_path) -> dict:
    """Reads WBO data from a file and parses it into a dictionary of bonds and their WBO values."""
    if not os.path.exists(wbo_path):
        raise Exception(FileNotFoundError(f'{wbo_path} not found.'))
    wbo_dict = {}
    with open(wbo_path, 'r') as file:
        for line in file:
            if line.strip():  # skip empty lines
                atom1, atom2, wbo = line.split()
                bond = tuple(sorted((int(atom1), int(atom2))))
                wbo_dict[bond] = float(wbo)
    return wbo_dict

import numpy as np

def read_hessian(file_path):
    """
    Reads a Hessian matrix from the xtb output format 
    """
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_lines = [line.strip() for line in lines if not line.lower().startswith("$hessian")]
    
    numbers = []
    for line in data_lines:
        if line:  # skip empty lines
            numbers.extend(map(float, line.split()))
    
    total_values = len(numbers)
    dim = int(np.sqrt(total_values))
    
    if dim * dim != total_values:
        raise ValueError(f"The number of Hessian elements ({total_values}) does not form a square matrix.")
    
    if dim % 3 != 0:
        raise ValueError(f"The Hessian dimensions ({dim}x{dim}) is not divisible by 3.")
    
    hessian = np.array(numbers).reshape((dim, dim))
    return hessian
