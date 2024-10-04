#!/bin/python

from src.datatype.calculation_data import *
from src.datatype.structure_data import *

import json 

def readin_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return CalculationParameters.from_json(config)


def readin_xyz(xyz_path, rdenergy=True):
    energy = 0
    with open(xyz_path, 'r') as file:
        lines = file.readlines()
    xyz_dict = {}
    for i in range(2, len(lines)):
        line = lines[i]
        if line.strip():  # skip empty lines
            atom, x, y, z = line.split()
            coordinate = [float(x), float(y), float(z)]
            xyz_dict[atom+str(i-1)] = coordinate
    if rdenergy:
        energy = float(lines[1].split(' ')[2])
    nat = int(lines[0].strip())
    return nat, energy, xyz_dict
    
def read_wbo_file(wbo_path) -> dict:
    """Reads WBO data from a file and parses it into a dictionary of bonds and their WBO values."""
    wbo_dict = {}
    with open(wbo_path, 'r') as file:
        for line in file:
            if line.strip():  # skip empty lines
                atom1, atom2, wbo = line.split()
                bond = tuple(sorted((int(atom1), int(atom2))))
                wbo_dict[bond] = float(wbo)
    return wbo_dict

