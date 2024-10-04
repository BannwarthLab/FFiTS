#!/bin/python

from src.datatype.calculation_data import *
from src.datatype.structure_data import *

import json 

def readin_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return CalculationParameters.from_json(config)



def readin_xyz(xyz_path):
    with open(xyz_path, 'r') as file:
        lines = file.readlines()
    xyz_dict = {}
    for i in range(2, len(lines)):
        line = lines[i]
        if line.strip():  # skip empty lines
            atom, x, y, z = line.split()
            coordinate = [float(x), float(y), float(z)]
            xyz_dict[atom+str(i-1)] = coordinate
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

# def readin_forcefield(ff_filename: str, ff: ForceField):
#     ff.c_bond = np.zeros((nat,nat))
#     ff.c_angle = np.zeros((nat,nat,nat))
#     ff.c_dihedral = np.zeros((nat,nat,nat,nat))
#     ff.c_lj = np.zeros((nat,nat))
#     ff.bond_list = []
#     ff.angle_list = []
#     ff.dihedral_list = []
#     ff.lj_list = []
#     ff.bondlengths = []
#     ff.angles = []
#     ff.dihedrals = []
#     ff.sigmas = []
#     with open(ff_filename, 'r') as file:
#         section = None
#         for line in file:
#             line = line.strip()
#             if line.startswith("$"):
#                 section = line.split(',')[0].strip(',')[1:]
#             elif section == "bonds":
#                 atom1, atom2, param, bondlength = map(float, line.split(','))
#                 ff.bond_list.append([int(atom1), int(atom2)])
#                 ff.c_bond[int(atom1)-1, int(atom2)-1] = param
#                 ff.bondlengths.append(bondlength)
#             elif section == "angles":
#                 atom1, atom2, atom3, param, angle = map(float, line.split(','))
#                 ff.angle_list.append([int(atom1), int(atom2), int(atom3)])
#                 ff.c_angle[int(atom1)-1, int(atom2)-1, int(atom3)-1] = param
#                 ff.angles.append(angle)
#             elif section == "dihedrals":
#                 atom1, atom2, atom3, atom4, param, dihedral_angle = map(float, line.split(','))
#                 ff.dihedral_list.append([int(atom1), int(atom2), int(atom3), int(atom4)])
#                 ff.c_dihedral[int(atom1)-1, int(atom2)-1, int(atom3)-1, int(atom4)-1] = param
#                 ff.dihedrals.append(dihedral_angle)
#             elif section == "lj-terms":
#                 atom1, atom2, param, sigma = map(float, line.split(','))
#                 ff.lj_list.append([int(atom1), int(atom2)])
#                 ff.c_lj[int(atom1)-1, int(atom2)-1] = param
#                 ff.sigmas.append(sigma)
#     if section == None:
#         raise Exception('File seems to be empty or not contain the section markers.')