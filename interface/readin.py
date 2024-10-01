#!/bin/python

from datatype.calculation_data import *
from datatype.structure_data import *

import json 

def readin_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return CalculationParameters.from_json(config)

    

def readin_xyz(xyz_path):
    with open(xyz_path, 'r') as file:
        lines = file.readlines()
    return lines
    


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