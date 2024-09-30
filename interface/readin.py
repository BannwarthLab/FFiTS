#!/bin/python

from datatype.calculation_data import *

import json 

def readin_config(calc_param: CalculationParameters, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    calc_param.struc1.path.xyz_filename =     config["filepaths_structure1"]["xyz"]
    calc_param.struc1.path.wbo_filename =     config["filepaths_structure1"]["wbo"]
    calc_param.struc1.path.hessian_filename = config["filepaths_structure1"]["hessian"]
    calc_param.struc1.path.ff_filename =      config["filepaths_structure1"]["forcefield"]

    calc_param.struc2.path.xyz_filename =     config["filepaths_structure2"]["xyz"]
    calc_param.struc2.path.wbo_filename =     config["filepaths_structure2"]["wbo"]
    calc_param.struc2.path.hessian_filename = config["filepaths_structure2"]["hessian"]
    calc_param.struc2.path.ff_filename =      config["filepaths_structure2"]["forcefield"]

    calc_param.ts.path.xyz_filename =     config["filepaths_transition_state"]["xyz"]
    calc_param.ts.path.wbo_filename =     config["filepaths_transition_state"]["wbo"]
    calc_param.ts.path.hessian_filename = config["filepaths_transition_state"]["hessian"]
    calc_param.ts.path.ff_filename =      config["filepaths_transition_state"]["forcefield"]

    calc_param.perform_alignment = config["calculation_details"]["perform_alignment"]
    calc_param.perform_ts_search = config["calculation_details"]["perform_ts_search"]

    

def readin_xyz(xyz_path):
    with open(xyz_path, 'r') as file:
        lines = file.readlines()
    return lines
    

def read_wbo_file(wbo_path):
    """Reads WBO data from a file and parses it into a dictionary of bonds and their WBO values."""
    wbo_dict = {}
    with open(wbo_path, 'r') as file:
        for line in file:
            if line.strip():  # skip empty lines
                atom1, atom2, wbo = line.split()
                bond = tuple(sorted((int(atom1), int(atom2))))
                wbo_dict[bond] = float(wbo)
    return wbo_dict

def readin_forcefield(struc: Structure):
    struc.ff.c_bond = np.zeros((nat,nat))
    struc.ff.c_angle = np.zeros((nat,nat,nat))
    struc.ff.c_dihedral = np.zeros((nat,nat,nat,nat))
    struc.ff.c_lj = np.zeros((nat,nat))
    struc.ff.bond_list = []
    struc.ff.angle_list = []
    struc.ff.dihedral_list = []
    struc.ff.lj_list = []
    struc.ff.bondlengths = []
    struc.ff.angles = []
    struc.ff.dihedrals = []
    struc.ff.sigmas = []
    with open(struc.path.ff_filename, 'r') as file:
        section = None
        for line in file:
            line = line.strip()
            if line.startswith("$"):
                section = line.split(',')[0].strip(',')[1:]
            elif section == "bonds":
                atom1, atom2, param, bondlength = map(float, line.split(','))
                struc.ff.bond_list.append([int(atom1), int(atom2)])
                struc.ff.c_bond[int(atom1)-1, int(atom2)-1] = param
                struc.ff.bondlengths.append(bondlength)
            elif section == "angles":
                atom1, atom2, atom3, param, angle = map(float, line.split(','))
                struc.ff.angle_list.append([int(atom1), int(atom2), int(atom3)])
                struc.ff.c_angle[int(atom1)-1, int(atom2)-1, int(atom3)-1] = param
                struc.ff.angles.append(angle)
            elif section == "dihedrals":
                atom1, atom2, atom3, atom4, param, dihedral_angle = map(float, line.split(','))
                struc.ff.dihedral_list.append([int(atom1), int(atom2), int(atom3), int(atom4)])
                struc.ff.c_dihedral[int(atom1)-1, int(atom2)-1, int(atom3)-1, int(atom4)-1] = param
                struc.ff.dihedrals.append(dihedral_angle)
            elif section == "lj-terms":
                atom1, atom2, param, sigma = map(float, line.split(','))
                struc.ff.lj_list.append([int(atom1), int(atom2)])
                struc.ff.c_lj[int(atom1)-1, int(atom2)-1] = param
                struc.ff.sigmas.append(sigma)
    if section == None:
        raise Exception('File seems to be empty or not contain the section markers.')