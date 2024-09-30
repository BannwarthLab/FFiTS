#!/bin/python

import numpy as np


class Forcefield:
    def __init__(self, filename_out='forcefield_out.txt', filename_in='forcefield_in.txt'):
        self.filename_in = filename_in
        self.filename_out = filename_out



class Hopot:
    def __init__(self, nat: int):
        """ 
        The force field (FF) is defined through FF parameters (matrices starting with c_) as well 
        as the reference values (bondlengths, angles, dihedral angles and repulsion references) 
        and corresponging atom combinations (between which two atoms is the bond). 
        """
        self.nat = nat
        self.c_bond = np.zeros((nat,nat))
        self.c_angle = np.zeros((nat,nat,nat))
        self.c_dihedral = np.zeros((nat,nat,nat,nat))
        self.c_lj = np.zeros((nat,nat))
        self.bond_list = []
        self.angle_list = []
        self.dihedral_list = []
        self.lj_list = []
        self.bondlengths = []
        self.angles = []
        self.dihedrals = []
        self.sigmas = []
        # self.readin_forcefield('force_field2.txt')

    def readin_forcefield(self, filename):
        with open(filename, 'r') as file:
            section = None
            for line in file:
                line = line.strip()
                if line.startswith("$"):
                    section = line.split(',')[0].strip(',')[1:]
                elif section == "bonds":
                    atom1, atom2, param, bondlength = map(float, line.split(','))
                    self.bond_list.append([int(atom1), int(atom2)])
                    self.c_bond[int(atom1)-1, int(atom2)-1] = param
                    self.bondlengths.append(bondlength)
                elif section == "angles":
                    atom1, atom2, atom3, param, angle = map(float, line.split(','))
                    self.angle_list.append([int(atom1), int(atom2), int(atom3)])
                    self.c_angle[int(atom1)-1, int(atom2)-1, int(atom3)-1] = param
                    self.angles.append(angle)
                elif section == "dihedrals":
                    atom1, atom2, atom3, atom4, param, dihedral_angle = map(float, line.split(','))
                    self.dihedral_list.append([int(atom1), int(atom2), int(atom3), int(atom4)])
                    self.c_dihedral[int(atom1)-1, int(atom2)-1, int(atom3)-1, int(atom4)-1] = param
                    self.dihedrals.append(dihedral_angle)
                elif section == "lj-terms":
                    atom1, atom2, param, sigma = map(float, line.split(','))
                    self.lj_list.append([int(atom1), int(atom2)])
                    self.c_lj[int(atom1)-1, int(atom2)-1] = param
                    self.sigmas.append(sigma)
        if section == None:
            raise Exception('File seems to be empty or not contain the section markers.')

    def write_force_field(self, filename):
        print('WRITES TO', filename)
        with open(filename, 'w') as file:
            if len(self.bond_list) > 0:
                file.write(f"$bonds, {len(self.bond_list)}\n")
                for (atom1, atom2), bondlength in zip(self.bond_list, self.bondlengths):
                    param = self.c_bond[atom1-1, atom2-1]
                    file.write(f"{atom1}, {atom2}, {param}, {bondlength}\n")

            if len(self.angle_list) > 0:
                file.write(f"$angles, {len(self.angle_list)}\n")
                for (atom1, atom2, atom3), angle in zip(self.angle_list, self.angles):
                    param = self.c_angle[atom1-1, atom2-1, atom3-1]
                    file.write(f"{atom1}, {atom2}, {atom3}, {param}, {angle}\n")

            if len(self.dihedral_list) > 0:
                file.write(f"$dihedrals, {len(self.dihedral_list)}\n")
                for (atom1, atom2, atom3, atom4), dihedral_angle in zip(self.dihedral_list, self.dihedrals):
                    param = self.c_dihedral[atom1-1, atom2-1, atom3-1, atom4-1]
                    file.write(f"{atom1}, {atom2}, {atom3}, {atom4}, {param}, {dihedral_angle}\n")

            if len(self.lj_list) > 0:
                file.write(f"$lj-terms, {len(self.lj_list)}\n")
                for (atom1, atom2), sigma in zip(self.lj_list, self.sigmas):
                    param = self.c_lj[atom1-1, atom2-1]
                    file.write(f"{atom1}, {atom2}, {param}, {sigma}\n")

    def increase_atompair_relevance(self, at1: int, at2: int, factor=10):
        if [at1, at2] not in self.lj_list and [at2, at1] not in self.lj_list and [at1,at2] not in [[0,0]]:
            print(self.lj_list)
            raise Exception('Value does not make sense.')
        if at1 > at2:
            temp = at1
            at1 = at2
            at2 = temp
        self.c_bond[at1-1, at2-1] = 0.35
        self.c_bond[at2-1, at1-1] = 0.35 #self.c_bond[at1-1,at2-1] * factor
        self.bond_list.append([at1, at2])
        self.bondlengths.append(self.sigmas[self.lj_list.index([at1,at2])] * 0.9)
        self.c_lj[at1-1, at2-1] = 0.0
        self.c_lj[at2-1, at1-1] = 0.0
        # self.sigmas[self.lj_list.index([at2,at1])] = self.sigmas[self.lj_list.index([at2,at1])] * 0.8
        # self.sigmas[self.lj_list.index([at1,at2])] = 

    


# ### Test code
# if __name__ == '__main__':
#     d = Hopot(16)
#     at_pair1 = [2,15]
#     at_pair2 = [12,14]
#     d.readin_forcefield('force_field2.txt')
#     d.c_bond = np.multiply(d.c_bond, 1)
#     d.c_angle = np.multiply(d.c_angle, 1)
#     d.c_dihedral = np.multiply(d.c_dihedral, 1)
#     at1 = 2
#     at2 = 15
#     d.c_lj[at1-1,at2-1] =2
#     at1 = 12
#     at2 = 14
#     d.c_lj[at1-1,at2-1] = d.c_lj[at1-1,at2-1] * 2
#     d.write_force_field('ff_new.txt')
#     # print(d.bond_list)
#     # print(d.angle_list)
    # print(d.bondlengths)
    # print(d.c_angle[2,3,9])