#!/bin/python

import numpy as np

import datatype.structure_data 


def increase_atompair_relevance(ff: ForceField, at1: int, at2: int, factor=10):
    if [at1, at2] not in ff.lj_list and [at2, at1] not in ff.lj_list and [at1,at2] not in [[0,0]]:
        print(ff.lj_list)
        raise Exception('Value does not make sense.')
    if at1 > at2:
        temp = at1
        at1 = at2
        at2 = temp
    ff.c_bond[at1-1, at2-1] = 0.35
    ff.c_bond[at2-1, at1-1] = 0.35 #self.c_bond[at1-1,at2-1] * factor
    ff.bond_list.append([at1, at2])
    ff.bondlengths.append(ff.sigmas[ff.lj_list.index([at1,at2])] * 0.9)
    ff.c_lj[at1-1, at2-1] = 0.0
    ff.c_lj[at2-1, at1-1] = 0.0
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