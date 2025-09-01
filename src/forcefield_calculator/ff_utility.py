import numpy as np


def convert_xyz_to_fortranstyle(nat, xyz) -> np.array:
    x = []
    y = []
    z = []
    for i in range(nat):
        x.append(xyz[i][0])
        y.append(xyz[i][1])
        z.append(xyz[i][2])
    return np.array([x,y,z],order='F')

def angstrom2bohr(val):
    if type(val) == np.array:
        return np.divide(val, 1/1.8897259)
    return val/(1/1.8897259)
