import src.fortran_forcefield.fortran_forcefield as fortran_ff
import numpy as np
from src.datatype.structure_data import ForceField 

class ForceFieldEnergyCalculator:
    def __init__(self, geometry, ff: ForceField) -> None:
        self.ff = ff
        self.geometry = geometry # TODO write code to translate angström to bohr and correct orientation
        
    def energy(self) -> float:
        pass

    def gradient(self) -> np.array:
        gradient = np.zeros(self.ff.nat, order='F')
        for i in range(len(self.ff.bond_list)):
            a1 = self.ff.bond_list[i][0]
            a2 = self.ff.bond_list[i][1]
            fortran_ff.bond_derivatives.get_single_bond_gradient(self.geometry, self.ff.bond_list[i], self.ff.bondlengths[i], self.ff.c_bond[a1-1][a2-1]**2, gradient)
        
        for i in range(len(self.ff.angle_list)):
            a1 = self.ff.angle_list[i][0]
            a2 = self.ff.angle_list[i][1]
            a3 = self.ff.angle_list[i][2]
            fortran_ff.angle_derivatives.get_single_dihedral_gradient(self.geometry, self.ff.bond_list[i], self.ff.bondlengths[i], self.ff.c_bond[a1-1][a2-1]**2, gradient)
        
        for i in range(len(self.ff.bond_list)):
            a1 = self.ff.bond_list[i][0]
            a2 = self.ff.bond_list[i][1]
            fortran_ff.bond_derivatives.get_single_bond_gradient(self.geometry, self.ff.bond_list[i], self.ff.bondlengths[i], self.ff.c_bond[a1-1][a2-1], gradient)
        

    def hessian(self) -> np.array:
        pass