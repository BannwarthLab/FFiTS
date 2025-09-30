import src.fortran_forcefield.fortran_forcefield as fortran_ff
import numpy as np
from src.datatype.structure_data import ForceField, Structure, StructuralInformation 

# class FortranForceFieldWrapper:
#     def __init__(self, structure: Structure):
#         self.struc: StructuralInformation = structure.info
#         self.ff: ForceField = structure.ff


# class Gradient(FortranForceFieldWrapper):
#     def __init__(self, struc: Structure):
#         super().__init__(struc)
#         self.bond = Gradient.Bond(struc)
#         self.angle = self.Angle(self)
#         self.dihedral = self.Dihedral(self)


#     def complete(self):
#         ff = self.ff
#         gradient = np.zeros((ff.nat*3),order='F')

#         self.bond.full(gradient)
#         self.angle.full(gradient)
#         self.dihedral.full(gradient)

#         return gradient

#     class Bond:
#         def __init__(self, struc: Structure):
#             self.ff = struc.ff

#         def full(self, gradient):            
#             """ 
#             adds gradient of all bonds onto input gradient
#             """
#             ff = self.ff
#             for i in range(len(ff.bond_list)):
#                 atom1 = ff.bond_list[i][0]
#                 atom2 = ff.bond_list[i][1]
#                 self.single_bond(ff.bond_list[i], ff.bondlengths[i], ff.c_bond[atom1][atom2], gradient)

#         def single(self, bond_atoms: np.array, bondlength: float, c: float, gradient: np.array):
#             """ 
#             adds gradient of specific bond onto input gradient
#             """
#             fortran_ff.bond_derivatives.get_single_bond_gradient(self.struc.fortran_xyz, bond_atoms, 
#                                                                 bondlength, c, gradient)
#             return gradient

#     class Angle:

#         def full(self, gradient):
#             ff = self.ff
#             for i in range(len(ff.bond_list)):
#                 atom1 = ff.bond_list[i][0]
#                 atom2 = ff.bond_list[i][1]
#                 self.single_(ff.bond_list[i], ff.bondlengths[i], ff.c_bond[atom1][atom2], gradient)
            
#         def single(self, angle_atoms: np.array, bondlength: float, c: float, gradient: np.array):
#             """ 
#             adds gradient of specific angle onto input gradient
#             """
#             fortran_ff.angle_derivatives.get_single_angle_gradient(self.struc.fortran_xyz, angle_atoms, 
#                                                                 bondlength, c, gradient)
#             return gradient

#     class Dihedral: 
#         def full(self, gradient):
#             ff = self.ff
#             for i in range(len(ff.bond_list)):
#                 atom1 = ff.bond_list[i][0]
#                 atom2 = ff.bond_list[i][1]
#                 self.single_bond(ff.bond_list[i], ff.bondlengths[i], ff.c_bond[atom1][atom2], gradient)

#         def single(self, dihedral_atoms: np.array, dihedral_angle: float, c: float, gradient: np.array):
#             """ 
#             adds gradient of specific dihedral angle onto input gradient
#             """
#             fortran_ff.dihedral_derivatives.get_single_dihedral_gradient(self.struc.fortran_xyz, dihedral_atoms, 
#                                                                         dihedral_angle, c, gradient)
#             return gradient


class ForceFieldEnergyCalculator:
    def __init__(self, geometry, struc: Structure) -> None:
        self.struc: Structure = struc
        self.ff: ForceField = struc.ff
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