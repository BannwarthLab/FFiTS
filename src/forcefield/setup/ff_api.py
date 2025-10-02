from src.datatype.structure_data import Structure, ForceField
from src.forcefield.setup.define_starting_parameters import *

#### TODO correct PSEUDO CODE
# def singlepoint(current_xyz, ff: ForceField):
    # return ff_energy(current_xyz, ff), ff_gradient(current_xyz, ff)

def setup_ff(struc: Structure):
    ff = struc.ff
    define_relevant_bonds(struc)
    