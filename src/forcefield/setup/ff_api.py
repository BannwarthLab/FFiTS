from src.datatype.structure_data import Structure, ForceField


#### TODO correct PSEUDO CODE
def singlepoint(current_xyz, ff: ForceField):
    return ff_energy(current_xyz, ff), ff_gradient(current_xyz, ff)

