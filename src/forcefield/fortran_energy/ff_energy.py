from src.forcefield.fortran_energy.fortran_bindings import *
from src.datatype.structure_data import ForceField
from src.forcefield.fortran_energy.geometry_calc import angle, bondlength, dihedral_angle
import numpy as np

def energy_ff(geometry_displ, ff: ForceField):
    """
    Python equivalent of the Fortran subroutine energy_ff.
    
    Parameters
    ----------
    geometry_displ : np.ndarray
        Shape (3, nat). Atomic displacements or positions.
    ff : object
        Holds FF parameters, including lists and constants.
        Must define:
          nat, count_bond, bond_list, c_bond, bondlengths,
          count_angle, angle_list, c_angle, angles,
          count_dihedral, dihedral_list, c_dihedral, dihedrals,
          count_lj, lj_list, c_lj, lj_lengths.
    Returns
    -------
    float
        The computed energy.
    """
    
    fact_bond = 1.0
    fact_ang = 1.0
    fact_dih = 1.0
    energy = 0.0
    
    # Bonds
    for a in range(len(ff.bond_list)):
        i = ff.bond_list[0, a] - 1  
        j = ff.bond_list[1, a] - 1
        bondlength_displ = bondlength(geometry_displ, i, j)
        energy += fact_bond * ff.c_bond[i, j]**2 * (bondlength_displ - ff.bondlengths[a])**2
    
    # Angles
    for a in range(len(ff.angle_list)):
        i = ff.angle_list[0, a] - 1
        j = ff.angle_list[1, a] - 1
        l = ff.angle_list[2, a] - 1
        angle_displ = angle(geometry_displ, i, j, l)
        energy += fact_ang * ff.c_angle[i, j, l]**2 * (angle_displ - ff.angles[a])**2
    
    # Dihedrals
    for a in range(len(ff.dihedral_list)):
        i = ff.dihedral_list[0, a] - 1
        j = ff.dihedral_list[1, a] - 1
        l = ff.dihedral_list[2, a] - 1
        m = ff.dihedral_list[3, a] - 1
        dihedral_displ = dihedral_angle(geometry_displ, i, j, l, m)
        
        energy += fact_dih * ff.c_dihedral[i, j, l, m]**2 * (
            (np.cos(ff.dihedrals[a]) - np.cos(dihedral_displ))**2
            + (np.sin(ff.dihedrals[a]) - np.sin(dihedral_displ))**2
        )
    
    # Lennard-Jones terms
    for a in range(len(ff.lj_list)):
        i = ff.lj_list[0, a] - 1
        j = ff.lj_list[1, a] - 1
        bondlength_displ = bondlength(geometry_displ, i, j)
        energy += 4 * ff.c_lj[i, j]**2 * (ff.lj_lengths[a]/bondlength_displ)**12
        # full LJ: - (ff.lj_lengths[a]/bondlength_displ)**6
    
    return energy


