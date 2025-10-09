import src.forcefield.fortran_energy.fortran_bindings as fb
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
    
    # bonds
    ff.bonds["current_reference"] = ff.bonds["atoms"].apply(
        lambda atoms: bondlength(geometry_displ, atoms[0], atoms[1]))
    ff.bonds['energy'] = fact_bond * ff.bonds['parameter']**2 * (ff.bonds['current_reference'] - ff.bonds['reference_value'])**2
    energy += ff.bonds['energy'].sum()

    # angles
    ff.angles["current_reference"] = ff.angles["atoms"].apply(
        lambda atoms: angle(geometry_displ, atoms[0], atoms[1], atoms[2]))
    ff.angles['energy'] = fact_ang * ff.angles['parameter']**2 * (ff.angles['current_reference'] - ff.angles['reference_value'])**2
    energy += ff.angles['energy'].sum()

    # dihedrals
    ff.dihedrals["current_reference"] = ff.dihedrals["atoms"].apply(
        lambda atoms: dihedral_angle(geometry_displ, atoms[0], atoms[1]))
    ff.dihedrals['energy'] = fact_dih * ff.dihedrals['parameter']**2 * (
        (np.cos(ff.dihedrals['current_reference']) - np.cos(ff.dihedrals['reference_value']))**2 
        + (np.sin(ff.dihedrals['current_reference']) - np.sin(ff.dihedrals['reference_value']))**2 
    )
    energy += ff.dihedrals['energy'].sum()

    # repulsive
    ff.repulsive["current_reference"] = ff.repulsive["atoms"].apply(
        lambda atoms: bondlength(geometry_displ, atoms[0], atoms[1]))
    ff.repulsive['energy'] = fact_bond * ff.repulsive['parameter']**2 * (ff.repulsive['reference_value']/ff.repulsive['current_reference'])**12
    energy += ff.repulsive['energy'].sum()

    # # Bonds
    # for a in range(len(ff.c_bond)):
    #     i = ff.bond_list[0, a] - 1  
    #     j = ff.bond_list[1, a] - 1
    #     bondlength_displ = bondlength(geometry_displ, i, j)
    #     energy += fact_bond * ff.c_bond[i, j]**2 * (bondlength_displ - ff.bondlengths[a])**2
    
    # # Angles
    # for a in range(len(ff.angle_list)):
    #     i = ff.angle_list[0, a] - 1
    #     j = ff.angle_list[1, a] - 1
    #     l = ff.angle_list[2, a] - 1
    #     angle_displ = angle(geometry_displ, i, j, l)
    #     energy += fact_ang * ff.c_angle[i, j, l]**2 * (angle_displ - ff.angles[a])**2
    
    # # Dihedrals
    # for a in range(len(ff.dihedral_list)):
    #     i = ff.dihedral_list[0, a] - 1
    #     j = ff.dihedral_list[1, a] - 1
    #     l = ff.dihedral_list[2, a] - 1
    #     m = ff.dihedral_list[3, a] - 1
    #     dihedral_displ = dihedral_angle(geometry_displ, i, j, l, m)
        
    #     energy += fact_dih * ff.c_dihedral[i, j, l, m]**2 * (
    #         (np.cos(ff.dihedrals[a]) - np.cos(dihedral_displ))**2
    #         + (np.sin(ff.dihedrals[a]) - np.sin(dihedral_displ))**2
    #     )
    
    # # Lennard-Jones terms
    # for a in range(len(ff.lj_list)):
    #     i = ff.lj_list[0, a] - 1
    #     j = ff.lj_list[1, a] - 1
    #     bondlength_displ = bondlength(geometry_displ, i, j)
    #     energy += 4 * ff.c_lj[i, j]**2 * (ff.lj_lengths[a]/bondlength_displ)**12
    #     # full LJ: - (ff.lj_lengths[a]/bondlength_displ)**6
    
    return energy





# -------------------------------------------------------------------------
# --- Bond Gradient -------------------------------------------------------
# -------------------------------------------------------------------------
def bond_gradient(xyz: np.ndarray, ff: ForceField, gradient: np.ndarray):
    """adds gradient"""
    for _, row in ff.bonds.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']

        fb.get_single_bond_gradient(xyz, atoms, param, c_val, gradient)


# -------------------------------------------------------------------------
# --- Angle Gradient ------------------------------------------------------
# -------------------------------------------------------------------------
def angle_gradient(xyz: np.ndarray, ff: ForceField, gradient: np.ndarray):
    """adds gradient"""
    for _, row in ff.angles.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']
        fb.get_single_angle_gradient(xyz, atoms, param, c_val, gradient)


# -------------------------------------------------------------------------
# --- Dihedral Gradient ---------------------------------------------------
# -------------------------------------------------------------------------
def dihedral_gradient(xyz: np.ndarray, ff: ForceField, gradient: np.ndarray):
    """adds gradient"""
    for _, row in ff.dihedrals.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']

        fb.get_single_dihedral_gradient(xyz, atoms, param, c_val, gradient)


# -------------------------------------------------------------------------
# --- Repulsive Gradient --------------------------------------------------
# -------------------------------------------------------------------------
def repulsive_gradient(xyz: np.array, ff: ForceField, gradient: np.ndarray):
    """adds gradient"""
    for _, row in ff.repulsive.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']
        fb.get_single_repulsive_gradient(xyz, atoms, row['parameter'], c_val, gradient)


def complete_gradient(xyz_displaced: np.ndarray, ff: ForceField) -> np.ndarray:
    """
    Computes total gradient (flattened, Fortran order) for all force field terms.
    Uses DataFrame-based force field representation.
    """
    gradient = np.zeros((ff.nat * 3), order='F')

    bond_gradient(xyz_displaced, ff, gradient)
    angle_gradient(xyz_displaced, ff, gradient)
    dihedral_gradient(xyz_displaced, ff, gradient)
    #repulsive_gradient(xyz_displaced, ff, gradient)

    return gradient




# -------------------------------------------------------------------------
# --- Bond hessian -------------------------------------------------------
# -------------------------------------------------------------------------
def bond_hessian(xyz: np.ndarray, ff: ForceField, hessian: np.ndarray):
    """adds hessian"""
    for _, row in ff.bonds.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']

        fb.get_single_bond_hessian(xyz, atoms, param, c_val, hessian)


# -------------------------------------------------------------------------
# --- Angle hessian ------------------------------------------------------
# -------------------------------------------------------------------------
def angle_hessian(xyz: np.ndarray, ff: ForceField, hessian: np.ndarray):
    """adds hessian"""
    for _, row in ff.angles.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']
        fb.get_single_angle_hessian(xyz, atoms, param, c_val, hessian)


# -------------------------------------------------------------------------
# --- Dihedral hessian ---------------------------------------------------
# -------------------------------------------------------------------------
def dihedral_hessian(xyz: np.ndarray, ff: ForceField, hessian: np.ndarray):
    """adds hessian"""
    for _, row in ff.dihedrals.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']

        fb.get_single_dihedral_hessian(xyz, atoms, param, c_val, hessian)


# -------------------------------------------------------------------------
# --- Repulsive hessian --------------------------------------------------
# -------------------------------------------------------------------------
def repulsive_hessian(xyz: np.array, ff: ForceField, hessian: np.ndarray):
    """adds hessian"""
    for _, row in ff.repulsive.iterrows():
        atoms = row['atoms']      
        param = row['reference_value']
        c_val = row['parameter']
        fb.get_single_repulsive_hessian(xyz, atoms, row['parameter'], c_val, hessian)


def complete_hessian(xyz_displaced: np.ndarray, ff: ForceField) -> np.ndarray:
    """
    Computes total hessian (flattened, Fortran order) for all force field terms.
    Uses DataFrame-based force field representation.
    """
    hessian = np.zeros((ff.nat * 3, ff.nat * 3), order='F')

    bond_hessian(xyz_displaced, ff, hessian)
    angle_hessian(xyz_displaced, ff, hessian)
    dihedral_hessian(xyz_displaced, ff, hessian)
    #repulsive_hessian(xyz_displaced, ff, hessian)

    return hessian
