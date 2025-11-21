import numpy as np
import os
from ffits.datatype.structure_data import ForceField, StructuralInformation, Structure, StructurePath
from ffits.ts_guess.parameterize_ff import derivative_c_first_atomwise, derivative_c_second_atomwise, ff_fit_objective_function, repulsive_derivative_c_first_atomwise, fit_ff_to_hessian, repulsive_derivative_c_second_atomwise
from ffits.forcefield.python_interface.ff_energy import complete_hessian
from ffits.io.reader import readin_xyz, read_wbo_file, read_hessian
from ffits.ts_guess.define_starting_parameters import fill_ff
import copy

def analy_first_derivative(ff: ForceField, info: StructuralInformation):
    derivatives = []  
    xyz = info.fortran_xyz

    hessian_ff = complete_hessian(xyz, ff)

    for row in ff.bonds.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        derivatives.append(derivative_c_first_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        ))
    
    for row in ff.angles.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        l = row.atoms[2]
        derivatives.append(derivative_c_first_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, c=row.parameter
        ))
    
    for row in ff.dihedrals.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        l = row.atoms[2]
        m = row.atoms[3]
        derivatives.append(derivative_c_first_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, atom4=m, c=row.parameter
        ))
    
    for row in ff.repulsive.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        derivatives.append(repulsive_derivative_c_first_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        ))
    
    return derivatives

def num_first_derivative(ff: ForceField, info: StructuralInformation, delta: float = 1e-5):
        xyz = info.fortran_xyz
        derivatives = []
        for i in range(len(ff.bonds)): 
            ff_m = copy.deepcopy(ff)
            ff_m.bonds.iloc[i, ff_m.bonds.columns.get_loc('parameter')] -= delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.bonds.iloc[i, ff_m.bonds.columns.get_loc('parameter')] += delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_p - res_m)/(delta * 2))

        for i in range(len(ff.angles)): 
            ff_m = copy.deepcopy(ff)
            ff_m.angles.iloc[i, ff_m.angles.columns.get_loc('parameter')] -= delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.angles.iloc[i, ff_m.angles.columns.get_loc('parameter')] += delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_p - res_m)/(delta * 2))

        for i in range(len(ff.dihedrals)): 
            ff_m = copy.deepcopy(ff)
            ff_m.dihedrals.iloc[i, ff_m.dihedrals.columns.get_loc('parameter')] -= delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.dihedrals.iloc[i, ff_m.dihedrals.columns.get_loc('parameter')] += delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_p - res_m)/(delta * 2))

        for i in range(len(ff.repulsive)): 
            ff_m = copy.deepcopy(ff)
            ff_m.repulsive.iloc[i, ff_m.repulsive.columns.get_loc('parameter')] -= delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.repulsive.iloc[i, ff_m.repulsive.columns.get_loc('parameter')] += delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_p - res_m)/(delta * 2))
        return derivatives

def test_objfun_first_derivatives():
    path1 = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule')
    nat, _, xyz, atom_types = readin_xyz(os.path.join(path1, 'struc1.xyz'))
    wbo = read_wbo_file(os.path.join(path1, 'wbo1'))
    info = StructuralInformation(nat, xyz, wbo, atom_types, read_hessian(os.path.join(path1, 'hess1')))
    ff = ForceField(nat, os.path.join(path1, 'ff1_new'), readff=False,hessian_calculator=complete_hessian)
    fill_ff(ff, info)
    #fit_ff_to_hessian(Structure(StructurePath('d','d','d','d'),ff, info))
    print(info.fortran_xyz)

    deriv_ana = analy_first_derivative(ff, info)
    deriv_num = num_first_derivative(ff, info)
    div = np.divide(deriv_ana, deriv_num)
    ones = np.ones((len(div)))
    print(div)
    print(ones)
    np.testing.assert_allclose(div, ones, rtol=1e-4)

    


def analy_second_derivative(ff: ForceField, info: StructuralInformation):
    derivatives = []  
    xyz = info.fortran_xyz

    hessian_ff = complete_hessian(xyz, ff)

    for row in ff.bonds.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        derivatives.append(derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        ))
    
    for row in ff.angles.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        l = row.atoms[2]
        derivatives.append(derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, c=row.parameter
        ))
    
    for row in ff.dihedrals.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        l = row.atoms[2]
        m = row.atoms[3]
        derivatives.append(derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, atom4=m, c=row.parameter
        ))
    
    for row in ff.repulsive.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        derivatives.append(repulsive_derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        ))
    
    return derivatives

def num_second_derivative(ff: ForceField, info: StructuralInformation, delta: float = 1e-4):
        xyz = info.fortran_xyz
        derivatives = []
        for i in range(len(ff.bonds)): 
            hess = complete_hessian(xyz, ff)
            res = ff_fit_objective_function(hess, info.hessian, 3*info.nat)

            ff_m = copy.deepcopy(ff)
            ff_m.bonds.iloc[i, ff_m.bonds.columns.get_loc('parameter')] -= 2*delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.bonds.iloc[i, ff_m.bonds.columns.get_loc('parameter')] += 2*delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_m + res_p - 2*res)/(4 * delta**2))

        for i in range(len(ff.angles)): 
            hess = complete_hessian(xyz, ff)
            res = ff_fit_objective_function(hess, info.hessian, 3*info.nat)
            
            ff_m = copy.deepcopy(ff)
            ff_m.angles.iloc[i, ff_m.angles.columns.get_loc('parameter')] -= 2*delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.angles.iloc[i, ff_m.angles.columns.get_loc('parameter')] += 2*delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_m + res_p - 2*res)/(4 * delta**2))

        for i in range(len(ff.dihedrals)): 
            hess = complete_hessian(xyz, ff)
            res = ff_fit_objective_function(hess, info.hessian, 3*info.nat)

            ff_m = copy.deepcopy(ff)
            ff_m.dihedrals.iloc[i, ff_m.dihedrals.columns.get_loc('parameter')] -= 2*delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.dihedrals.iloc[i, ff_m.dihedrals.columns.get_loc('parameter')] += 2*delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_m + res_p - 2*res)/(4 * delta**2))

        for i in range(len(ff.repulsive)): 
            hess = complete_hessian(xyz, ff)
            res = ff_fit_objective_function(hess, info.hessian, 3*info.nat)

            ff_m = copy.deepcopy(ff)
            ff_m.repulsive.iloc[i, ff_m.repulsive.columns.get_loc('parameter')] -= 2*delta
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)

            ff_p = copy.deepcopy(ff)
            ff_p.repulsive.iloc[i, ff_m.repulsive.columns.get_loc('parameter')] += 2*delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)

            derivatives.append((res_m + res_p - 2*res)/(4 * delta**2))
        return derivatives

def test_objfun_second_derivatives():
    path1 = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule')
    nat, _, xyz, atom_types = readin_xyz(os.path.join(path1, 'struc1.xyz'))
    wbo = read_wbo_file(os.path.join(path1, 'wbo1'))
    info = StructuralInformation(nat, xyz, wbo, atom_types, read_hessian(os.path.join(path1, 'hess1')))
    ff = ForceField(nat, os.path.join(path1, 'ff1_new'), readff=False,hessian_calculator=complete_hessian)
    fill_ff(ff, info)
    #fit_ff_to_hessian(Structure(StructurePath('d','d','d','d'),ff, info))
    print(info.fortran_xyz)

    deriv_ana = analy_second_derivative(ff, info)
    deriv_num = num_second_derivative(ff, info)
    div = np.divide(deriv_ana, deriv_num)
    ones = np.ones((len(div)))
    print(div)
    print(ones)
    np.testing.assert_allclose(div, ones, rtol=5e-4)


    



