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
    xyz = info.fortran_xyz

    hessian_ff = complete_hessian(xyz, ff)
    
    # Calculate total number of parameters
    n_params = len(ff.bonds) + len(ff.angles) + len(ff.dihedrals) + len(ff.repulsive)
    
    # Initialize Hessian matrix
    hessian = np.zeros((n_params, n_params))
    
    param_idx = 0
    
    # Fill diagonal for bonds
    for row in ff.bonds.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        hessian[param_idx, param_idx] = derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        )
        param_idx += 1
    
    # Fill diagonal for angles
    for row in ff.angles.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        l = row.atoms[2]
        hessian[param_idx, param_idx] = derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, c=row.parameter
        )
        param_idx += 1
    
    # Fill diagonal for dihedrals
    for row in ff.dihedrals.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        l = row.atoms[2]
        m = row.atoms[3]
        hessian[param_idx, param_idx] = derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, atom3=l, atom4=m, c=row.parameter
        )
        param_idx += 1
    
    # Fill diagonal for repulsive
    for row in ff.repulsive.itertuples(): 
        i = row.atoms[0]
        j = row.atoms[1]
        hessian[param_idx, param_idx] = repulsive_derivative_c_second_atomwise(
            info.nat, xyz,
            row.reference_value,
            hessian_ff, info.hessian,
            atom1=i, atom2=j, c=row.parameter
        )
        param_idx += 1
    
    return hessian

def num_second_derivative(ff: ForceField, info: StructuralInformation, delta: float = 1e-4):
        xyz = info.fortran_xyz
        
        # Calculate total number of parameters
        n_params = len(ff.bonds) + len(ff.angles) + len(ff.dihedrals) + len(ff.repulsive)
        
        # Initialize Hessian matrix
        hessian = np.zeros((n_params, n_params))
        
        # Helper function to get parameter lists in order
        def get_all_params():
            params = []
            for idx, row in enumerate(ff.bonds.itertuples()):
                params.append(('bond', idx, row))
            for idx, row in enumerate(ff.angles.itertuples()):
                params.append(('angle', idx, row))
            for idx, row in enumerate(ff.dihedrals.itertuples()):
                params.append(('dihedral', idx, row))
            for idx, row in enumerate(ff.repulsive.itertuples()):
                params.append(('repulsive', idx, row))
            return params
        
        params = get_all_params()
        
        # Helper function to perturb a parameter
        def perturb_ff(ff_copy, param_type, param_idx, delta_val):
            if param_type == 'bond':
                ff_copy.bonds.iloc[param_idx, ff_copy.bonds.columns.get_loc('parameter')] += delta_val
            elif param_type == 'angle':
                ff_copy.angles.iloc[param_idx, ff_copy.angles.columns.get_loc('parameter')] += delta_val
            elif param_type == 'dihedral':
                ff_copy.dihedrals.iloc[param_idx, ff_copy.dihedrals.columns.get_loc('parameter')] += delta_val
            elif param_type == 'repulsive':
                ff_copy.repulsive.iloc[param_idx, ff_copy.repulsive.columns.get_loc('parameter')] += delta_val
        
        # Calculate diagonal and off-diagonal elements
        for i in range(n_params):
            for j in range(i, n_params):
                if i == j:
                    # Diagonal: second derivative with respect to same parameter
                    param_type_i, idx_i, _ = params[i]
                    
                    hess_center = complete_hessian(xyz, ff)
                    energy_center = ff_fit_objective_function(hess_center, info.hessian, 3*info.nat)
                    
                    ff_m = copy.deepcopy(ff)
                    perturb_ff(ff_m, param_type_i, idx_i, -2*delta)
                    hess_m = complete_hessian(xyz, ff_m)
                    energy_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)
                    
                    ff_p = copy.deepcopy(ff)
                    perturb_ff(ff_p, param_type_i, idx_i, 2*delta)
                    hess_p = complete_hessian(xyz, ff_p)
                    energy_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)
                    
                    hessian[i, j] = (energy_m + energy_p - 2*energy_center) / (4.0 * delta**2)
                
                else:
                    # Off-diagonal: mixed partial derivative
                    param_type_i, idx_i, _ = params[i]
                    param_type_j, idx_j, _ = params[j]
                    
                    # Energy at (0, 0)
                    hess_00 = complete_hessian(xyz, ff)
                    energy_00 = ff_fit_objective_function(hess_00, info.hessian, 3*info.nat)
                    
                    # Energy at (+delta_i, +delta_j)
                    ff_pp = copy.deepcopy(ff)
                    perturb_ff(ff_pp, param_type_i, idx_i, delta)
                    perturb_ff(ff_pp, param_type_j, idx_j, delta)
                    hess_pp = complete_hessian(xyz, ff_pp)
                    energy_pp = ff_fit_objective_function(hess_pp, info.hessian, 3*info.nat)
                    
                    # Energy at (+delta_i, -delta_j)
                    ff_pm = copy.deepcopy(ff)
                    perturb_ff(ff_pm, param_type_i, idx_i, delta)
                    perturb_ff(ff_pm, param_type_j, idx_j, -delta)
                    hess_pm = complete_hessian(xyz, ff_pm)
                    energy_pm = ff_fit_objective_function(hess_pm, info.hessian, 3*info.nat)
                    
                    # Energy at (-delta_i, +delta_j)
                    ff_mp = copy.deepcopy(ff)
                    perturb_ff(ff_mp, param_type_i, idx_i, -delta)
                    perturb_ff(ff_mp, param_type_j, idx_j, delta)
                    hess_mp = complete_hessian(xyz, ff_mp)
                    energy_mp = ff_fit_objective_function(hess_mp, info.hessian, 3*info.nat)
                    
                    # Energy at (-delta_i, -delta_j)
                    ff_mm = copy.deepcopy(ff)
                    perturb_ff(ff_mm, param_type_i, idx_i, -delta)
                    perturb_ff(ff_mm, param_type_j, idx_j, -delta)
                    hess_mm = complete_hessian(xyz, ff_mm)
                    energy_mm = ff_fit_objective_function(hess_mm, info.hessian, 3*info.nat)
                    
                    # Mixed partial derivative
                    mixed_deriv = (energy_pp - energy_pm - energy_mp + energy_mm) / (4.0 * delta**2)
                    hessian[i, j] = mixed_deriv
                    hessian[j, i] = mixed_deriv  # Symmetric matrix
        
        return hessian

def test_objfun_second_derivatives():
    path1 = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule')
    nat, _, xyz, atom_types = readin_xyz(os.path.join(path1, 'struc1.xyz'))
    wbo = read_wbo_file(os.path.join(path1, 'wbo1'))
    info = StructuralInformation(nat, xyz, wbo, atom_types, read_hessian(os.path.join(path1, 'hess1')))
    ff = ForceField(nat, os.path.join(path1, 'ff1_new'), readff=False,hessian_calculator=complete_hessian)
    fill_ff(ff, info)
    print(info.fortran_xyz)

    hess_ana = analy_second_derivative(ff, info)
    hess_num = num_second_derivative(ff, info)
    
    # Extract diagonal elements for comparison
    diag_ana = np.diag(hess_ana)
    diag_num = np.diag(hess_num)
    
    # Check if only diagonal are correct
    div = np.divide(diag_ana, diag_num)
    ones = np.ones((len(div)))
    np.testing.assert_allclose(div, ones, rtol=5e-4)

    # Check full Hessian
    np.testing.assert_allclose(hess_ana, hess_num, rtol=5e-4)
    for i in range(hess_ana.shape[0]):
        for j in range(hess_ana.shape[1]):
            # if i != j:
            # if abs(hess_ana[i,j]) < 1e-8 and abs(hess_num[i,j]) < 1e-12:
            #     continue
            # ratio = hess_ana[i,j] / hess_num[i,j]
            print(i, j, hess_ana[i,j] , hess_num[i,j])
    # print("\nAnalytical Hessian (diagonal):")
    # print(diag_ana)
    # print("\nNumerical Hessian (diagonal):")
    # print(diag_num)
    
    # print("\nNumerical Hessian (full matrix):")
    # print(hess_num)
    # print("\nAnalytical Hessian (full matrix - currently diagonal only):")
    # print(hess_ana)
    assert False


    



