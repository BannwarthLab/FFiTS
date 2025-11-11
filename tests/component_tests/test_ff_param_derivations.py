import numpy as np
import os
from src.datatype.structure_data import ForceField, StructuralInformation, Structure, StructurePath
from src.ts_guess.parameterize_ff import derivative_c_first_atomwise, derivative_c_second_atomwise, ff_fit_objective_function, repulsive_derivative_c_first_atomwise, fit_ff_to_hessian
from src.forcefield.python_interface.ff_energy import complete_hessian
from src.io.reader import readin_xyz, read_wbo_file, read_hessian
from src.ts_guess.define_starting_parameters import fill_ff
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
    
    # for row in ff.angles.itertuples(): 
    #     i = row.atoms[0]
    #     j = row.atoms[1]
    #     j = row.atoms[2]
    #     derivatives.append(derivative_c_first_atomwise(
    #         info.nat, xyz,
    #         row.reference_value,
    #         hessian_ff, info.hessian,
    #         atom1=i, atom2=j, ggc=row.parameter
    #     ))
    # for row in ff.dihedrals.itertuples(): 
    #     i = row.atoms[0]
    #     j = row.atoms[1]
    #     j = row.atoms[2]
    #     j = row.atoms[3]
    #     derivatives.append(derivative_c_first_atomwise(
    #         info.nat, xyz,
    #         row.reference_value,
    #         hessian_ff, info.hessian,
    #         atom1=i, atom2=j, ggc=row.parameter
    #     ))
    # for row in ff.repulsive.itertuples(): 
    #     i = row.atoms[0]
    #     j = row.atoms[1]
    #     derivatives.append(repulsive_derivative_c_first_atomwise(
    #         info.nat, xyz,
    #         row.reference_value,
    #         hessian_ff, info.hessian,
    #         atom1=i, atom2=j, c=row.parameter
    #     ))
    return derivatives

def num_first_derivative(ff: ForceField, info: StructuralInformation, delta: float = 1e-2):
        xyz = info.fortran_xyz
        derivatives = []
        for i in range(len(ff.bonds)): 
            ff_m = copy.deepcopy(ff)
            ff_m.bonds.iloc[i, ff_m.bonds.columns.get_loc('parameter')] -= delta
            print(ff_m.bonds)
            hess_m = complete_hessian(xyz, ff_m)
            res_m = ff_fit_objective_function(hess_m, info.hessian, 3*info.nat)
            print(res_m)

            ff_p = copy.deepcopy(ff)
            ff_p.bonds.iloc[i, ff_m.bonds.columns.get_loc('parameter')] += delta
            hess_p = complete_hessian(xyz, ff_p)
            res_p = ff_fit_objective_function(hess_p, info.hessian, 3*info.nat)
            print(res_p)

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


    deriv_ana = analy_first_derivative(ff, info)
    deriv_num = num_first_derivative(ff, info)
    print(deriv_ana)
    print(deriv_num)
    print(np.divide(deriv_ana, deriv_num))
    assert False

    



