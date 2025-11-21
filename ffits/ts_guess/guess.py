import os
import pandas as pd
import numpy as np
import sys
from collections.abc import Callable
from ffits.ts_guess.mix_ff import create_tsff
from ffits.datatype.structure_data import  Structure
from ffits.datatype.calculation_data import CalculationData, TSCalculationOptions
from ffits.external.molbar_optimizer import anc_optimizer, scipy_optimizer, failed_anc_opt, write_last_valid_xyz
from ffits.forcefield.python_interface.ff_energy import energy_ff, complete_gradient,complete_hessian
from ffits.io.print.details import print_optimization_end, print_optimization_start


def get_ts_guess(struc1: Structure, struc2: Structure, calcoptions: TSCalculationOptions, optimizer: Callable = anc_optimizer):
    trajectory_filename: str = 'trajectory.xyz'
    final_geometry_filename: str = 'optimized.xyz'
    print_optimization_start()

    #### ------- Create TS Force Field by mixing reactant and product FFs ------- ####
    tsff = create_tsff(ff1 = struc1.ff, 
                       info1 = struc1.info, 
                       ff2 = struc2.ff, 
                       info2 = struc2.info, 
                       fact1 = calcoptions.factor_reactant, 
                       fact2 = calcoptions.factor_product)
    tsff.energy_calculator = energy_ff
    tsff.gradient_calculator = complete_gradient
    tsff.hessian_calculator = complete_hessian
    
    # TODO add time and also return it

    #### ------- TS Optimization with generated FF as potential ------- ####
    opt_stdout_filename = 'ts_optimization.out'
    orig_stdout = sys.stdout
    f = open(opt_stdout_filename, 'w')
    sys.stdout = f
    if optimizer == anc_optimizer:
        converged, energy, final_geom, steps, time, message = optimizer(
            struc1.info.xyz,
            tsff,
            struc1.info.atom_types,
            g_tol=calcoptions.molbar_optimizer_g_tol,
            x_tol=calcoptions.molbar_optimizer_x_tol,
            e_tol=calcoptions.molbar_optimizer_e_tol,
            trajectory_filename=trajectory_filename,
            final_geometry_filename=final_geometry_filename,
            max_micro_steps=5
            )
        sys.stdout = orig_stdout
        f.close()

    elif optimizer == scipy_optimizer:
        result = scipy_optimizer(struc1.info.xyz, tsff, struc1)
        sys.stdout = orig_stdout
        f.close()
        return result
    else:
        raise Exception(f'Optimizer {optimizer} not recognized.')

    if failed_anc_opt(opt_stdout_filename): 
        print(f'[WARNING] The last valid structure of the optimization trajectory is written to {final_geometry_filename}.')
        write_last_valid_xyz()
        
    
    print_optimization_end(converged, energy, final_geom, steps, time, message)

    return converged, energy, final_geom 
