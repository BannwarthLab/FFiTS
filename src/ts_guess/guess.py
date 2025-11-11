import os
import pandas as pd
import numpy as np
import sys
from collections.abc import Callable
from src.ts_guess.mix_ff import create_tsff
from src.datatype.structure_data import ForceField, StructuralInformation, Structure
from src.external.molbar_optimizer import anc_optimizer, scipy_optimizer, failed_anc_opt, write_last_valid_xyz
from src.forcefield.python_interface.ff_energy import energy_ff, complete_gradient,complete_hessian
from src.io.print.details import print_optimization_end, print_optimization_start


def get_ts_guess(struc1: Structure, struc2: Structure, optimizer: Callable = anc_optimizer):
    trajectory_filename: str = 'trajectory.xyz'
    final_geometry_filename: str = 'optimized.xyz'
    print_optimization_start()
    tsff = create_tsff(struc1.ff, struc1.info, struc2.ff, struc2.info)
    tsff.energy_calculator = energy_ff
    tsff.gradient_calculator = complete_gradient
    tsff.hessian_calculator = complete_hessian
    
    # TODO add time and also return it

    opt_stdout_filename = 'ts_optimization.out'
    orig_stdout = sys.stdout
    f = open(opt_stdout_filename, 'w')
    sys.stdout = f
    converged, energy, final_geom, steps, time, message = optimizer(struc1.info.xyz, tsff, struc1.info.atom_types, max_micro_steps=5)

    sys.stdout = orig_stdout
    f.close()
    if failed_anc_opt(opt_stdout_filename): 
        print(f'[WARNING] The last valid structure of the optimization trajectory is written to {final_geometry_filename}.')
        write_last_valid_xyz()
        
    
    print_optimization_end(converged, energy, final_geom, steps, time, message)

    return converged, energy, final_geom 
