import os
import pandas as pd
import numpy as np
from collections.abc import Callable
from src.ts_guess.mix_ff import create_tsff
from src.datatype.structure_data import ForceField, StructuralInformation, Structure
from src.external.molbar_optimizer import anc_optimizer, scipy_optimizer
from src.forcefield.python_interface.ff_energy import energy_ff, complete_gradient, complete_hessian


def get_ts_guess(struc1: Structure, struc2: Structure, dir: str = '/home/dbabushkina/1_ts_search2024/ffits/_manual_test/small_single_molecule', optimizer: Callable = anc_optimizer):
    tsff = create_tsff(struc1.ff, struc1.info, struc2.ff, struc2.info)
    tsff.energy_calculator = energy_ff
    tsff.gradient_calculator = complete_gradient
    tsff.hessian_calculator = complete_hessian
    cwd = os.getcwd()
    os.chdir(dir)
    converged, energy, final_geom, _, _, _ = optimizer(struc1.info.xyz, tsff, struc1.info.atom_types, max_micro_steps=5)

    os.chdir(cwd)
    print(converged, energy)
    return tsff
