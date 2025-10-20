import os
import numpy as np
import copy
import pandas as pd
import pytest
from src.datatype.structure_data import ForceField, StructuralInformation
from src.forcefield.fit2hessian.parameterize_ff import update_bond
from src.interface.reader import read_hessian
from src.forcefield.fortran_energy.ff_energy import complete_hessian
from src.forcefield.setup.define_starting_parameters import fill_ff

XYZ = np.array([
    [-2.33287094, 3.31176687, 0.20110100],
    [-0.91630217, 2.85867268, -0.04327585],
    [0.06256276, 3.55862185, 0.08938727],
    [-2.92591176, 3.18375556, -0.71288068],
    [-2.79725946, 2.67965821, 0.96793335],
    [-0.81484498, 1.79716556, -0.36699673],
    [-2.35087245, 4.35655928, 0.51563164]
])
WBO = {
    (1, 2): 1.02668632226515, 
    (2, 3): 1.92755303185758, 
    (1, 4): 0.955689824153634,  
    (1, 5): 0.955863695522291,  
    (2, 6): 0.933812077856736,  
    (1, 7): 0.982636418257069  
}
ATOM_TYPES = ['C', 'C', 'O', 'H', 'H', 'H', 'H']

NAT = 7


def test_fit_ff_to_hessian():
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1_new'
    )    
    path2hess = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/hess1')
    hessian = read_hessian(path2hess)
    ff = ForceField(7, path, readff=True) 
    result = fit_ff_to_hessian(ff)
    print(ff)
    # assert result['final_rmsd'] <= 0.1

def test_update_bond():
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1_newi'
    )
    path2hess = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/hess1')
    hessian = read_hessian(path2hess)

    info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES, hessian=hessian)
    ff = ForceField(7, path, readff=False, hessian_calculator=complete_hessian) 
    fill_ff(ff, info)
    hessian_ff = ff.get_hessian(info.fortran_xyz)
    
    old_bonds = copy.deepcopy(ff.bonds)
    new_values = []
    for row in ff.bonds.itertuples(): 
        new_param = update_bond(row, info, hessian_ff, 1)
        new_values.append((row.Index, new_param))
    for idx, val in new_values:
        ff.bonds.at[idx, 'parameter'] = val


    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(ff.bonds['parameter'], old_bonds['parameter']) 
    pd.testing.assert_series_equal(ff.bonds['reference_value'], old_bonds['reference_value'])
    pd.testing.assert_series_equal(ff.bonds['atoms'], old_bonds['atoms'])
    
