
import numpy as np
import os
from src.interface.molbar_optimizer import anc_optimizer
from src.forcefield.fortran_energy.ff_energy import energy_ff, complete_gradient, complete_hessian
from src.datatype.structure_data import ForceField, StructuralInformation
from src.interface.reader import readin_xyz

NAT = 7

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

def test_anc_optimizer():
    cwd = os.getcwd()
    temp_wd = os.path.join(cwd, '_manual_test/small_single_molecule')
    os.chdir(temp_wd)

    struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    ff = ForceField(7, 'ff1_new', readff=True, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian) 
    _, _, xyz_start, _ = readin_xyz('struc2.xyz') 

    converged, energy, _, _, _, _ = anc_optimizer(xyz_start, ff, struc)

    assert converged 
    assert energy <= 0.1

    os.chdir(cwd)
    assert False

