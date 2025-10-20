import os
from src.datatype.structure_data import ForceField


def test_fit_ff_to_hessian():
    path = os.path.join(os.getcwd(), 
        'tests/examples/small_single_molecule',
        'ff1_new'
    )
    ff = ForceField(7, path, readff=True) 