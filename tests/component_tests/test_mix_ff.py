import os
from src.forcefield.transition_state.mix_ff import combine_ff_terms
from src.datatype.structure_data import ForceField, StructuralInformation
from src.interface.reader import readin_xyz, read_wbo_file
from src.forcefield.setup.define_starting_parameters import fill_ff

def define_ff_examples():
    """Hard-coded example ff information"""
    ### ff1
    path1 = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule')
    nat, _, xyz, atom_types = readin_xyz(os.path.join(path1, 'struc1.xyz'))
    wbo = read_wbo_file(os.path.join(path1, 'wbo1'))
    info1 = StructuralInformation(nat, xyz, wbo, atom_types)
    ff1 = ForceField(nat, os.path.join(path1, 'ff1_new'), readff=False)
    fill_ff(ff1, info1)

    ### ff2
    path2 = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule')
    nat, _, xyz, atom_types = readin_xyz(os.path.join(path2, 'struc2.xyz'))
    wbo = read_wbo_file(os.path.join(path2, 'wbo2'))
    info2 = StructuralInformation(nat, xyz, wbo, atom_types)
    ff2 = ForceField(nat, os.path.join(path2, 'ff2_new'), readff=False)
    fill_ff(ff2, info2)

    return ff1, info1, ff2, info2




def test_combine_ff_terms():
    ff1, info1, ff2, info2 = define_ff_examples()
    print(ff1.bonds)
    print(ff2.bonds)
    print(combine_ff_terms(ff1.bonds, ff2.bonds))
    assert False