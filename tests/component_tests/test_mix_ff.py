import os
import pandas as pd
import numpy as np
from src.forcefield.transition_state.mix_ff import combine_ff_terms, remove_bonds_from_repulsive, mix_c
from src.datatype.structure_data import ForceField, StructuralInformation
from src.interface.reader import readin_xyz, read_wbo_file
from src.forcefield.setup.define_starting_parameters import fill_ff

def _define_ff_examples():
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

def _all_values_in_either(tsff_ref: pd.Series, ff1_ref: pd.Series, ff2_ref: pd.Series):
    combined_values = pd.concat([ff1_ref, ff2_ref]).unique()
    missing = tsff_ref[~tsff_ref.isin(combined_values)]
    return missing.empty # true if all values present in either ff1 or ff2

def test_combine_ff_terms():
    """Tests whether the bond, angle and dihedral atoms are mixed correctly."""
    ff1, _, ff2, _ = _define_ff_examples()
    tsff = ForceField(7, 'temp', readff=False)
    tsff.bonds = combine_ff_terms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_terms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_terms(ff1.dihedrals, ff2.dihedrals)

    assert _all_values_in_either(tsff.bonds['atoms'], ff1.bonds['atoms'], ff2.bonds['atoms'])
    assert _all_values_in_either(tsff.angles['atoms'], ff1.angles['atoms'], ff2.angles['atoms'])
    assert _all_values_in_either(tsff.dihedrals['atoms'], ff1.dihedrals['atoms'], ff2.dihedrals['atoms'])
        
def test_remove_bonds_from_repulsive():
    """Tests whether the repulsive atoms are mixed correctly."""
    ff1, _, ff2, _ = _define_ff_examples()
    tsff = ForceField(7, 'temp', readff=False)
    tsff.repulsive = remove_bonds_from_repulsive(combine_ff_terms(ff1.repulsive, ff2.repulsive), combine_ff_terms(ff1.bonds, ff2.bonds))
    
    assert _all_values_in_either(tsff.repulsive['atoms'], ff1.repulsive['atoms'], ff2.repulsive['atoms'])
    for _, val in tsff.repulsive['atoms'].items():
        in_ff1 = any(val == b for b in ff1.bonds['atoms'])
        in_ff2 = any(val == b for b in ff2.bonds['atoms'])
        if in_ff1 or in_ff2:
            raise AssertionError(f"{val} is present in another list")
    
def test_mix_c():
    ff1, _, ff2, _ = _define_ff_examples()
    tsff = ForceField(7, 'temp', readff=False)
    print(ff1.bonds)
    print(ff2.bonds)
    tsff.bonds = combine_ff_terms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_terms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_terms(ff1.dihedrals, ff2.dihedrals)
    tsff.repulsive = remove_bonds_from_repulsive(combine_ff_terms(ff1.repulsive, ff2.repulsive), combine_ff_terms(ff1.bonds, ff2.bonds))

    mix_c(tsff.bonds, ff1.bonds, ff2.bonds)
    mix_c(tsff.angles, ff1.angles, ff2.angles)
    mix_c(tsff.dihedrals, ff1.dihedrals, ff2.dihedrals)
    mix_c(tsff.repulsive, ff1.repulsive, ff2.repulsive)

    
    assert not all(tsff.bonds['parameter'].isnull())
    assert not all(np.isnan(tsff.bonds['parameter']))
    # compared values and they seemed to be averaged as expected
