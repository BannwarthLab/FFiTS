import os
import pandas as pd
import shutil
import numpy as np
from src.ts_guess.mix_ff import combine_ff_atoms, remove_bonds_from_repulsive, mix_parameters, mix_reference_values
from src.datatype.structure_data import ForceField, StructuralInformation, StructurePath, Structure
from src.io.reader import readin_xyz, read_wbo_file, read_hessian
from src.ts_guess.define_starting_parameters import fill_ff
from src.ts_guess.parameterize_ff import fit_ff_to_hessian

from src.ts_guess.guess import get_ts_guess
from src.forcefield.python_interface.ff_energy import energy_ff, complete_gradient, complete_hessian

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
    print(atom_types)
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
    tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)

    assert _all_values_in_either(tsff.bonds['atoms'], ff1.bonds['atoms'], ff2.bonds['atoms'])
    assert _all_values_in_either(tsff.angles['atoms'], ff1.angles['atoms'], ff2.angles['atoms'])
    assert _all_values_in_either(tsff.dihedrals['atoms'], ff1.dihedrals['atoms'], ff2.dihedrals['atoms'])
        
def test_remove_bonds_from_repulsive():
    """Tests whether the repulsive atoms are mixed correctly."""
    ff1, _, ff2, _ = _define_ff_examples()
    tsff = ForceField(7, 'temp', readff=False)
    tsff.repulsive = remove_bonds_from_repulsive(combine_ff_atoms(ff1.repulsive, ff2.repulsive), combine_ff_atoms(ff1.bonds, ff2.bonds))
    
    assert _all_values_in_either(tsff.repulsive['atoms'], ff1.repulsive['atoms'], ff2.repulsive['atoms'])
    for _, val in tsff.repulsive['atoms'].items():
        in_ff1 = any(val == b for b in ff1.bonds['atoms'])
        in_ff2 = any(val == b for b in ff2.bonds['atoms'])
        if in_ff1 or in_ff2:
            raise AssertionError(f"{val} is present in another list")
    
def test_mix_parameters():
    ff1, _, ff2, _ = _define_ff_examples()
    tsff = ForceField(7, 'temp', readff=False)
    print(ff1.bonds)
    print(ff2.bonds)
    tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
    tsff.repulsive = remove_bonds_from_repulsive(combine_ff_atoms(ff1.repulsive, ff2.repulsive), combine_ff_atoms(ff1.bonds, ff2.bonds))

    mix_parameters(tsff.bonds, ff1.bonds, ff2.bonds)
    mix_parameters(tsff.angles, ff1.angles, ff2.angles)
    mix_parameters(tsff.dihedrals, ff1.dihedrals, ff2.dihedrals)
    mix_parameters(tsff.repulsive, ff1.repulsive, ff2.repulsive)

    
    assert not all(tsff.bonds['parameter'].isnull())
    assert not all(np.isnan(tsff.bonds['parameter']))
    # compared values and they seemed to be averaged as expected
    
def test_mix_reference_values():
    ff1, info1, ff2, info2 = _define_ff_examples()
    tsff = ForceField(7, 'temp', readff=False)
    print(ff1.bonds)
    print(ff2.bonds)
    tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
    tsff.repulsive = remove_bonds_from_repulsive(combine_ff_atoms(ff1.repulsive, ff2.repulsive), combine_ff_atoms(ff1.bonds, ff2.bonds))

    mix_reference_values(tsff, ff1, ff2, info1, info2)
    print('TSFF -------------')
    print(tsff.bonds)
    print(tsff.angles)
    print(tsff.dihedrals)
    print(tsff.repulsive)
    print('FF 1. -------------')
    print(ff1.bonds)
    print(ff1.angles)
    print(ff1.dihedrals)
    print(ff1.repulsive)
    print('FF 2 -------------')
    print(ff2.bonds)
    print(ff2.angles)
    print(ff2.dihedrals)
    print(ff2.repulsive)
    assert not all(np.isnan(tsff.bonds['parameter']))
    assert not all(np.isnan(tsff.angles['parameter']))
    assert not all(np.isnan(tsff.dihedrals['parameter']))
    assert not all(np.isnan(tsff.repulsive['parameter']))
    assert all(tsff.bonds.apply(lambda row: row.reference_value <= info1.vander_matrix[row.atoms[0], row.atoms[1]], axis=1)) # checks whether all bondlengths are shorter than the vdw distance
    assert all(tsff.angles.apply(lambda row: row.reference_value <= np.pi, axis=1))
    assert all(tsff.angles.apply(lambda row: np.abs(row.reference_value) <= np.pi, axis=1))
    assert all(tsff.repulsive.apply(lambda row: row.reference_value <= info1.vander_matrix[row.atoms[0], row.atoms[1]], axis=1))

def test_get_ts_guess(): # TODO VERY unfinished and sloppy 
    cwd = os.getcwd()
    temp_wd = os.path.join(cwd, '_manual_test/small_single_molecule')
    
    data_dir = os.path.join(cwd, 'tests/examples/small_single_molecule')
    shutil.copy(os.path.join(data_dir, 'struc1.xyz'), temp_wd)
    shutil.copy(os.path.join(data_dir, 'struc2.xyz'), temp_wd)
    shutil.copy(os.path.join(data_dir, 'wbo1'), temp_wd)
    shutil.copy(os.path.join(data_dir, 'wbo2'), temp_wd)
    
    path1 = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule')
    nat, _, xyz, atom_types = readin_xyz(os.path.join(path1, 'struc1.xyz'))
    wbo = read_wbo_file(os.path.join(path1, 'wbo1'))
    path2hess = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/hess1')
    hessian = read_hessian(path2hess)
    info1 = StructuralInformation(nat, xyz, wbo, atom_types, hessian=hessian)
    ff1 = ForceField(nat, os.path.join(path1, 'ff1_new'), readff=False)

    ff1.energy_calculator = energy_ff
    ff1.gradient_calculator = complete_gradient
    ff1.hessian_calculator = complete_hessian
    fill_ff(ff1, info1)
    result = fit_ff_to_hessian(ff1, info1, constant_repulsion=False, stepsize=0.05, threshold=0.001)

    ### ff2
    path2 = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule')
    nat, _, xyz, atom_types = readin_xyz(os.path.join(path2, 'struc2.xyz'))
    print(atom_types)
    wbo = read_wbo_file(os.path.join(path2, 'wbo2'))
    path2hess = os.path.join(os.getcwd(), 'tests/examples/small_single_molecule/hess2')
    hessian = read_hessian(path2hess)
    info2 = StructuralInformation(nat, xyz, wbo, atom_types, hessian=hessian)
    ff2 = ForceField(nat, os.path.join(path2, 'ff2_new'), readff=False)
    ff2.energy_calculator = energy_ff
    ff2.gradient_calculator = complete_gradient
    ff2.hessian_calculator = complete_hessian
    fill_ff(ff2, info2)
    result = fit_ff_to_hessian(ff2, info2, constant_repulsion=False, stepsize=0.05, threshold=0.001)

    os.chdir(temp_wd)

    struc1 = Structure(StructurePath('t','t','t','t'), ff1, info1)
    struc2 = Structure(StructurePath('t','t','t','t'), ff2, info2)
    
    tsff = get_ts_guess(struc1, struc2)
    
    os.chdir(cwd)


    # compared values and they seemed to be averaged as expected
