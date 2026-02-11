"""
Unit tests for WBO analysis module.

Tests compare_wbo_differences and related functions.
"""

import pytest
import numpy as np
from pathlib import Path
from ffits.io.reader import read_xtb_hessian, readin_xyz, read_wbo_file
from ffits.datatype.structure_data import (
    Structure,
    StructurePath,
    ForceField,
    StructuralInformation
)
from ffits.utils.wbo_analysis import (
    compare_wbo_differences,
    get_wbo_matrix_difference,
    identify_affected_atoms,
    print_wbo_comparison,
)

from ffits.setup.structure_preparatation import combine_information_on_both_structures

@pytest.fixture
def reactant_structure():
    """Create a reactant structure for testing using real example files."""
    test_dir = Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
    
    xyz_file = str(test_dir / "struc1.xyz")
    wbo_file = str(test_dir / "wbo1")
    ff_file = str(test_dir / "ff1.csv")
    hessian_file = str(test_dir / "struc1.hess")
    
    nat, _, xyz, atom_types = readin_xyz(xyz_file)
    wbo = read_wbo_file(wbo_file)
    hessian = read_xtb_hessian(hessian_file)
    
    path = StructurePath(
        xyz_filename=xyz_file,
        wbo_filename=wbo_file,
        hessian_filename=hessian_file,
        ff_filename=ff_file,
    )
    ff = ForceField(nat=nat, ff_filename=ff_file, readff=True)
    info = StructuralInformation(
        nat=nat,
        xyz=xyz,
        wbo_dict=wbo,
        atom_types=np.array(atom_types),
        hessian=hessian,
    )
    return Structure(path=path, ff=ff, info=info)


@pytest.fixture
def product_structure():
    """Create a product structure for testing using real example files."""
    test_dir = Path(__file__).parent.parent.parent / "examples" / "small_single_molecule"
    
    xyz_file = str(test_dir / "struc2.xyz")
    wbo_file = str(test_dir / "wbo2")
    ff_file = str(test_dir / "ff2.csv")
    hessian_file = str(test_dir / "struc2.hess")
    nat, _, xyz, atom_types = readin_xyz(xyz_file)
    wbo = read_wbo_file(wbo_file)
    hessian = read_xtb_hessian(hessian_file)
    
    path = StructurePath(
        xyz_filename=xyz_file,
        wbo_filename=wbo_file,
        hessian_filename=hessian_file,
        ff_filename=ff_file,
    )
    ff = ForceField(nat=nat, ff_filename=ff_file, readff=True)
    info = StructuralInformation(
        nat=nat,
        xyz=xyz,
        wbo_dict=wbo,
        atom_types=np.array(atom_types), 
        hessian=hessian,
    )
    return Structure(path=path, ff=ff, info=info)


def test_structrechange(reactant_structure, product_structure):
    combine_information_on_both_structures(reactant_structure, product_structure)
    assert False


class TestCompareWboDifferences:
    """Test compare_wbo_differences function."""
    
    def test_compare_wbo_basic(self, reactant_structure, product_structure):
        """Test basic WBO comparison."""
        result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.1)
        
        assert isinstance(result, dict)
        assert 'changing_bonds' in result
        assert 'disappearing_bonds' in result
        assert 'forming_bonds' in result
        assert 'all_wbo_changes' in result
        assert result['threshold'] == 0.1
    
    def test_changing_bonds_detected(self, reactant_structure, product_structure):
        """Test that changing bonds are correctly identified."""
        result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.05)
        
        # Should have at least some changing bonds or disappearing/forming bonds
        changing_bonds = result['changing_bonds']
        disappearing = result['disappearing_bonds']
        forming = result['forming_bonds']
        
        # At least one category should have changes (for real structures)
        assert len(changing_bonds) > 0 or len(disappearing) > 0 or len(forming) > 0
        
        # Each changing bond should have valid structure
        for bond, change, r_wbo, p_wbo in changing_bonds:
            assert isinstance(bond, tuple)
            assert len(bond) == 2
            assert change >= 0
    
    def test_disappearing_bonds(self, reactant_structure, product_structure):
        """Test detection of disappearing bonds."""
        result = compare_wbo_differences(reactant_structure, product_structure)
        
        # Disappearing bonds should be in reactant but not product
        disappearing = result['disappearing_bonds']
        reactant_wbo = reactant_structure.info.wbo
        product_wbo = product_structure.info.wbo
        
        for bond, wbo in disappearing:
            assert bond in reactant_wbo
            assert bond not in product_wbo
            assert wbo == reactant_wbo[bond]
    
    def test_forming_bonds(self, reactant_structure, product_structure):
        """Test detection of forming bonds."""
        result = compare_wbo_differences(reactant_structure, product_structure)
        
        # Forming bonds should be in product but not reactant
        forming = result['forming_bonds']
        reactant_wbo = reactant_structure.info.wbo
        product_wbo = product_structure.info.wbo
        
        for bond, wbo in forming:
            assert bond not in reactant_wbo
            assert bond in product_wbo
            assert wbo == product_wbo[bond]
    
    def test_threshold_effect(self, reactant_structure, product_structure):
        """Test that threshold correctly filters bonds."""
        result_low = compare_wbo_differences(reactant_structure, product_structure, threshold=0.05)
        result_high = compare_wbo_differences(reactant_structure, product_structure, threshold=0.5)
        
        # Higher threshold should find fewer bonds
        assert len(result_high['changing_bonds']) <= len(result_low['changing_bonds'])
    
    def test_error_on_no_wbo(self, reactant_structure, product_structure):
        """Test error handling when WBO is missing."""
        # Create structure with no WBO
        reactant_structure.info.wbo = {}
        
        with pytest.raises(ValueError, match="Reactant structure has no WBO data"):
            compare_wbo_differences(reactant_structure, product_structure)
    
    def test_error_on_atom_count_mismatch(self, reactant_structure, product_structure):
        """Test error handling for atom count mismatch."""
        # Modify product to have different atom count
        original_nat = product_structure.info.nat
        product_structure.info.nat = original_nat + 1
        
        with pytest.raises(ValueError, match="different atom counts"):
            compare_wbo_differences(reactant_structure, product_structure)
        
        # Restore for other tests
        product_structure.info.nat = original_nat
    
    def test_all_wbo_changes_completeness(self, reactant_structure, product_structure):
        """Test that all_wbo_changes contains all bonds."""
        result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.0)
        
        # Get all unique bonds from both structures
        all_bonds = set(reactant_structure.info.wbo.keys()) | set(product_structure.info.wbo.keys())
        
        # All bonds should be in all_wbo_changes
        for bond in all_bonds:
            assert bond in result['all_wbo_changes']
            change_data = result['all_wbo_changes'][bond]
            assert 'change' in change_data
            assert 'reactant_wbo' in change_data
            assert 'product_wbo' in change_data
        
    # def test_expected_changes(self, reactant_structure, product_structure):
    #     """Test that specific expected changes are detected."""
    #     result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.05)
        
    #     # For the example structures, we expect certain bonds to change significantly
    #     expected_changing_bonds = [
    #         (1, 2),  # Example bond that should change
    #         (2, 3),  # Another example bond
    #     ]
        
    #     changing_bonds = [bond for bond, _, _, _ in result['changing_bonds']]
    #     print(changing_bonds)
    #     for expected in expected_changing_bonds:
    #         assert expected in changing_bonds

    #     assert False 


class TestGetWboMatrixDifference:
    """Test get_wbo_matrix_difference function."""
    
    def test_matrix_shape(self, reactant_structure, product_structure):
        """Test that returned matrix has correct shape."""
        diff_matrix = get_wbo_matrix_difference(reactant_structure, product_structure)
        
        nat = reactant_structure.info.nat
        assert diff_matrix.shape == (nat, nat)
    
    def test_matrix_symmetry(self, reactant_structure, product_structure):
        """Test that difference matrix is symmetric."""
        diff_matrix = get_wbo_matrix_difference(reactant_structure, product_structure)
        
        np.testing.assert_array_almost_equal(diff_matrix, diff_matrix.T)
    
    def test_matrix_values(self, reactant_structure, product_structure):
        """Test that matrix values are correct."""
        diff_matrix = get_wbo_matrix_difference(reactant_structure, product_structure)
        
        # Get a bond from the structures to verify
        reactant_wbo = reactant_structure.info.wbo
        product_wbo = product_structure.info.wbo
        
        # Check a specific bond that exists in both
        for bond in reactant_wbo.keys():
            if bond in product_wbo:
                i, j = bond[0] - 1, bond[1] - 1  # Convert to 0-based
                expected_change = product_wbo[bond] - reactant_wbo[bond]
                np.testing.assert_almost_equal(diff_matrix[i, j], expected_change)
                break  # Just test one bond
    
    def test_error_on_atom_count_mismatch(self, reactant_structure, product_structure):
        """Test error on atom count mismatch."""
        product_structure.info.nat = product_structure.info.nat + 1
        
        with pytest.raises(ValueError, match="same number of atoms"):
            get_wbo_matrix_difference(reactant_structure, product_structure)


class TestIdentifyAffectedAtoms:
    """Test identify_affected_atoms function."""
    
    def test_affected_atoms_returned(self, reactant_structure, product_structure):
        """Test that affected atoms are returned."""
        result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.1)
        affected = identify_affected_atoms(result, threshold=0.1)
        
        assert isinstance(affected, list)
        assert len(affected) > 0
    
    def test_affected_atoms_sorted(self, reactant_structure, product_structure):
        """Test that affected atoms are sorted by number of impacts."""
        result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.1)
        affected = identify_affected_atoms(result, threshold=0.1)
        
        # Check that list is sorted
        for i in range(len(affected) - 1):
            assert len(affected[i][1]) >= len(affected[i + 1][1])
    
    def test_affected_atoms_content(self, reactant_structure, product_structure):
        """Test that affected atoms contain correct bonds."""
        result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.1)
        affected = identify_affected_atoms(result, threshold=0.1)
        
        for atom, bonds in affected:
            assert isinstance(atom, (int, np.integer))
            assert isinstance(bonds, list)
            for bond in bonds:
                # Atom should be in bond
                assert atom in bond


class TestPrintWboComparison:
    """Test print_wbo_comparison function."""
    
    def test_print_format(self, reactant_structure, product_structure):
        """Test that output is formatted correctly."""
        result = compare_wbo_differences(reactant_structure, product_structure)
        output = print_wbo_comparison(result)
        
        assert isinstance(output, str)
        assert "WBO COMPARISON SUMMARY" in output
        assert "=" * 70 in output
    
    def test_print_contains_data(self, reactant_structure, product_structure):
        """Test that output contains bond data."""
        result = compare_wbo_differences(reactant_structure, product_structure, threshold=0.1)
        output = print_wbo_comparison(result)
        
        # Should contain bond numbers
        assert "Bond" in output
    
    def test_print_verbose_flag(self, reactant_structure, product_structure):
        """Test that verbose flag works."""
        result = compare_wbo_differences(reactant_structure, product_structure)
        output_normal = print_wbo_comparison(result, verbose=False)
        output_verbose = print_wbo_comparison(result, verbose=True)
        
        # Both should be strings (verbose flag doesn't change much for this simple implementation)
        assert isinstance(output_normal, str)
        assert isinstance(output_verbose, str)
