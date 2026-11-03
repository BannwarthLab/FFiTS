"""
Unit tests for WBO analysis module.

Tests compare_wbo_differences and related functions.
"""

import pytest
import numpy as np
from ffits.datatype.structure_data import StructuralInformation
from ffits.utils.wbo_analysis import (
    compare_wbo_differences,
    get_wbo_matrix_difference,
)
from tests.test_utils import reactant_structure_main, product_structure_main


@pytest.fixture
def reactant_structure():
    return reactant_structure_main()


@pytest.fixture
def product_structure():
    return product_structure_main()


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
                i, j = bond
                expected_change = product_wbo[bond] - reactant_wbo[bond]
                np.testing.assert_almost_equal(diff_matrix[i, j], expected_change)
                break  # Just test one bond

    def test_error_on_atom_count_mismatch(self, reactant_structure, product_structure):
        """Test error on atom count mismatch."""
        product_structure.info.nat = product_structure.info.nat + 1

        with pytest.raises(ValueError, match="same number of atoms"):
            get_wbo_matrix_difference(reactant_structure, product_structure)


class TestCompareWboDifferences:
    """Test compare_wbo_differences function."""

    def test_changing_bonds_keep_0based_indices(self):
        """Changing bonds are reported with the 0-based indices of info.wbo (bond to atom 0 included)."""
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        atom_types = np.array(["O", "H", "O"])
        reac = StructuralInformation(
            nat=3,
            xyz=xyz,
            wbo_dict={(0, 1): 0.9, (1, 2): 0.05},
            atom_types=atom_types,
            hessian=np.zeros((9, 9)),
        )
        prod = StructuralInformation(
            nat=3,
            xyz=xyz,
            wbo_dict={(0, 1): 0.05, (1, 2): 0.9},
            atom_types=atom_types,
            hessian=np.zeros((9, 9)),
        )

        result = compare_wbo_differences(reac, prod, threshold=0.1)

        assert {bond for bond, _, _, _ in result["changing_bonds"]} == {(0, 1), (1, 2)}
