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
    StructuralInformation,
)
from ffits.utils.wbo_analysis import (
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
                i, j = bond[0] - 1, bond[1] - 1  # Convert to 0-based
                expected_change = product_wbo[bond] - reactant_wbo[bond]
                np.testing.assert_almost_equal(diff_matrix[i, j], expected_change)
                break  # Just test one bond

    def test_error_on_atom_count_mismatch(self, reactant_structure, product_structure):
        """Test error on atom count mismatch."""
        product_structure.info.nat = product_structure.info.nat + 1

        with pytest.raises(ValueError, match="same number of atoms"):
            get_wbo_matrix_difference(reactant_structure, product_structure)
