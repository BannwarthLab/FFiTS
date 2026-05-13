import numpy as np
from pathlib import Path
import pytest
from ffits.external.molbar import get_combinded_priorities, _define_bonds_for_molbar
from molbar.molecule.molecule import Molecule
from tests.test_utils import reactant_structure_main, product_structure_main


@pytest.fixture
def reactant_structure():
    return reactant_structure_main()


@pytest.fixture
def product_structure():
    return product_structure_main()


class TestDefineBondsForMolbar:
    """Test _define_bonds_for_molbar function."""

    def test_cn_matrix_is_correct(self, reactant_structure, product_structure):
        """Test that bonds are defined correctly."""
        bo_threshold = 0.5
        mol = Molecule(
            coordinates=reactant_structure.info.xyz,
            elements=reactant_structure.info.atom_types,
        )
        _define_bonds_for_molbar(
            mol,
            reactant_structure.info,
            product_structure.info,
            bo_threshold=bo_threshold,
        )
        for i in range(mol.cn_matrix.shape[0]):
            for j in range(1, mol.cn_matrix.shape[1]):
                if mol.cn_matrix[i, j] == 1:
                    assert (
                        reactant_structure.info.bo_matrix[i, j]
                        + product_structure.info.bo_matrix[i, j]
                        > bo_threshold
                    )

    def test_cn_matrix_is_symmetric(self, reactant_structure, product_structure):
        """Test that the connectivity matrix is symmetric."""
        mol = Molecule(
            coordinates=reactant_structure.info.xyz,
            elements=reactant_structure.info.atom_types,
        )
        _define_bonds_for_molbar(mol, reactant_structure.info, product_structure.info)
        assert np.array_equal(mol.cn_matrix, mol.cn_matrix.T)

    def test_cn_matrix_diagonal_is_zero(self, reactant_structure, product_structure):
        """Test that the diagonal of the connectivity matrix is zero."""
        mol = Molecule(
            coordinates=reactant_structure.info.xyz,
            elements=reactant_structure.info.atom_types,
        )
        _define_bonds_for_molbar(mol, reactant_structure.info, product_structure.info)
        assert np.all(np.diag(mol.cn_matrix) == 0)

    def test_cn_matrix_is_filled_only_with_zeros_and_ones(
        self, reactant_structure, product_structure
    ):
        """Test that the connectivity matrix is filled only with 0s and 1s."""
        mol = Molecule(
            coordinates=reactant_structure.info.xyz,
            elements=reactant_structure.info.atom_types,
        )
        _define_bonds_for_molbar(mol, reactant_structure.info, product_structure.info)
        assert np.all(np.logical_or(mol.cn_matrix == 0, mol.cn_matrix == 1))


class TestGetCombinedPriorities:
    """Test get_combinded_priorities function."""

    def test_priorities_have_correct_length(
        self, reactant_structure, product_structure
    ):
        """Test that combined priorities have the correct length."""
        priorities = get_combinded_priorities(
            reactant_structure.info, product_structure.info
        )
        assert len(priorities) == reactant_structure.info.nat

    def test_priorities_are_integers_and_not_none(
        self, reactant_structure, product_structure
    ):
        """Test that combined priorities are integers."""
        priorities = get_combinded_priorities(
            reactant_structure.info, product_structure.info
        )
        assert all(
            priority is not None for priority in priorities.values()
        ), "Priorities should not be None."
        assert all(
            isinstance(priority, int) for priority in priorities.values()
        ), "Priorities should be integers."

    def test_lowest_priority_is_one(self, reactant_structure, product_structure):
        """Test that the lowest priority is 1."""
        priorities = get_combinded_priorities(
            reactant_structure.info, product_structure.info
        )
        assert min(priorities.values()) == 1, "The lowest priority should be 1."
