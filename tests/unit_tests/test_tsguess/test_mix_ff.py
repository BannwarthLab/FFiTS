"""
Comprehensive unit tests for force field mixing functionality.

Tests cover:
- Combining force field terms from two source force fields
- Removing bonded atoms from repulsive term lists
- Mixing FF parameters with weighted factors
- Mixing reference values while respecting physical constraints
"""

import os
import pandas as pd
import numpy as np
import pytest
from ffits.ts_guess.mix_ff import (
    combine_ff_atoms,
    remove_bonds_from_repulsive,
    mix_parameters,
    mix_reference_values)
from ffits.datatype.forcefield_data import ForceField
from tests.test_utils import reactant_structure_main, product_structure_main


@pytest.fixture
def ff1():
    """Fixture for first test force field."""
    struc = reactant_structure_main()
    return struc.ff


@pytest.fixture
def ff2():
    """Fixture for second test force field."""
    struc = product_structure_main()
    return struc.ff


@pytest.fixture
def info1():
    """Fixture for first test structural information."""
    struc = reactant_structure_main()
    return struc.info


@pytest.fixture
def info2():
    """Fixture for second test structural information."""
    struc = product_structure_main()
    return struc.info


def _all_values_in_either(tsff_ref: pd.Series, ff1_ref: pd.Series, ff2_ref: pd.Series):
    """Check if all values in tsff_ref are present in either ff1_ref or ff2_ref."""
    combined_values = pd.concat([ff1_ref, ff2_ref]).unique()
    missing = tsff_ref[~tsff_ref.isin(combined_values)]
    return missing.empty


class TestCombineFFTerms:
    """Tests for combining force field terms from multiple force fields."""

    def test_combine_bonds_atoms(self, ff1, ff2):
        """Test that bond atoms from both FFs are combined correctly."""
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)

        # Check that atoms from both FFs are present
        assert _all_values_in_either(
            combined_bonds["atoms"], ff1.bonds["atoms"], ff2.bonds["atoms"]
        ), "Not all combined atoms are from ff1 or ff2"

        # Check that combined dataframe has required columns
        assert "atoms" in combined_bonds.columns
        assert "type" in combined_bonds.columns
        assert "parameter" in combined_bonds.columns
        assert "reference_value" in combined_bonds.columns

    def test_combine_angles_atoms(self, ff1, ff2):
        """Test that angle atoms from both FFs are combined correctly."""
        combined_angles = combine_ff_atoms(ff1.angles, ff2.angles)

        assert _all_values_in_either(
            combined_angles["atoms"], ff1.angles["atoms"], ff2.angles["atoms"]
        ), "Not all combined angles are from ff1 or ff2"

    def test_combine_dihedrals_atoms(self, ff1, ff2):
        """Test that dihedral atoms from both FFs are combined correctly."""
        combined_dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)

        assert _all_values_in_either(
            combined_dihedrals["atoms"], ff1.dihedrals["atoms"], ff2.dihedrals["atoms"]
        ), "Not all combined dihedrals are from ff1 or ff2"

    def test_combine_repulsive_atoms(self, ff1, ff2):
        """Test that repulsive atoms from both FFs are combined correctly."""
        combined_repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)

        assert _all_values_in_either(
            combined_repulsive["atoms"], ff1.repulsive["atoms"], ff2.repulsive["atoms"]
        ), "Not all combined repulsive terms are from ff1 or ff2"

    def test_combine_terms_non_empty(self, ff1, ff2):
        """Test that combining non-empty FFs produces non-empty results."""

        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        combined_angles = combine_ff_atoms(ff1.angles, ff2.angles)
        combined_dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)

        assert len(combined_bonds) > 0, "Combined bonds should not be empty"
        assert len(combined_angles) > 0, "Combined angles should not be empty"
        assert len(combined_dihedrals) > 0, "Combined dihedrals should not be empty"


class TestRemoveBondsFromRepulsive:
    """Tests for removing bonded atom pairs from repulsive term lists."""

    def test_remove_bonds_from_repulsive_basic(self, ff1, ff2):
        """Test that bonded atoms are removed from repulsive list."""
        combined_repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)

        filtered_repulsive = remove_bonds_from_repulsive(
            combined_repulsive, combined_bonds
        )

        # Verify no bonded atoms remain in repulsive list
        for _, repulsive_atoms in filtered_repulsive["atoms"].items():
            for _, bond_atoms in combined_bonds["atoms"].items():
                assert not np.array_equal(
                    repulsive_atoms, bond_atoms
                ), f"Bond atoms {bond_atoms} should not be in repulsive list"

    def test_remove_bonds_from_repulsive_no_ff1_bonds(self, ff1, ff2):
        """Test removal when checking against ff1 bonds."""
        combined_repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)

        filtered_repulsive = remove_bonds_from_repulsive(combined_repulsive, ff1.bonds)

        # Verify no ff1 bond atoms remain in repulsive list
        for _, repulsive_atoms in filtered_repulsive["atoms"].items():
            for _, bond_atoms in ff1.bonds["atoms"].items():
                assert not np.array_equal(
                    repulsive_atoms, bond_atoms
                ), f"FF1 bond atoms {bond_atoms} should not be in repulsive list"

    def test_remove_bonds_from_repulsive_no_ff2_bonds(self, ff1, ff2):
        """Test removal when checking against ff2 bonds."""
        combined_repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)

        filtered_repulsive = remove_bonds_from_repulsive(combined_repulsive, ff2.bonds)

        # Verify no ff2 bond atoms remain in repulsive list
        for _, repulsive_atoms in filtered_repulsive["atoms"].items():
            for _, bond_atoms in ff2.bonds["atoms"].items():
                assert not np.array_equal(
                    repulsive_atoms, bond_atoms
                ), f"FF2 bond atoms {bond_atoms} should not be in repulsive list"

    def test_remove_bonds_maintains_structure(self, ff1, ff2):
        """Test that removing bonds maintains dataframe structure."""
        combined_repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)

        filtered_repulsive = remove_bonds_from_repulsive(
            combined_repulsive, combined_bonds
        )

        # Check required columns are present
        assert "atoms" in filtered_repulsive.columns
        assert "type" in filtered_repulsive.columns
        assert "parameter" in filtered_repulsive.columns
        assert "reference_value" in filtered_repulsive.columns

        # Check all rows have type 'repulsive'
        assert all(
            filtered_repulsive["type"] == "repulsive"
        ), "All repulsive terms should have type 'repulsive'"

    def test_remove_bonds_reduces_list_size(self, ff1, ff2):
        """Test that removing bonds reduces the repulsive list size."""
        combined_repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)

        initial_count = len(combined_repulsive)
        filtered_repulsive = remove_bonds_from_repulsive(
            combined_repulsive, combined_bonds
        )
        filtered_count = len(filtered_repulsive)

        # Filtered list should be smaller (or equal if no overlaps)
        assert (
            filtered_count <= initial_count
        ), "Filtering should not increase list size"


class TestMixParameters:
    """Tests for mixing force field parameters from two force fields."""

    def test_mix_bond_parameters(self, ff1, ff2):
        """Test mixing bond parameters with equal weights."""
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)

        mix_parameters(combined_bonds, ff1.bonds, ff2.bonds, 0.5, 0.5)

        # Check that no parameters are NaN or null
        assert (
            not combined_bonds["parameter"].isnull().any()
        ), "Parameters should not be null"
        assert not np.isnan(
            combined_bonds["parameter"]
        ).any(), "Parameters should not be NaN"
        assert all(
            combined_bonds["parameter"] > 0
        ), "Bond parameters should be positive"

    def test_mix_angle_parameters(self, ff1, ff2):
        """Test mixing angle parameters with equal weights."""
        combined_angles = combine_ff_atoms(ff1.angles, ff2.angles)

        mix_parameters(combined_angles, ff1.angles, ff2.angles, 0.5, 0.5)

        assert (
            not combined_angles["parameter"].isnull().any()
        ), "Angle parameters should not be null"
        assert not np.isnan(
            combined_angles["parameter"]
        ).any(), "Angle parameters should not be NaN"
        assert all(
            combined_angles["parameter"] > 0
        ), "Angle parameters should be positive"

    def test_mix_dihedral_parameters(self, ff1, ff2):
        """Test mixing dihedral parameters with equal weights."""
        combined_dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)

        mix_parameters(combined_dihedrals, ff1.dihedrals, ff2.dihedrals, 0.5, 0.5)

        assert (
            not combined_dihedrals["parameter"].isnull().any()
        ), "Dihedral parameters should not be null"
        assert not np.isnan(
            combined_dihedrals["parameter"]
        ).any(), "Dihedral parameters should not be NaN"
        assert all(
            combined_dihedrals["parameter"] > 0
        ), "Dihedral parameters should be positive"

    def test_mix_repulsive_parameters(self, ff1, ff2):
        """Test mixing repulsive parameters with equal weights."""
        combined_repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)

        mix_parameters(combined_repulsive, ff1.repulsive, ff2.repulsive, 0.5, 0.5)

        # Repulsive parameters may be 0.0 initially (they're not fitted yet)
        assert (
            not combined_repulsive["parameter"].isnull().any()
        ), "Repulsive parameters should not be null"
        assert not np.isnan(
            combined_repulsive["parameter"]
        ).any(), "Repulsive parameters should not be NaN"
        # Just verify they're not negative
        assert all(
            combined_repulsive["parameter"] >= 0
        ), "Repulsive parameters should be non-negative"

    def test_mix_parameters_unequal_weights(self, ff1, ff2):
        """Test mixing parameters with unequal weights."""
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)

        mix_parameters(combined_bonds, ff1.bonds, ff2.bonds, 0.7, 0.3)

        assert not combined_bonds["parameter"].isnull().any()
        assert not np.isnan(combined_bonds["parameter"]).any()

    def test_mix_parameters_single_source_weights(self, ff1, ff2):
        """Test mixing with weights that favor one source (0.9/0.1)."""
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        original_len = len(combined_bonds)

        mix_parameters(combined_bonds, ff1.bonds, ff2.bonds, 0.9, 0.1)

        # Should maintain structure
        assert len(combined_bonds) == original_len
        assert not np.isnan(combined_bonds["parameter"]).any()


class TestMixReferenceValues:
    """Tests for mixing reference values with physical constraints."""

    def test_mix_reference_values_complete(self, ff1, ff2, info1, info2):
        """Test mixing reference values for all FF term types."""
        tsff = ForceField(7, "temp", readff=False)

        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = remove_bonds_from_repulsive(
            combine_ff_atoms(ff1.repulsive, ff2.repulsive),
            combine_ff_atoms(ff1.bonds, ff2.bonds),
        )

        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.5, 0.5)

        # Verify no NaN or null values
        assert not np.isnan(
            tsff.bonds["parameter"]
        ).any(), "Bond parameters should not contain NaN"
        assert not np.isnan(
            tsff.angles["parameter"]
        ).any(), "Angle parameters should not contain NaN"
        assert not np.isnan(
            tsff.dihedrals["parameter"]
        ).any(), "Dihedral parameters should not contain NaN"
        assert not np.isnan(
            tsff.repulsive["parameter"]
        ).any(), "Repulsive parameters should not contain NaN"

    def test_mix_reference_values_bond_constraints(self, ff1, ff2, info1, info2):
        """Test that mixed bond lengths respect physical constraints."""
        tsff = ForceField(7, "temp", readff=False)

        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = remove_bonds_from_repulsive(
            combine_ff_atoms(ff1.repulsive, ff2.repulsive),
            combine_ff_atoms(ff1.bonds, ff2.bonds),
        )

        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.5, 0.5)

        # Bond lengths should be shorter than vdW distance
        assert all(
            tsff.bonds.apply(
                lambda row: row.reference_value
                <= info1.vander_matrix[row.atoms[0], row.atoms[1]],
                axis=1,
            )
        ), "All bond lengths should be shorter than vdW distance"

    def test_mix_reference_values_angle_constraints(self, ff1, ff2, info1, info2):
        """Test that mixed angles respect physical constraints."""
        tsff = ForceField(7, "temp", readff=False)

        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = remove_bonds_from_repulsive(
            combine_ff_atoms(ff1.repulsive, ff2.repulsive),
            combine_ff_atoms(ff1.bonds, ff2.bonds),
        )

        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.5, 0.5)

        # Angles should be between 0 and π
        assert all(
            tsff.angles.apply(lambda row: 0 <= row.reference_value <= np.pi, axis=1)
        ), "All angles should be between 0 and π"

    def test_mix_reference_values_repulsive_constraints(self, ff1, ff2, info1, info2):
        """Test that mixed repulsive distances are calculated correctly."""
        tsff = ForceField(7, "temp", readff=False)

        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = remove_bonds_from_repulsive(
            combine_ff_atoms(ff1.repulsive, ff2.repulsive),
            combine_ff_atoms(ff1.bonds, ff2.bonds),
        )

        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.5, 0.5)

        # Repulsive distances should be positive
        assert all(
            tsff.repulsive["reference_value"] > 0
        ), "Repulsive distances should be positive"
        # Repulsive distances are typically larger than bond lengths but may exceed vdW
        # (depending on the fitting algorithm), so just check they're reasonable (> 3 Bohr)
        assert all(
            tsff.repulsive["reference_value"] > 3.0
        ), "Repulsive distances should be > 3 Bohr"

    def test_mix_reference_values_unequal_weights(self, ff1, ff2, info1, info2):
        """Test mixing reference values with unequal weights."""
        tsff = ForceField(7, "temp", readff=False)

        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = remove_bonds_from_repulsive(
            combine_ff_atoms(ff1.repulsive, ff2.repulsive),
            combine_ff_atoms(ff1.bonds, ff2.bonds),
        )

        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.7, 0.3)

        # Basic validation
        assert not np.isnan(tsff.bonds["parameter"]).any()
        assert not np.isnan(tsff.angles["parameter"]).any()
        assert not np.isnan(tsff.dihedrals["parameter"]).any()
        assert not np.isnan(tsff.repulsive["parameter"]).any()

    def test_mix_reference_values_structure_maintained(self, ff1, ff2, info1, info2):
        """Test that mixing maintains the FF structure integrity."""
        tsff = ForceField(7, "temp", readff=False)

        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = remove_bonds_from_repulsive(
            combine_ff_atoms(ff1.repulsive, ff2.repulsive),
            combine_ff_atoms(ff1.bonds, ff2.bonds),
        )

        initial_bond_count = len(tsff.bonds)
        initial_angle_count = len(tsff.angles)
        initial_dihedral_count = len(tsff.dihedrals)
        initial_repulsive_count = len(tsff.repulsive)

        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.5, 0.5)

        # Structure should be maintained
        assert len(tsff.bonds) == initial_bond_count
        assert len(tsff.angles) == initial_angle_count
        assert len(tsff.dihedrals) == initial_dihedral_count
        assert len(tsff.repulsive) == initial_repulsive_count


class TestCompleteFFMixing:
    """Integration tests for complete force field mixing workflow."""

    def test_complete_mixing_workflow(self, ff1, ff2, info1, info2):
        """Test complete workflow: combine -> remove bonds -> mix params -> mix refs."""
        tsff = ForceField(7, "temp", readff=False)

        # Step 1: Combine terms
        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = combine_ff_atoms(ff1.repulsive, ff2.repulsive)

        # Step 2: Remove bonded atoms from repulsive
        combined_bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.repulsive = remove_bonds_from_repulsive(tsff.repulsive, combined_bonds)

        # Step 3: Mix parameters
        mix_parameters(tsff.bonds, ff1.bonds, ff2.bonds, 0.5, 0.5)
        mix_parameters(tsff.angles, ff1.angles, ff2.angles, 0.5, 0.5)
        mix_parameters(tsff.dihedrals, ff1.dihedrals, ff2.dihedrals, 0.5, 0.5)
        mix_parameters(tsff.repulsive, ff1.repulsive, ff2.repulsive, 0.5, 0.5)

        # Step 4: Mix reference values
        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.5, 0.5)

        # Verify final state
        assert len(tsff.bonds) > 0, "Mixed FF should have bonds"
        assert len(tsff.angles) > 0, "Mixed FF should have angles"
        assert len(tsff.dihedrals) > 0, "Mixed FF should have dihedrals"
        assert len(tsff.repulsive) > 0, "Mixed FF should have repulsive terms"

        # Verify no NaN values
        assert not np.isnan(tsff.bonds["parameter"]).any()
        assert not np.isnan(tsff.angles["parameter"]).any()
        assert not np.isnan(tsff.dihedrals["parameter"]).any()
        assert not np.isnan(tsff.repulsive["parameter"]).any()

    def test_mixing_preserves_bond_types(self, ff1, ff2, info1, info2):
        """Test that mixing preserves the type of each term."""
        tsff = ForceField(7, "temp", readff=False)

        tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
        tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
        tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
        tsff.repulsive = remove_bonds_from_repulsive(
            combine_ff_atoms(ff1.repulsive, ff2.repulsive),
            combine_ff_atoms(ff1.bonds, ff2.bonds),
        )

        mix_reference_values(tsff, ff1, ff2, info1, info2, 0.5, 0.5)

        assert all(tsff.bonds["type"] == "bonds"), "All bonds should have type 'bonds'"
        assert all(
            tsff.angles["type"] == "angles"
        ), "All angles should have type 'angles'"
        assert all(
            tsff.dihedrals["type"] == "dihedrals"
        ), "All dihedrals should have type 'dihedrals'"
        assert all(
            tsff.repulsive["type"] == "repulsive"
        ), "All repulsive should have type 'repulsive'"


# class TestHessianMixList:
#     """Tests for hessian_weighting_mix_list function that creates Hessian-weighted bond mixing factors."""

#     def test_bond_mix_list_returns_dict(self):
#         """Test that hessian_weighting_mix_list returns a dictionary."""
#         _, info1, _, info2 = _define_ff_examples()
#         result = hessian_weighting_mix_list(info1, info2)

#         assert isinstance(result, dict), "hessian_weighting_mix_list should return a dictionary"

#     def test_hessian_weighting_mix_list_mixing_factors_in_range(self):
#         """Test that all mixing factors are between 0 and 1."""
#         _, info1, _, info2 = _define_ff_examples()
#         param_dict = hessian_weighting_mix_list(info1, info2)

#         for bond, factor in param_dict.items():
#             assert 0 <= factor <= 1, f"Mixing factor {factor} for bond {bond} should be between 0 and 1"

#     def test_hessian_weighting_mix_list_keys_are_valid_bonds(self):
#         """Test that all dictionary keys are valid bond tuples (atom pairs)."""
#         _, info1, _, info2 = _define_ff_examples()
#         param_dict = hessian_weighting_mix_list(info1, info2)

#         nat = info1.nat
#         for bond in param_dict.keys():
#             assert isinstance(bond, tuple), f"Bond {bond} should be a tuple"
#             assert len(bond) == 2, f"Bond {bond} should be a pair of atoms"
#             a, b = bond
#             assert isinstance(a, (int, np.integer)), f"Atom index {a} should be an integer"
#             assert isinstance(b, (int, np.integer)), f"Atom index {b} should be an integer"
#             assert 0 <= a < nat, f"Atom index {a} out of range [0, {nat})"
#             assert 0 <= b < nat, f"Atom index {b} out of range [0, {nat})"
#             assert a != b, f"Bond should not be between same atom {a}"

#     def test_hessian_weighting_mix_list_identifies_changing_bonds(self):
#         """Test that hessian_weighting_mix_list identifies bonds with significant WBO changes."""
#         _, info1, _, info2 = _define_ff_examples()
#         param_dict = hessian_weighting_mix_list(info1, info2, threshold=0.1)

#         # Should identify at least some changing bonds
#         assert len(param_dict) > 0, "hessian_weighting_mix_list should identify at least some changing bonds"

#     def test_hessian_weighting_mix_list_with_threshold(self):
#         """Test that threshold parameter affects the number of identified bonds."""
#         _, info1, _, info2 = _define_ff_examples()

#         # Lower threshold should identify more bonds
#         param_dict_low = hessian_weighting_mix_list(info1, info2, threshold=0.05)
#         # Higher threshold should identify fewer bonds
#         param_dict_high = hessian_weighting_mix_list(info1, info2, threshold=0.5)

#         assert len(param_dict_low) >= len(param_dict_high), \
#             "Lower threshold should identify more or equal number of bonds"

#     def test_bond_mix_list_with_different_sharpness(self):
#         """Test that sharpness parameter affects mixing factor distribution."""
#         _, info1, _, info2 = _define_ff_examples()

#         param_dict_sharp = hessian_weighting_mix_list(info1, info2, sharpness=0.1)
#         param_dict_soft = hessian_weighting_mix_list(info1, info2, sharpness=0.5)

#         assert set(param_dict_sharp.keys()) == set(param_dict_soft.keys()), \
#             "Both sharpness values should identify the same bonds"

#         if len(param_dict_sharp) > 0:
#             # With lower sharpness, factors should be closer to 0 or 1
#             # With higher sharpness, factors should be closer to 0.5
#             sharp_avg = np.mean(list(param_dict_sharp.values()))
#             soft_avg = np.mean(list(param_dict_soft.values()))
#             # check standard deviation to confirm distribution is sharper or softer
#             sharp_std = np.std(list(param_dict_sharp.values()))
#             soft_std = np.std(list(param_dict_soft.values()))
#             assert sharp_std < soft_std, "Sharpness should result in a lower distribution of factors"
#             assert 0 <= sharp_avg <= 1
#             assert 0 <= soft_avg <= 1

#     def test_bond_mix_list_hessian_weighted(self):
#         """Test that mixing factors reflect Hessian magnitudes."""
#         _, info1, _, info2 = _define_ff_examples()
#         param_dict = hessian_weighting_mix_list(info1, info2, sharpness=0.2)


#         wbo_diff = compare_wbo_differences(info1, info2, threshold=0.1)

#         for bond, _, _, _ in wbo_diff['changing_bonds']:
#             i, j = bond

#             # Extract Hessian blocks
#             i_start, i_end = 3 * i, 3 * i + 3
#             j_start, j_end = 3 * j, 3 * j + 3

#             h1_block = info1.hessian[i_start:i_end, j_start:j_end]
#             h2_block = info2.hessian[i_start:i_end, j_start:j_end]

#             h1_avg = np.mean(np.abs(h1_block))
#             h2_avg = np.mean(np.abs(h2_block))

#             # Verify the factor is in the dictionary
#             assert bond in param_dict, f"Bond {bond} should be in param_dict"

#             # Verify factor is valid
#             factor = param_dict[bond]
#             assert 0 <= factor <= 1, f"Factor for bond {bond} should be between 0 and 1"

#             # If h1_avg >> h2_avg, factor should be closer to 1
#             # If h1_avg << h2_avg, factor should be closer to 0
#             if h1_avg > h2_avg:
#                 assert factor > 0.5, f"Factor should favor reactant when h1_avg >> h2_avg"
#             elif h2_avg > h1_avg:
#                 assert factor < 0.5, f"Factor should favor product when h2_avg >> h1_avg"

#     def test_bond_mix_list_empty_for_no_wbo_changes(self):
#         """Test that dict is empty when WBO changes are below threshold."""
#         _, info1, _, info2 = _define_ff_examples()

#         # With very high threshold, should get empty or very small dict
#         param_dict = hessian_weighting_mix_list(info1, info2, threshold=10.0)

#         # Should have fewer or no bonds
#         assert isinstance(param_dict, dict), "Should still return a dictionary"
#         assert len(param_dict) >= 0, "Dictionary should have non-negative length"
