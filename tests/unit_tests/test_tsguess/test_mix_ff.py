"""
Comprehensive unit tests for force field mixing functionality.

Tests cover:
- Combining force field terms from two source force fields
- Removing bonded atoms from repulsive term lists
- Mixing FF parameters with weighted factors
- Mixing reference values while respecting physical constraints
- The create_tsff interface
"""

import inspect
from types import SimpleNamespace
import networkx as nx
import pandas as pd
import numpy as np
import pytest
from ffits.ts_guess import mix_ff
from ffits.ts_guess.mix_ff import (
    combine_ff_atoms,
    create_tsff,
    remove_bonds_from_repulsive,
    mix_parameters,
    mix_reference_values,
)
from ffits.datatype.calculation_data import CalculationData
from ffits.utils.geometry import angle, angstrom2bohr, dihedral_angle
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

    def test_remove_bonds_keeps_reversed_pairs(self):
        """A bond (a, b) removes only the repulsive pair (a, b), not (b, a)."""
        repulsive = _term_df(
            "repulsive",
            [((0, 1), 1, 1), ((2, 3), 1, 1), ((3, 4), 1, 1), ((4, 5), 1, 1)],
        )
        bonds = _term_df("bonds", [((0, 1), 1, 1), ((3, 2), 1, 1), ((5, 6), 1, 1)])
        filtered = remove_bonds_from_repulsive(repulsive, bonds)
        assert filtered["atoms"].tolist() == [(2, 3), (3, 4), (4, 5)]
        assert filtered.index.tolist() == [0, 1, 2]


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
            tsff.bonds["reference_value"]
        ).any(), "Bond reference values should not contain NaN"
        assert not np.isnan(
            tsff.angles["reference_value"]
        ).any(), "Angle reference values should not contain NaN"
        assert not np.isnan(
            tsff.dihedrals["reference_value"]
        ).any(), "Dihedral reference values should not contain NaN"
        assert not np.isnan(
            tsff.repulsive["reference_value"]
        ).any(), "Repulsive reference values should not contain NaN"

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
        assert not np.isnan(tsff.bonds["reference_value"]).any()
        assert not np.isnan(tsff.angles["reference_value"]).any()
        assert not np.isnan(tsff.dihedrals["reference_value"]).any()
        assert not np.isnan(tsff.repulsive["reference_value"]).any()

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
        for term in ("bonds", "angles", "dihedrals", "repulsive"):
            assert not np.isnan(getattr(tsff, term)["reference_value"]).any()

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


def _term_df(term: str, rows: list) -> pd.DataFrame:
    """FF term DataFrame from (atoms, parameter, reference_value) rows."""
    return pd.DataFrame(
        {
            "type": term,
            "atoms": [r[0] for r in rows],
            "parameter": [float(r[1]) for r in rows],
            "reference_value": [float(r[2]) for r in rows],
        }
    )


def _forcefield(nat: int, **terms) -> ForceField:
    """ForceField with the given term DataFrames (others empty)."""
    ff = ForceField(nat, "unused.csv", readff=False)
    for term, df in terms.items():
        setattr(ff, term, df)
    return ff


class TestCombineFFAtomsBehaviour:
    """Exact behaviour of combine_ff_atoms on small hand-made inputs."""

    def test_keeps_first_and_sorts(self):
        df1 = _term_df("bonds", [((2, 3), 1.0, 1.0), ((0, 1), 2.0, 2.0)])
        df2 = _term_df("bonds", [((0, 1), 99.0, 99.0), ((1, 2), 3.0, 3.0)])
        combined = combine_ff_atoms(df1, df2)
        assert combined["atoms"].tolist() == [(0, 1), (1, 2), (2, 3)]
        assert combined["parameter"].tolist() == [2.0, 3.0, 1.0]
        assert combined.index.tolist() == [0, 1, 2]

    def test_reversed_tuples_are_different_terms(self):
        df1 = _term_df("bonds", [((0, 1), 1.0, 1.0)])
        df2 = _term_df("bonds", [((1, 0), 1.0, 1.0)])
        assert combine_ff_atoms(df1, df2)["atoms"].tolist() == [(0, 1), (1, 0)]

    def test_atoms_become_tuples(self):
        df1 = _term_df("angles", [([0, 1, 2], 1.0, 1.0)])
        df2 = _term_df("angles", [((0, 1, 2), 5.0, 5.0)])
        combined = combine_ff_atoms(df1, df2)
        assert combined["atoms"].tolist() == [(0, 1, 2)]
        assert combined["parameter"].tolist() == [1.0]

    def test_inputs_not_modified(self, ff1, ff2):
        before1, before2 = ff1.bonds.copy(), ff2.bonds.copy()
        combine_ff_atoms(ff1.bonds, ff2.bonds)
        pd.testing.assert_frame_equal(ff1.bonds, before1)
        pd.testing.assert_frame_equal(ff2.bonds, before2)


class TestMixReferenceValuesRules:
    """The term-specific rules of mix_reference_values on a hand-made 4-atom system."""

    @pytest.fixture
    def infos(self):
        xyz1 = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 2.0, 0.0], [4.0, 2.5, 1.0]]
        )
        xyz2 = xyz1 + np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.8], [1.0, 0.0, -2.0]]
        )
        vander = np.full((4, 4), 5.0)
        atom_types = ["C", "H", "O", "h"]
        graph = nx.complete_graph(4)
        graph.remove_edge(1, 2)  # H1...O2 only a hydrogen bond
        return (
            SimpleNamespace(
                fortran_xyz=xyz1.T,
                vander_matrix=vander,
                atom_types=atom_types,
                complete_graph=graph,
            ),
            SimpleNamespace(
                fortran_xyz=xyz2.T,
                vander_matrix=vander,
                atom_types=atom_types,
                complete_graph=graph,
            ),
        )

    def _mix(
        self,
        infos,
        ff1,
        ff2,
        fact1=0.5,
        fact2=0.5,
        hydrogen_covalent_radius=False,
    ):
        tsff = _forcefield(4)
        for term in ("bonds", "angles", "dihedrals", "repulsive"):
            setattr(
                tsff, term, combine_ff_atoms(getattr(ff1, term), getattr(ff2, term))
            )
        mix_reference_values(
            tsff,
            ff1,
            ff2,
            *infos,
            fact1,
            fact2,
            hydrogen_covalent_radius=hydrogen_covalent_radius,
        )
        return tsff

    def test_bonds_capped_at_vdw(self, infos):
        ff1 = _forcefield(
            4, bonds=_term_df("bonds", [((0, 1), 1, 6.0), ((1, 2), 1, 2.0)])
        )
        ff2 = _forcefield(4, bonds=_term_df("bonds", [((0, 1), 1, 3.0)]))
        tsff = self._mix(infos, ff1, ff2)
        # (0, 1): min(6, vdW 5) and 3; (1, 2): 2 and the vdW distance 5 for the missing bond
        assert tsff.bonds["reference_value"].tolist() == [4.0, 3.5]

    def test_hydrogen_covalent_radius_only_for_changing_covalent_h_heavy_bonds(
        self, infos
    ):
        # atoms: C0, H1, O2, h3; (0, 1) in both, (0, 3) only in ff2, the others only in ff1
        ff1 = _forcefield(
            4,
            bonds=_term_df(
                "bonds",
                [
                    ((0, 1), 1, 2.0),
                    ((0, 2), 1, 2.0),
                    ((1, 2), 1, 4.5),
                    ((1, 3), 1, 2.0),
                    ((2, 3), 1, 2.0),
                ],
            ),
        )
        ff2 = _forcefield(
            4, bonds=_term_df("bonds", [((0, 1), 1, 3.0), ((0, 3), 1, 2.0)])
        )
        shift = angstrom2bohr(0.91 - 0.32)  # vdW - covalent radius of H
        tsff = self._mix(infos, ff1, ff2, hydrogen_covalent_radius=True)
        mixed = dict(zip(tsff.bonds["atoms"], tsff.bonds["reference_value"]))
        assert mixed[(0, 1)] == pytest.approx(2.5)  # C-H: present in both
        assert mixed[(0, 2)] == pytest.approx(3.5)  # C-O: no hydrogen
        assert mixed[(1, 3)] == pytest.approx(3.5)  # H-h: no heavier element
        assert mixed[(1, 2)] == pytest.approx(4.75)  # H...O: hydrogen bond only
        assert mixed[(0, 3)] == pytest.approx((5.0 - shift + 2.0) / 2)  # C-h forming
        assert mixed[(2, 3)] == pytest.approx((2.0 + 5.0 - shift) / 2)  # O-h breaking

    def test_hydrogen_covalent_radius_not_below_bonded_length(self, infos):
        ff1 = _forcefield(4, bonds=_term_df("bonds", [((2, 3), 1, 4.5)]))
        tsff = self._mix(infos, ff1, _forcefield(4), hydrogen_covalent_radius=True)
        # missing side max(5 - shift, 4.5) = 4.5
        assert tsff.bonds["reference_value"].tolist() == [4.5]

    def test_missing_angle_and_dihedral_from_own_geometry(self, infos):
        info1, info2 = infos
        ff1 = _forcefield(4)
        ff2 = _forcefield(
            4,
            angles=_term_df("angles", [((0, 1, 2), 1, 1.0)]),
            dihedrals=_term_df("dihedrals", [((0, 1, 2, 3), 1, 1.0)]),
        )
        tsff = self._mix(infos, ff1, ff2, 1.0, 0.0)
        assert tsff.angles["reference_value"].iloc[0] == pytest.approx(
            angle(info1.fortran_xyz, 0, 1, 2), abs=1e-8
        )
        assert tsff.dihedrals["reference_value"].iloc[0] == pytest.approx(
            dihedral_angle(info1.fortran_xyz, 0, 1, 2, 3), abs=1e-8
        )

        tsff = self._mix(infos, ff2, ff1, 0.0, 1.0)
        assert tsff.angles["reference_value"].iloc[0] == pytest.approx(
            angle(info2.fortran_xyz, 0, 1, 2), abs=1e-8
        )

    def test_dihedral_circular_average(self, infos):
        ff1 = _forcefield(
            4, dihedrals=_term_df("dihedrals", [((0, 1, 2, 3), 1, np.pi - 0.1)])
        )
        ff2 = _forcefield(
            4, dihedrals=_term_df("dihedrals", [((0, 1, 2, 3), 1, -np.pi + 0.1)])
        )
        tsff = self._mix(infos, ff1, ff2)
        assert tsff.dihedrals["reference_value"].iloc[0] == pytest.approx(
            np.pi, abs=1e-7
        )

    def test_angles_and_repulsive_use_factors(self, infos):
        ff1 = _forcefield(
            4,
            angles=_term_df("angles", [((0, 1, 2), 1, 1.0)]),
            repulsive=_term_df("repulsive", [((0, 3), 1, 10.0)]),
        )
        ff2 = _forcefield(
            4,
            angles=_term_df("angles", [((0, 1, 2), 1, 2.0)]),
            repulsive=_term_df("repulsive", [((0, 3), 1, 20.0)]),
        )
        tsff = self._mix(infos, ff1, ff2, 0.9, 0.1)
        assert tsff.angles["reference_value"].iloc[0] == pytest.approx(1.1)
        assert tsff.repulsive["reference_value"].iloc[0] == pytest.approx(11.0)

    def test_missing_repulsive_from_other_ff(self, infos):
        ff1 = _forcefield(4, repulsive=_term_df("repulsive", [((0, 3), 1, 10.0)]))
        ff2 = _forcefield(4)
        tsff = self._mix(infos, ff1, ff2, 0.7, 0.3)
        assert tsff.repulsive["reference_value"].iloc[0] == pytest.approx(10.0)

    def test_factors_must_sum_to_one(self, infos):
        with pytest.raises(ValueError):
            self._mix(infos, _forcefield(4), _forcefield(4), 0.7, 0.7)


class TestCreateTsff:
    """Interface and behaviour of create_tsff on the example reaction."""

    @pytest.fixture
    def calcdata(self, tmp_path):
        cd = CalculationData.from_default()
        cd.ts_path.ff_filename = str(tmp_path / "tsff.csv")
        return cd

    def test_returns_tsff(self, ff1, info1, ff2, info2, calcdata):
        tsff = create_tsff(ff1=ff1, info1=info1, ff2=ff2, info2=info2, calcdata=calcdata)
        assert isinstance(tsff, ForceField)
        assert tsff.nat == ff1.nat
        assert tsff.ff_filename == calcdata.ts_path.ff_filename

    def test_writes_tsff(self, ff1, info1, ff2, info2, calcdata):
        tsff = create_tsff(ff1, info1, ff2, info2, calcdata)
        reread = ForceField(ff1.nat, calcdata.ts_path.ff_filename, readff=True)
        for term in ("bonds", "angles", "dihedrals", "repulsive"):
            assert (
                reread.__dict__[term]["atoms"].tolist()
                == getattr(tsff, term)["atoms"].tolist()
            )
            np.testing.assert_allclose(
                reread.__dict__[term]["reference_value"],
                getattr(tsff, term)["reference_value"],
            )

    def test_terms_are_union_of_both_ffs(self, ff1, info1, ff2, info2, calcdata):
        tsff = create_tsff(ff1, info1, ff2, info2, calcdata)
        for term in ("bonds", "angles", "dihedrals"):
            expected = set(getattr(ff1, term)["atoms"]) | set(
                getattr(ff2, term)["atoms"]
            )
            assert set(getattr(tsff, term)["atoms"]) == expected
        assert not set(tsff.repulsive["atoms"]) & set(tsff.bonds["atoms"])

    def test_inputs_not_modified(self, ff1, info1, ff2, info2, calcdata):
        before = {
            (i, term): getattr(ff, term).copy()
            for i, ff in enumerate((ff1, ff2))
            for term in ("bonds", "angles", "dihedrals", "repulsive")
        }
        create_tsff(ff1, info1, ff2, info2, calcdata)
        for (i, term), df in before.items():
            pd.testing.assert_frame_equal(getattr((ff1, ff2)[i], term), df)

    def test_constant_factors(self, ff1, info1, ff2, info2, calcdata):
        calcdata.ts_calc.factor_reactant = 0.7
        calcdata.ts_calc.factor_product = 0.3
        tsff = create_tsff(ff1, info1, ff2, info2, calcdata)
        assert tsff.start_from_reactant is True
        p1 = dict(zip(ff1.bonds["atoms"], ff1.bonds["parameter"]))
        p2 = dict(zip(ff2.bonds["atoms"], ff2.bonds["parameter"]))
        for atoms, mixed in zip(tsff.bonds["atoms"], tsff.bonds["parameter"]):
            c1, c2 = p1.get(atoms, p2.get(atoms)), p2.get(atoms, p1.get(atoms))
            assert mixed == pytest.approx(0.7 * c1 + 0.3 * c2, abs=1e-8)


@pytest.mark.parametrize(
    "module, name, parameters",
    [
        (mix_ff, "create_tsff", ["ff1", "info1", "ff2", "info2", "calcdata"]),
        (mix_ff, "combine_ff_topology", ["ff1", "ff2", "filename"]),
        (mix_ff, "combine_ff_atoms", ["df1", "df2"]),
        (mix_ff, "remove_bonds_from_repulsive", ["df_source", "df_reference"]),
        (
            mix_ff,
            "mix_parameters",
            ["tsff_df", "ff1_df", "ff2_df", "fact1", "fact2"],
        ),
        (
            mix_ff,
            "mix_reference_values",
            [
                "tsff",
                "ff1",
                "ff2",
                "info1",
                "info2",
                "fact1",
                "fact2",
                "hydrogen_covalent_radius",
            ],
        ),
    ],
)
def test_public_interface(module, name, parameters):
    """The public functions of mix_ff keep their names and argument lists."""
    assert list(inspect.signature(getattr(module, name)).parameters) == parameters
