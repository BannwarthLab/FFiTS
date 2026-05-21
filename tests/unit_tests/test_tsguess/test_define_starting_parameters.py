import pytest
import numpy as np
import pandas as pd
import math
from ffits.ts_guess.define_starting_parameters import (
    canonical_dihedral,
    _normalize_improper,
    filter_dihedrals,
    _find_hydrogen_bonding,
    _remove_duplicate_improper_dihedrals,
)
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import StructuralInformation
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)



# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_structure():
    """Create a simple 3-atom structure (O-H-C) for testing."""
    # Coordinates: O at origin, H at (1, 0, 0), C at (0, 1, 1)
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [1.0, 0.0, 0.0],  # H
            [0.0, 1.0, 1.0],  # C
        ]
    )

    # Bond order matrix: O-H bond and O-C bond
    wbo_dict = {
        (0, 1): 0.9,  # O-H
        (0, 2): 1.0,  # O-C
    }

    atom_types = np.array(["O", "H", "C"])

    ff = ForceField(nat=3, ff_filename='s')

    info = StructuralInformation(
        nat=3,
        xyz=xyz,
        wbo_dict=wbo_dict,
        atom_types=atom_types,
        hessian=np.zeros((3, 3)),
    )

    return ff, info


@pytest.fixture
def structure_with_linear_hbond():
    """Create a structure with a potential hydrogen bond (nearly linear O...H-C)."""
    # O-H-C nearly linear (angle ~180°)
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],  # O (acceptor)
            [1.5, 0.0, 0.0],  # H (hydrogen)
            [3.0, 0.0, 0.0],  # C (bonded to H)
        ]
    )

    wbo_dict = {
        (1, 2): 1.0,  # H-C bond
    }

    atom_types = np.array(["O", "H", "C"])

    ff = ForceField(nat=3, ff_filename='s')

    info = StructuralInformation(
        nat=3,
        xyz=xyz,
        wbo_dict=wbo_dict,
        atom_types=atom_types,
        hessian=np.zeros((3, 3)),
    )

    return ff, info


@pytest.fixture
def structure_with_bent_hbond():
    """Create a structure with a bent hydrogen bond (angle ~90°)."""
    # O-H-C bent (angle ~90°)
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [1.5, 0.0, 0.0],  # H
            [1.5, 1.5, 0.0],  # C
        ]
    )

    wbo_dict = {
        (1, 2): 1.0,  # H-C bond
    }

    atom_types = np.array(["O", "H", "C"])

    ff = ForceField(nat=3, ff_filename='s')

    info = StructuralInformation(
        nat=3,
        xyz=xyz,
        wbo_dict=wbo_dict,
        atom_types=atom_types,
        hessian=np.zeros((3, 3)),
    )

    return ff, info


class TestCanonicalDihedral:
    """Test canonical_dihedral function."""

    def test_forward_and_reverse_are_equal(self):
        """Forward and reverse dihedral should normalize to same tuple."""
        forward = canonical_dihedral(1, 2, 3, 4)
        reverse = canonical_dihedral(4, 3, 2, 1)
        assert forward == reverse

    def test_canonical_ordering(self):
        """Canonical dihedral should return minimum of forward and reverse."""
        result = canonical_dihedral(1, 2, 3, 4)
        # (1,2,3,4) vs (4,3,2,1) -> (1,2,3,4) is smaller
        assert result == (1, 2, 3, 4)

    def test_different_dihedrals(self):
        """Different dihedrals should have different canonical forms."""
        dih1 = canonical_dihedral(1, 2, 3, 4)
        dih2 = canonical_dihedral(1, 2, 3, 5)
        assert dih1 != dih2


class TestNormalizeImproper:
    """Test _normalize_improper function."""

    def test_normalize_improper_sorts_terminals(self):
        """Terminals should be sorted in normalized form."""
        dihedral = (0, 5, 2, 3)
        result = _normalize_improper(dihedral)
        # Central atom 0, terminals sorted: (2, 3, 5)
        assert result == (0, 2, 3, 5)

    def test_different_terminal_permutations_normalize_same(self):
        """Different permutations of terminals should normalize to same form."""
        dih1 = _normalize_improper((0, 1, 2, 3))
        dih2 = _normalize_improper((0, 3, 2, 1))
        dih3 = _normalize_improper((0, 2, 3, 1))
        assert dih1 == dih2 == dih3 == (0, 1, 2, 3)

    def test_central_atom_preserved(self):
        """Central atom should always be first."""
        result = _normalize_improper((5, 10, 7, 3))
        assert result[0] == 5


class TestRemoveDuplicateImproperDihedrals:
    """Test _remove_duplicate_improper_dihedrals function."""

    def test_removes_duplicate_improper_dihedrals(self):
        """Duplicate improper dihedrals should be removed."""
        dihedrals = [
            ((0, 1, 2, 3), False),
            ((0, 3, 2, 1), False),  # Duplicate of above
            ((0, 1, 2, 3), False),  # Duplicate
        ]
        result = _remove_duplicate_improper_dihedrals(dihedrals)
        assert len(result) == 1
        assert result[0][0] == (0, 1, 2, 3)
        assert result[0][1] is False

    def test_keeps_proper_dihedrals_unchanged(self):
        """Proper dihedrals should not be deduplicated."""
        dihedrals = [
            ((0, 1, 2, 3), True),
            ((0, 1, 2, 3), True),
        ]
        result = _remove_duplicate_improper_dihedrals(dihedrals)
        # Proper dihedrals are kept as-is, so both should remain
        assert len(result) == 2

    def test_mixed_proper_and_improper(self):
        """Mix of proper and improper dihedrals."""
        dihedrals = [
            ((0, 1, 2, 3), True),
            ((0, 1, 2, 3), False),
            ((0, 1, 2, 3), False),
        ]
        result = _remove_duplicate_improper_dihedrals(dihedrals)
        assert len(result) == 2


class TestFilterDihedrals:
    """Test filter_dihedrals function."""

    def test_filters_dihedrals_with_priorities(self):
        """Should filter dihedrals and keep proper with highest priority."""
        dihedrals = [
            (0, 1, 2, 3),
            (0, 1, 2, 4),
        ]
        priorities = {0: 5, 3: 3, 4: 10}
        A = np.array(
            [
                [0, 1, 0, 1, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ]
        )

        result = filter_dihedrals(dihedrals, priorities, A)
        assert len(result) > 0
        # Result should contain tuples of (dihedral, is_proper)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)


class TestFindHydrogenBonding:
    """Test _find_hydrogen_bonding function with real data."""

    def test_detects_linear_hydrogen_bond(self, structure_with_linear_hbond):
        """Should detect hydrogen bond with nearly linear geometry."""
        ff, info = structure_with_linear_hbond

        # Create repulsive dataframe (O not bonded to H in this test)
        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        # Should detect O-H as hydrogen bond (linear geometry)
        assert len(result) > 0
        assert (0, 1) in result or (1, 0) in result
        detected_bond = (0, 1) if (0, 1) in result else (1, 0)
        assert "bl" in result[detected_bond]
        assert "angle" in result[detected_bond]
        assert "bonded_atom" in result[detected_bond]

    def test_rejects_bent_hydrogen_bond(self, structure_with_bent_hbond):
        """Should reject hydrogen bond with bent geometry (~90°)."""
        ff, info = structure_with_bent_hbond

        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        # Should not detect H-bond due to wrong angle 
        assert len(result) == 0

    def test_angle_approximately_180_degrees(self, structure_with_linear_hbond):
        """Detected hydrogen bond should have angle close to 180°."""
        ff, info = structure_with_linear_hbond

        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        if result:
            for bond_data in result.values():
                angle_rad = bond_data["angle"]
                angle_deg = math.degrees(angle_rad)
                # Should be within 20° of 180°
                assert 160 <= angle_deg <= 180 or 0 <= angle_deg <= 20

    def test_hydrogen_bond_distance_reasonable(self, structure_with_linear_hbond):
        """H-bond distance should be reasonable (< vdW distance)."""
        ff, info = structure_with_linear_hbond

        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        if result:
            for bond_data in result.values():
                distance = bond_data["bl"]
                # Distance should be positive and reasonable
                assert distance > 0
                assert distance < 4.0  # Less than typical vdW sum

    def test_no_hydrogen_bonds_for_non_donors(self):
        """Should not detect H-bonds for non-donor atom pairs."""
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],  # C
                [1.0, 0.0, 0.0],  # C
            ]
        )

        wbo_dict = {}
        atom_types = np.array(["C", "C"])

        ff = ForceField(nat=2, ff_filename='s')
        info = StructuralInformation(
            nat=2,
            xyz=xyz,
            wbo_dict=wbo_dict,
            atom_types=atom_types,
            hessian=np.zeros((2, 2)),
        )

        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        # Should not detect any H-bonds for C-C
        assert len(result) == 0

    def test_nitrogen_hydrogen_detected(self):
        """Should detect N-H hydrogen bonds."""
        # N-H-C nearly linear
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],  # N (acceptor-like)
                [1.5, 0.0, 0.0],  # H
                [3.0, 0.0, 0.0],  # C
            ]
        )

        wbo_dict = {
            (1, 2): 1.0,  # H-C
        }

        atom_types = np.array(["N", "H", "C"])
        ff = ForceField(nat=3, ff_filename='s')
        info = StructuralInformation(
            nat=3,
            xyz=xyz,
            wbo_dict=wbo_dict,
            atom_types=atom_types,
            hessian=np.zeros((3, 3)),
        )

        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        # Should detect N-H bond
        assert len(result) > 0

    def test_fluorine_hydrogen_detected(self):
        """Should detect F-H hydrogen bonds."""
        # F-H-C nearly linear
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],  # F
                [1.5, 0.0, 0.0],  # H
                [3.0, 0.0, 0.0],  # C
            ]
        )

        wbo_dict = {
            (1, 2): 1.0,  # H-C
        }

        atom_types = np.array(["F", "H", "C"])
        ff = ForceField(nat=3, ff_filename='s')
        info = StructuralInformation(
            nat=3,
            xyz=xyz,
            wbo_dict=wbo_dict,
            atom_types=atom_types,
            hessian=np.zeros((3, 3)),
        )

        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        # Should detect F-H bond
        assert len(result) > 0

    def test_returns_dictionary_structure(self, structure_with_linear_hbond):
        """Return value should be a dictionary with correct structure."""
        ff, info = structure_with_linear_hbond

        ff.repulsive = pd.DataFrame({"atoms": [(0, 1)]})

        result = _find_hydrogen_bonding(ff, info)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(value, dict)
            assert "bl" in value
            assert "angle" in value
            assert "bonded_atom" in value
            assert isinstance(value["bl"], (int, float))
            assert isinstance(value["angle"], (int, float))
            assert isinstance(value["bonded_atom"], (int, np.integer))



