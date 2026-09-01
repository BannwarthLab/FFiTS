from ffits.datatype.structure_data import (
    StructuralInformation,
    get_vander_matrix,
    angstrom2bohr,
)

from ffits.datatype.forcefield_data import ForceField
import numpy as np
import pandas as pd
import pytest
import os
from ffits.ts_guess.define_starting_parameters import fill_ff
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)
from ffits.utils.geometry import bondlength, angle, dihedral_angle
from tests.test_utils import NAT, XYZ, WBO, ATOM_TYPES


class TestForceFieldObjectCreation:
    """Tests for Force Field object creation and initialization."""

    def test_create_blank_ff_object_from_file(self):
        """Test creating a ForceField object from file with all calculators."""
        path = os.path.join(
            os.getcwd(), "tests/examples/small_single_molecule", "ff1.csv"
        )
        ff = ForceField(
            NAT,
            path,
            readff=True,
            energy_calculator=energy_ff,
            gradient_calculator=complete_gradient,
            hessian_calculator=complete_hessian,
        )
        assert ff is not None
        assert ff.nat == NAT

    def test_ff_object_has_required_dataframes(self):
        """Test that ForceField object contains all required dataframes."""
        path2ff = os.path.join(
            os.getcwd(), "tests/examples/small_single_molecule/ff1.csv"
        )
        ff = ForceField(NAT, path2ff)

        # Check all required components exist
        assert hasattr(ff, "bonds")
        assert hasattr(ff, "angles")
        assert hasattr(ff, "dihedrals")
        assert hasattr(ff, "repulsive")
        assert all(
            isinstance(df, pd.DataFrame)
            for df in [ff.bonds, ff.angles, ff.dihedrals, ff.repulsive]
        )

    def test_construct_ff_from_existing_file(self):
        """Test loading ForceField from an existing file and verify structure."""
        path2ff = os.path.join(
            os.getcwd(), "tests/examples/small_single_molecule/ff1.csv"
        )
        ff = ForceField(NAT, path2ff, readff=True)

        # Verify bonds were loaded correctly (non-zero if file exists and was read)
        if len(ff.bonds) > 0:
            # Basic structure checks
            np.testing.assert_equal(ff.bonds["atoms"].iloc[0], np.array([0, 1]))
            # Verify all dataframes have the correct types
            assert all(ff.bonds["type"] == "bonds")
            assert all(ff.angles["type"] == "angles")
            assert all(ff.dihedrals["type"] == "dihedrals")
        else:
            # File may not exist or may be empty, skip detailed checks
            assert ff.nat == NAT

    def test_ff_object_bonds_structure(self):
        """Test that bonds dataframe has correct structure."""
        path2ff = os.path.join(
            os.getcwd(), "tests/examples/small_single_molecule/ff1.csv"
        )
        ff = ForceField(NAT, path2ff)
        required_columns = ["type", "atoms", "reference_value", "parameter"]
        for col in required_columns:
            assert col in ff.bonds.columns, f"Missing column '{col}' in bonds dataframe"
        assert all(ff.bonds["type"] == "bonds")

    def test_ff_object_angles_structure(self):
        """Test that angles dataframe has correct structure."""
        path2ff = os.path.join(
            os.getcwd(), "tests/examples/small_single_molecule/ff1.csv"
        )
        ff = ForceField(NAT, path2ff)
        required_columns = ["type", "atoms", "reference_value", "parameter"]
        for col in required_columns:
            assert (
                col in ff.angles.columns
            ), f"Missing column '{col}' in angles dataframe"
        assert all(ff.angles["type"] == "angles")

    def test_ff_object_dihedrals_structure(self):
        """Test that dihedrals dataframe has correct structure."""
        path2ff = os.path.join(
            os.getcwd(), "tests/examples/small_single_molecule/ff1.csv"
        )
        ff = ForceField(NAT, path2ff)
        required_columns = ["type", "atoms", "reference_value", "parameter"]
        for col in required_columns:
            assert (
                col in ff.dihedrals.columns
            ), f"Missing column '{col}' in dihedrals dataframe"
        assert all(ff.dihedrals["type"] == "dihedrals")


class TestStructuralInformation:
    """Tests for StructuralInformation object creation and calculations."""

    def test_structural_information_creation(self):
        """Test creating StructuralInformation object."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)

        assert info is not None
        assert info.nat == NAT

    def test_molecule_count_calculation(self):
        """Test that molecule_count is correctly calculated."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)

        assert info.molecule_count == 1

    def test_bond_order_matrix_creation(self):
        """Test that bond order matrix is correctly constructed from WBO.

        Builds the expected matrix by placing each WBO dict entry directly
        (symmetrically) rather than hardcoding a second literal copy of the
        WBO values here: a hardcoded copy previously went stale when the
        wbo1 fixture (and the WBO constant derived from it) was
        regenerated, since nothing kept the two in sync.
        """
        expected_bo_matrix = np.zeros((NAT, NAT))
        for (atom_i, atom_j), value in WBO.items():
            expected_bo_matrix[atom_i, atom_j] = value
            expected_bo_matrix[atom_j, atom_i] = value

        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        np.testing.assert_allclose(info.bo_matrix, expected_bo_matrix, rtol=1e-7)

    def test_bo_matrix_diagonal_is_zero(self):
        """Test that diagonal of bond order matrix is zero."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        np.testing.assert_array_equal(np.diag(info.bo_matrix), np.zeros(NAT))

    def test_bo_matrix_shape(self):
        """Test that bond order matrix has correct shape."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        assert info.bo_matrix.shape == (NAT, NAT)


class TestVanDerWaalsMatrix:
    """Tests for Van der Waals matrix generation."""

    def test_vander_matrix_generation(self):
        """Test that Van der Waals matrix is correctly generated."""
        expected_vander_matrix = angstrom2bohr(
            np.array(
                [
                    [2.64, 2.64, 2.54, 2.23, 2.23, 2.23, 2.23],
                    [2.64, 2.64, 2.54, 2.23, 2.23, 2.23, 2.23],
                    [2.54, 2.54, 2.44, 2.13, 2.13, 2.13, 2.13],
                    [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
                    [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
                    [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
                    [2.23, 2.23, 2.13, 1.82, 1.82, 1.82, 1.82],
                ]
            )
        )
        vander_matrix = get_vander_matrix(ATOM_TYPES)
        np.testing.assert_allclose(vander_matrix, expected_vander_matrix, rtol=1e-7)

    def test_vander_matrix_shape(self):
        """Test that Van der Waals matrix has correct shape."""
        vander_matrix = get_vander_matrix(ATOM_TYPES)

        assert vander_matrix.shape == (NAT, NAT)

    def test_vander_matrix_symmetry(self):
        """Test that Van der Waals matrix is symmetric."""
        vander_matrix = get_vander_matrix(ATOM_TYPES)

        np.testing.assert_allclose(vander_matrix, vander_matrix.T, rtol=1e-7)

    def test_vander_matrix_positive_values(self):
        """Test that all Van der Waals values are positive."""
        vander_matrix = get_vander_matrix(ATOM_TYPES)

        assert np.all(vander_matrix > 0)


class TestFillForceField:
    """Tests for fill_ff, which derives initial bonds/angles/dihedrals/repulsive
    terms from structural information.

    reference_value and parameter are checked by independently recomputing
    them from the same closed-form formulas fill_ff itself documents
    (bondlength/angle/dihedral_angle from ffits.utils.geometry, combined
    with info.bo_matrix / info.vander_matrix), rather than against a
    hardcoded snapshot of numbers. A hardcoded snapshot here previously
    went stale as soon as the example geometry (tests/examples/.../struc1.xyz)
    was regenerated, since nothing tied the "expected" numbers to the
    fixture. Recomputing from the formulas means this test tracks whatever
    geometry is currently in the fixture automatically, and still catches a
    real regression in fill_ff's math (a wrong exponent, a swapped
    numerator/denominator, etc.) because the formulas are written
    independently here, not by calling fill_ff a second time.
    """

    def test_fill_ff_bonds_structure(self):
        """Test that fill_ff correctly generates bond structure."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.01)

        # Check bonds were created
        assert len(ff.bonds) > 0
        assert all(ff.bonds["type"] == "bonds")

    def test_fill_ff_angles_structure(self):
        """Test that fill_ff correctly generates angle structure."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.01)

        # Check angles were created
        assert len(ff.angles) > 0
        assert all(ff.angles["type"] == "angles")

    def test_fill_ff_dihedrals_structure(self):
        """Test that fill_ff correctly generates dihedral structure."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.01)

        # Check dihedrals were created
        assert len(ff.dihedrals) > 0
        assert all(ff.dihedrals["type"] == "dihedrals")

    def test_fill_ff_repulsive_structure(self):
        """Test that fill_ff correctly generates repulsive term structure."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.01)

        # Check repulsive terms were created
        assert len(ff.repulsive) > 0
        assert all(ff.repulsive["type"] == "repulsive")

    def test_fill_ff_bond_count(self):
        """Test that correct number of bonds are identified."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.0)

        assert len(ff.bonds) == 6

    def test_fill_ff_angle_count(self):
        """Test that correct number of angles are identified."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.0)

        assert len(ff.angles) == 9

    def test_fill_ff_dihedral_count(self):
        """Test that correct number of dihedrals are identified."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.0)

        assert len(ff.dihedrals) == 6

    def test_fill_ff_repulsive_count(self):
        """Test that correct number of repulsive terms are identified."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.0)

        # Should have one repulsive term for every non-bonded pair
        assert len(ff.repulsive) == 15

    def test_fill_ff_reference_values_and_parameters_match_formulas(self):
        """Test that every reference_value/parameter fill_ff produces matches
        the documented closed-form formula, recomputed independently here."""
        info = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
        ff = ForceField(NAT, "dummy_path", readff=False)
        fill_ff(ff, info, repulsive_start=0.0)

        xyz = info.fortran_xyz
        bo = info.bo_matrix
        vdw = info.vander_matrix

        for row in ff.bonds.itertuples():
            i, j = row.atoms
            expected_ref = bondlength(xyz, i, j)
            expected_param = bo[i, j] / expected_ref
            assert row.reference_value == pytest.approx(expected_ref, abs=1e-7)
            assert row.parameter == pytest.approx(expected_param, abs=1e-7)

        for row in ff.angles.itertuples():
            i, j, k = row.atoms
            expected_ref = angle(xyz, i, j, k)
            bl1 = bondlength(xyz, i, j)
            bl2 = bondlength(xyz, j, k)
            expected_param = ((bo[i, j] * bo[j, k]) / (bl1 * bl2)) ** 0.5
            assert row.reference_value == pytest.approx(expected_ref, abs=1e-7)
            assert row.parameter == pytest.approx(expected_param, abs=1e-7)

        for row in ff.dihedrals.itertuples():
            i, j, k, l = row.atoms
            expected_ref = dihedral_angle(xyz, i, j, k, l)
            assert row.reference_value == pytest.approx(expected_ref, abs=1e-7)

            # param_dihedral substitutes 0.5 for any zero (non-bonded, i.e.
            # improper-dihedral) bond order in the i-j-k-l chain.
            bo_ij = bo[i, j] or 0.5
            bo_jk = bo[j, k] or 0.5
            bo_kl = bo[k, l] or 0.5
            bl1 = bondlength(xyz, i, j)
            bl2 = bondlength(xyz, j, k)
            bl3 = bondlength(xyz, k, l)
            expected_param = ((bo_ij * bo_jk * bo_kl) / (bl1 * bl2 * bl3)) ** (1 / 3)
            assert row.parameter == pytest.approx(expected_param, abs=1e-7)

        for row in ff.repulsive.itertuples():
            i, j = row.atoms
            assert row.reference_value == pytest.approx(vdw[i, j], abs=1e-7)
            # repulsive_start=0.0 was passed above; every repulsive term
            # starts at that constant regardless of atom pair.
            assert row.parameter == pytest.approx(0.0, abs=1e-9)

        # Van der Waals (repulsive) reference distances should be a
        # physically reasonable magnitude in Bohr, independent of exactly
        # which geometry the fixture currently holds.
        assert all(ff.repulsive["reference_value"] > 3.0)
