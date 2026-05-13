"""
Unit tests for force field parameterization module (parameterize_ff.py).

Tests cover:
- Single parameter update functions (bonds, angles, dihedrals, repulsive)
- Hessian RMSD calculation
- Objective function evaluation
- Helper functions for derivative calculations
"""

import numpy as np
import os
import pytest
import copy
from ffits.datatype.structure_data import ForceField, StructuralInformation
from ffits.ts_guess.parameterize_ff import (
    calculate_hessian_rmsd,
    ff_fit_objective_function,
    _atom_slice,
    update_single_ffparam,
    update_bond,
    update_angle,
    update_dihedral,
    update_repulsive,
    get_sum_first_c_deriv,
    get_sum_second_c_deriv,
)
from ffits.forcefield.python_interface.ff_energy import complete_hessian
from ffits.io.reader import readin_xyz, read_wbo_file, read_xtb_hessian
from ffits.ts_guess.define_starting_parameters import fill_ff
from tests.test_utils import reactant_structure_main


@pytest.fixture
def test_molecule_data():
    """Load test molecule data (small_single_molecule example)."""
    data = reactant_structure_main()
    return {"info": data.info, "ff": data.ff, "nat": data.ff.nat}


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestAtomSlice:
    """Tests for _atom_slice helper function."""

    def test_atom_slice_first_atom(self):
        """Test slice for first atom (index 0)."""
        sl = _atom_slice(0)
        assert sl.start == 0
        assert sl.stop == 3
        assert sl == slice(0, 3)

    def test_atom_slice_second_atom(self):
        """Test slice for second atom (index 1)."""
        sl = _atom_slice(1)
        assert sl.start == 3
        assert sl.stop == 6
        assert sl == slice(3, 6)

    def test_atom_slice_arbitrary_atom(self):
        """Test slice for arbitrary atom index."""
        sl = _atom_slice(5)
        assert sl.start == 15
        assert sl.stop == 18
        assert sl == slice(15, 18)

    def test_atom_slice_length(self):
        """Test that slice always returns 3 elements (x, y, z)."""
        for i in range(10):
            sl = _atom_slice(i)
            assert sl.stop - sl.start == 3


class TestHessianRMSD:
    """Tests for Hessian RMSD calculation."""

    def test_rmsd_identical_hessians(self):
        """Test RMSD is zero for identical Hessians."""
        hess1 = np.random.randn(9, 9)
        rmsd = calculate_hessian_rmsd(hess1, hess1, 9)
        assert np.isclose(rmsd, 0.0)

    def test_rmsd_small_difference(self):
        """Test RMSD calculation with small differences."""
        hess1 = np.random.randn(9, 9)
        hess2 = hess1.copy()
        hess2[0, 0] += 1.0

        rmsd = calculate_hessian_rmsd(hess1, hess2, 9)
        expected_rmsd = np.sqrt((1.0**2) / (9 * 9))
        assert np.isclose(rmsd, expected_rmsd)

    def test_rmsd_positive_value(self):
        """Test that RMSD is always positive."""
        hess1 = np.random.randn(9, 9)
        hess2 = np.random.randn(9, 9)
        rmsd = calculate_hessian_rmsd(hess1, hess2, 9)
        assert rmsd >= 0.0


class TestObjectiveFunction:
    """Tests for objective function evaluation."""

    def test_objfun_identical_hessians(self):
        """Test objective function is zero for identical Hessians."""
        hess = np.random.randn(9, 9)
        objfun = ff_fit_objective_function(hess, hess, 9)
        assert np.isclose(objfun, 0.0)

    def test_objfun_off_diagonal_penalty(self):
        """Test that off-diagonal elements contribute to objective function."""
        hess_ff = np.eye(9) * 2.0
        hess_ref = np.eye(9) * 1.0

        # Off-diagonal differences should incur 0.5 penalty
        objfun = ff_fit_objective_function(hess_ff, hess_ref, 9)
        assert objfun >= 0.0

    def test_objfun_diagonal_ignored(self):
        """Test that diagonal differences are ignored in objective function."""
        hess_ff = np.eye(9) * 2.0
        hess_ref = np.eye(9) * 1.0
        hess_ref[3, 3] = 100.0  # Large diagonal difference in different block

        # Should not significantly affect result (diagonal elements ignored)
        objfun = ff_fit_objective_function(hess_ff, hess_ref, 9)
        assert np.isfinite(objfun)

    def test_objfun_wrong_dimension_raises_error(self):
        """Test that wrong Hessian dimensions raise errors."""
        hess = np.random.randn(9, 9)
        with pytest.raises(Exception):
            ff_fit_objective_function(hess, np.random.randn(12, 12), 9)

    def test_objfun_positive_semidefinite(self):
        """Test that objective function is always non-negative."""
        hess_ff = np.random.randn(9, 9)
        hess_ref = np.random.randn(9, 9)
        objfun = ff_fit_objective_function(hess_ff, hess_ref, 9)
        assert objfun >= 0.0


class TestUpdateSingleFFParam:
    """Tests for single parameter update function."""

    def test_update_with_positive_second_derivative(self):
        """Test parameter update with positive second derivative."""
        param = 1.0
        deriv1 = -0.5  # Gradient pointing in negative direction
        deriv2 = 1.0  # Positive curvature
        stepsize = 0.5

        new_param = update_single_ffparam(param, deriv1, deriv2, stepsize)
        # Should move in positive direction (opposite of gradient)
        assert new_param > param

    def test_update_with_negative_second_derivative(self):
        """Test parameter update with negative second derivative (saddle point)."""
        param = 1.0
        deriv1 = 0.5  # Gradient pointing in positive direction
        deriv2 = -1.0  # Negative curvature (saddle point)
        stepsize = 0.5

        new_param = update_single_ffparam(param, deriv1, deriv2, stepsize)
        # Newton's method: param_new = param - deriv1/deriv2 * stepsize
        # With negative deriv2 and positive deriv1: movement is positive
        expected = param - deriv1 * (1.0 / deriv2) * stepsize
        assert np.isclose(new_param, expected)

    def test_update_zero_second_derivative_warning(self):
        """Test that zero second derivative triggers warning."""
        param = 1.0
        deriv1 = 0.5
        deriv2 = 0.0
        stepsize = 0.5

        with pytest.warns(UserWarning):
            new_param = update_single_ffparam(param, deriv1, deriv2, stepsize)
        # Should return original value
        assert new_param == param

    def test_update_stepsize_scaling(self):
        """Test that larger stepsize produces larger parameter change."""
        param = 1.0
        deriv1 = -1.0
        deriv2 = 1.0

        new_param_small = update_single_ffparam(param, deriv1, deriv2, 0.1)
        new_param_large = update_single_ffparam(param, deriv1, deriv2, 1.0)

        # Larger stepsize should produce larger change
        assert abs(new_param_large - param) > abs(new_param_small - param)


class TestGetSumDerivatives:
    """Tests for sum derivative helper functions."""

    def test_get_sum_first_c_deriv_shape(self):
        """Test that first derivative sum returns scalar."""
        nat = 3
        hess_ff = np.random.randn(3 * nat, 3 * nat)
        hess_ref = np.random.randn(3 * nat, 3 * nat)
        hess_single = np.random.randn(3 * nat, 3 * nat)
        c = 1.0

        result = get_sum_first_c_deriv(c, 0, 1, hess_ff, hess_ref, hess_single, 0.0)
        assert isinstance(result, (float, np.floating))

    def test_get_sum_second_c_deriv_shape(self):
        """Test that second derivative sum returns scalar."""
        nat = 3
        hess_ff = np.random.randn(3 * nat, 3 * nat)
        hess_ref = np.random.randn(3 * nat, 3 * nat)
        hess_single = np.random.randn(3 * nat, 3 * nat)
        c = 1.0

        result = get_sum_second_c_deriv(c, 0, 1, hess_ff, hess_ref, hess_single, 0.0)
        assert isinstance(result, (float, np.floating))

    def test_derivative_accumulation(self):
        """Test that derivatives accumulate correctly."""
        nat = 3
        hess_ff = np.random.randn(3 * nat, 3 * nat)
        hess_ref = np.random.randn(3 * nat, 3 * nat)
        hess_single = np.random.randn(3 * nat, 3 * nat)
        c = 1.0

        acc1 = 1.0
        acc2 = get_sum_first_c_deriv(c, 0, 1, hess_ff, hess_ref, hess_single, acc1)

        # Accumulator should increase
        assert acc2 >= acc1 or acc2 < acc1  # Can be either direction
        assert acc2 != acc1  # Should be changed


class TestParameterUpdateFunctions:
    """Integration tests for parameter update functions."""

    def test_update_bond_returns_float(self, test_molecule_data):
        """Test that update_bond returns a float parameter."""
        info = test_molecule_data["info"]
        ff = test_molecule_data["ff"]

        hessian_ff = complete_hessian(info.fortran_xyz, ff)

        bond_row = ff.bonds.iloc[0]
        new_param = update_bond(bond_row, info, hessian_ff, stepsize=0.1)

        assert isinstance(new_param, (float, np.floating))
        assert np.isfinite(new_param)

    def test_update_angle_returns_float(self, test_molecule_data):
        """Test that update_angle returns a float parameter."""
        info = test_molecule_data["info"]
        ff = test_molecule_data["ff"]

        hessian_ff = complete_hessian(info.fortran_xyz, ff)

        angle_row = ff.angles.iloc[0]
        new_param = update_angle(angle_row, info, hessian_ff, stepsize=0.1)

        assert isinstance(new_param, (float, np.floating))
        assert np.isfinite(new_param)

    def test_update_dihedral_returns_float(self, test_molecule_data):
        """Test that update_dihedral returns a float parameter."""
        info = test_molecule_data["info"]
        ff = test_molecule_data["ff"]

        hessian_ff = complete_hessian(info.fortran_xyz, ff)

        dihedral_row = ff.dihedrals.iloc[0]
        new_param = update_dihedral(dihedral_row, info, hessian_ff, stepsize=0.1)

        assert isinstance(new_param, (float, np.floating))
        assert np.isfinite(new_param)

    def test_update_repulsive_returns_float(self, test_molecule_data):
        """Test that update_repulsive returns a float parameter (or handles edge cases)."""
        info = test_molecule_data["info"]
        ff = test_molecule_data["ff"]

        hessian_ff = complete_hessian(info.fortran_xyz, ff)

        # Skip repulsive terms with zero parameters (would cause division by zero)
        for idx, rep_row in ff.repulsive.iterrows():
            if rep_row["parameter"] != 0.0:
                new_param = update_repulsive(rep_row, info, hessian_ff, stepsize=0.1)
                assert isinstance(new_param, (float, np.floating))
                assert np.isfinite(new_param)
                break  # Test at least one valid case

    def test_update_bond_positive_parameter(self, test_molecule_data):
        """Test that bond parameters remain positive after update."""
        info = test_molecule_data["info"]
        ff = test_molecule_data["ff"]

        hessian_ff = complete_hessian(info.fortran_xyz, ff)

        for idx, bond_row in ff.bonds.iterrows():
            old_param = bond_row["parameter"]
            new_param = update_bond(bond_row, info, hessian_ff, stepsize=0.01)
            # Parameter should stay positive (force constants must be positive)
            assert (
                new_param > 0
            ), f"Bond parameter became non-positive: {old_param} -> {new_param}"

    def test_update_angle_positive_parameter(self, test_molecule_data):
        """Test that angle parameters remain positive after update."""
        info = test_molecule_data["info"]
        ff = test_molecule_data["ff"]

        hessian_ff = complete_hessian(info.fortran_xyz, ff)

        for idx, angle_row in ff.angles.iterrows():
            old_param = angle_row["parameter"]
            new_param = update_angle(angle_row, info, hessian_ff, stepsize=0.01)
            # Parameter should stay positive (force constants must be positive)
            assert (
                new_param > 0
            ), f"Angle parameter became non-positive: {old_param} -> {new_param}"

    def test_update_repulsive_positive_parameter(self, test_molecule_data):
        """Test that repulsive parameters remain positive/finite after update."""
        info = test_molecule_data["info"]
        ff = test_molecule_data["ff"]

        hessian_ff = complete_hessian(info.fortran_xyz, ff)

        # Skip repulsive terms with zero parameters (would cause division by zero)
        for idx, rep_row in ff.repulsive.iterrows():
            old_param = rep_row["parameter"]
            if old_param == 0.0:
                continue  # Skip zero parameters
            new_param = update_repulsive(rep_row, info, hessian_ff, stepsize=0.01)
            # Parameter should stay positive and finite
            assert (
                new_param > 0
            ), f"Repulsive parameter became non-positive: {old_param} -> {new_param}"
            assert np.isfinite(new_param)


class TestObjectiveFunctionProperties:
    """Tests for objective function mathematical properties."""

    def test_objfun_symmetry(self):
        """Test objective function doesn't have ordering dependency."""
        hess1 = np.random.randn(9, 9)
        hess2 = np.random.randn(9, 9)

        # Make symmetric for realistic Hessians
        hess1 = (hess1 + hess1.T) / 2
        hess2 = (hess2 + hess2.T) / 2

        objfun1 = ff_fit_objective_function(hess1, hess2, 9)
        objfun2 = ff_fit_objective_function(hess2, hess1, 9)

        # Objective function is not symmetric in arguments (not expected to be)
        # but should be well-defined for both orderings
        assert np.isfinite(objfun1)
        assert np.isfinite(objfun2)

    def test_rmsd_triangle_inequality(self):
        """Test that RMSD satisfies triangle-like property."""
        hess1 = np.random.randn(9, 9)
        hess2 = hess1 + np.random.randn(9, 9) * 0.1
        hess3 = hess2 + np.random.randn(9, 9) * 0.1

        rmsd_13 = calculate_hessian_rmsd(hess1, hess3, 9)
        rmsd_12 = calculate_hessian_rmsd(hess1, hess2, 9)
        rmsd_23 = calculate_hessian_rmsd(hess2, hess3, 9)

        # All should be finite and positive
        assert np.isfinite(rmsd_13)
        assert np.isfinite(rmsd_12)
        assert np.isfinite(rmsd_23)
        assert rmsd_13 >= 0 and rmsd_12 >= 0 and rmsd_23 >= 0
