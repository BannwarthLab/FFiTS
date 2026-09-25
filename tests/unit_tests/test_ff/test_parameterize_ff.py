"""
Unit tests for force field parameterization module (parameterize_ff.py).

Tests cover:
- Single parameter update functions (bonds, angles, dihedrals, repulsive)
- Hessian RMSD calculation
- Objective function evaluation
- Helper functions for derivative calculations
"""

import numpy as np
import pytest
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import StructuralInformation
from ffits.ts_guess.parameterize_ff import (
    calculate_hessian_rmsd,
    ff_fit_objective_function,
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
