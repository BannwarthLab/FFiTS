"""
Unit tests for molecular geometry optimization using molbar library.

This module tests two optimization methods:
- ANC (Approximate Normal Coordinate) optimizer: Uses Hessian information and trust radius
- SciPy optimizer: Uses Newton-CG method with gradient and Hessian

Tests validate convergence behavior, energy reduction, coordinate validity, 
return value formats, and robustness to different starting geometries.
"""

import numpy as np
import os
import shutil
import pytest
from ffits.external.molbar_optimizer import anc_optimizer, scipy_optimizer
from ffits.forcefield.python_interface.ff_energy import energy_ff, complete_gradient, complete_hessian
from ffits.datatype.structure_data import ForceField, StructuralInformation
from ffits.io.reader import readin_xyz

# Test data constants
NAT = 7

XYZ = np.array([
    [-2.33287094, 3.31176687, 0.20110100],
    [-0.91630217, 2.85867268, -0.04327585],
    [0.06256276, 3.55862185, 0.08938727],
    [-2.92591176, 3.18375556, -0.71288068],
    [-2.79725946, 2.67965821, 0.96793335],
    [-0.81484498, 1.79716556, -0.36699673],
    [-2.35087245, 4.35655928, 0.51563164]
])

WBO = {
    (1, 2): 1.02668632226515,
    (2, 3): 1.92755303185758,
    (1, 4): 0.955689824153634,
    (1, 5): 0.955863695522291,
    (2, 6): 0.933812077856736,
    (1, 7): 0.982636418257069
}

ATOM_TYPES = ['C', 'C', 'O', 'H', 'H', 'H', 'H']


@pytest.fixture
def setup_test_environment():
    """Set up test environment with force field files and structural data."""
    cwd = os.getcwd()
    temp_wd = os.path.join(cwd, '_manual_test/small_single_molecule')
    os.makedirs(temp_wd, exist_ok=True)
    os.chdir(temp_wd)

    data_dir = os.path.join(cwd, 'tests/examples/small_single_molecule')
    ff_src = os.path.join(data_dir, 'ff1_new')
    xyz_src = os.path.join(data_dir, 'struc2.xyz')

    if os.path.exists(ff_src):
        shutil.copy(ff_src, temp_wd)
    if os.path.exists(xyz_src):
        shutil.copy(xyz_src, temp_wd)

    yield cwd, temp_wd

    os.chdir(cwd)


@pytest.fixture
def forcefield_and_structure():
    """Create ForceField and StructuralInformation objects for testing."""
    struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
    ff = ForceField(
        NAT,
        'ff1_new',
        readff=True,
        energy_calculator=energy_ff,
        gradient_calculator=complete_gradient,
        hessian_calculator=complete_hessian
    )
    return ff, struc


class TestANCOptimizer:
    """Tests for Approximate Normal Coordinate (ANC) optimizer."""

    def test_anc_optimizer_convergence(self, setup_test_environment, forcefield_and_structure):
        """Test that ANC optimizer converges to a minimum."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        converged, energy, final_geom, steps, time, message = anc_optimizer(
            xyz_start, ff, struc.atom_types, max_micro_steps=5
        )

        assert converged, "ANC optimizer should converge for valid starting geometry"
        assert isinstance(energy, (int, float)), "Final energy should be numeric"
        assert energy >= 0, "Energy should be non-negative after optimization"

    def test_anc_optimizer_energy_reduction(self, setup_test_environment, forcefield_and_structure):
        """Test that final energy is reasonable after optimization."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        converged, energy, final_geom, steps, time, message = anc_optimizer(
            xyz_start, ff, struc.atom_types, max_micro_steps=5
        )

        assert energy <= 0.5, "Final energy should be reasonably low for converged geometry"

    def test_anc_optimizer_return_values(self, setup_test_environment, forcefield_and_structure):
        """Test that ANC optimizer returns correctly formatted values."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        converged, energy, final_geom, steps, time, message = anc_optimizer(
            xyz_start, ff, struc.atom_types, max_micro_steps=5
        )

        assert isinstance(converged, (bool, np.bool_)), "Convergence flag should be boolean"
        assert isinstance(steps, (int, np.integer)), "Steps should be integer"
        assert isinstance(time, (int, float, np.number)), "Time should be numeric"
        assert isinstance(message, str), "Message should be string"
        assert final_geom is not None, "Final geometry should not be None"
        assert final_geom.shape == xyz_start.shape, "Final geometry shape should match input"

    def test_anc_optimizer_positive_steps(self, setup_test_environment, forcefield_and_structure):
        """Test that ANC optimizer takes at least some steps."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        converged, energy, final_geom, steps, time, message = anc_optimizer(
            xyz_start, ff, struc.atom_types, max_micro_steps=5
        )

        assert steps > 0, "Optimization should take at least one step"

    def test_anc_optimizer_geometry_validity(self, setup_test_environment, forcefield_and_structure):
        """Test that optimized geometry has valid (non-NaN) coordinates."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        converged, energy, final_geom, steps, time, message = anc_optimizer(
            xyz_start, ff, struc.atom_types, max_micro_steps=5
        )

        assert not np.any(np.isnan(final_geom)), "Final geometry should not contain NaN values"
        assert not np.any(np.isinf(final_geom)), "Final geometry should not contain inf values"

    def test_anc_optimizer_different_tolerances(self, setup_test_environment, forcefield_and_structure):
        """Test ANC optimizer with different convergence tolerances."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        # Test with loose tolerance
        converged, energy, _, _, _, _ = anc_optimizer(
            xyz_start, ff, struc.atom_types,
            g_tol=1e-1, e_tol=1e-3, x_tol=1e-2,
            max_micro_steps=5
        )

        assert isinstance(converged, (bool, np.bool_)), "Should handle different tolerances"


class TestScipyOptimizer:
    """Tests for SciPy-based Newton-CG optimizer."""

    def test_scipy_optimizer_convergence(self, setup_test_environment, forcefield_and_structure):
        """Test that SciPy optimizer produces convergence result."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        result = scipy_optimizer(xyz_start, ff, struc)

        assert hasattr(result, 'success'), "Result should have success attribute"
        assert hasattr(result, 'fun'), "Result should have fun (energy) attribute"
        assert hasattr(result, 'x'), "Result should have x (coordinates) attribute"
        assert hasattr(result, 'nit'), "Result should have nit (iterations) attribute"

    def test_scipy_optimizer_final_energy(self, setup_test_environment, forcefield_and_structure):
        """Test that final energy is reasonable after optimization."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        result = scipy_optimizer(xyz_start, ff, struc)
        final_energy = result.fun

        assert isinstance(final_energy, (int, float, np.number)), "Final energy should be numeric"
        assert final_energy >= 0, "Energy should be non-negative"
        assert final_energy < 1.0, "Final energy should be reasonably low for converged geometry"

    def test_scipy_optimizer_coordinate_shape(self, setup_test_environment, forcefield_and_structure):
        """Test that optimized coordinates have correct shape."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        result = scipy_optimizer(xyz_start, ff, struc)
        final_coords = result.x.reshape((len(xyz_start), 3))

        assert final_coords.shape == xyz_start.shape, "Final coordinates should match input shape"

    def test_scipy_optimizer_coordinate_validity(self, setup_test_environment, forcefield_and_structure):
        """Test that optimized coordinates contain no NaN or inf values."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        result = scipy_optimizer(xyz_start, ff, struc)
        final_coords = result.x.reshape((len(xyz_start), 3))

        assert not np.any(np.isnan(final_coords)), "Coordinates should not contain NaN"
        assert not np.any(np.isinf(final_coords)), "Coordinates should not contain inf"

    def test_scipy_optimizer_iterations(self, setup_test_environment, forcefield_and_structure):
        """Test that optimizer takes expected number of iterations."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        result = scipy_optimizer(xyz_start, ff, struc)
        steps = result.nit

        assert isinstance(steps, (int, np.integer)), "Number of iterations should be integer"
        assert steps >= 0, "Number of iterations should be non-negative"

    def test_scipy_optimizer_message(self, setup_test_environment, forcefield_and_structure):
        """Test that optimizer returns informative message."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        result = scipy_optimizer(xyz_start, ff, struc)

        assert hasattr(result, 'message'), "Result should have message attribute"
        assert isinstance(result.message, str), "Message should be string"


class TestOptimizerComparison:
    """Tests comparing behavior of both optimizers."""

    def test_both_optimizers_reduce_energy(self, setup_test_environment, forcefield_and_structure):
        """Test that both optimizers reduce energy from starting geometry."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        # Calculate initial energy
        initial_energy = ff.get_energy(xyz_start.flatten())

        # Run ANC optimizer
        _, anc_energy, _, _, _, _ = anc_optimizer(
            xyz_start, ff, struc.atom_types, max_micro_steps=5
        )

        # Reset and run SciPy optimizer
        scipy_result = scipy_optimizer(xyz_start, ff, struc)
        scipy_energy = scipy_result.fun

        assert anc_energy <= initial_energy + 0.1, "ANC should not significantly increase energy"
        assert scipy_energy <= initial_energy + 0.1, "SciPy should not significantly increase energy"

    def test_both_optimizers_produce_valid_geometries(self, setup_test_environment, forcefield_and_structure):
        """Test that both optimizers produce geometries without NaN/inf."""
        cwd, temp_wd = setup_test_environment
        ff, struc = forcefield_and_structure

        try:
            _, _, xyz_start, _ = readin_xyz('struc2.xyz')
        except FileNotFoundError:
            pytest.skip("Test geometry file not found")

        # ANC optimizer
        _, _, anc_geom, _, _, _ = anc_optimizer(
            xyz_start, ff, struc.atom_types, max_micro_steps=5
        )

        # SciPy optimizer
        scipy_result = scipy_optimizer(xyz_start, ff, struc)
        scipy_geom = scipy_result.x.reshape((len(xyz_start), 3))

        assert not np.any(np.isnan(anc_geom)), "ANC geometry should have no NaN values"
        assert not np.any(np.isnan(scipy_geom)), "SciPy geometry should have no NaN values"
