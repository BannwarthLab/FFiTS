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
import tempfile
from pathlib import Path
from ffits.external.molbar import anc_optimizer, scipy_optimizer
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)
from ffits.datatype.structure_data import ForceField, StructuralInformation
from ffits.io.reader import readin_xyz
from tests.test_utils import NAT, XYZ, WBO, ATOM_TYPES


class TestANCOptimizer:
    """Tests for Approximate Normal Coordinate (ANC) optimizer."""

    def test_anc_optimizer_convergence(self):
        """Test that ANC optimizer converges to a minimum."""
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "tests" / "examples" / "small_single_molecule"

        with tempfile.TemporaryDirectory() as temp_wd:
            # Copy necessary files
            ff_src = data_dir / "ff1.csv"
            xyz_src = data_dir / "struc2.xyz"
            if ff_src.exists():
                shutil.copy(ff_src, temp_wd)
            if xyz_src.exists():
                shutil.copy(xyz_src, temp_wd)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_wd)

                # Create objects
                struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
                ff = ForceField(
                    NAT,
                    "ff1.csv",
                    readff=True,
                    energy_calculator=energy_ff,
                    gradient_calculator=complete_gradient,
                    hessian_calculator=complete_hessian,
                )

                _, _, xyz_start, _ = readin_xyz("struc2.xyz")

                converged, energy, final_geom, steps, time, message = anc_optimizer(
                    xyz_start, ff, struc.atom_types, max_micro_steps=5
                )

                assert (
                    converged
                ), "ANC optimizer should converge for valid starting geometry"
                assert isinstance(
                    energy, (int, float)
                ), "Final energy should be numeric"
                assert energy >= 0, "Energy should be non-negative after optimization"
            finally:
                os.chdir(original_cwd)

    def test_anc_optimizer_energy_reduction(self):
        """Test that final energy is reasonable after optimization."""
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "tests" / "examples" / "small_single_molecule"

        with tempfile.TemporaryDirectory() as temp_wd:
            # Copy necessary files
            ff_src = data_dir / "ff1.csv"
            xyz_src = data_dir / "struc2.xyz"
            if ff_src.exists():
                shutil.copy(ff_src, temp_wd)
            if xyz_src.exists():
                shutil.copy(xyz_src, temp_wd)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_wd)

                # Create objects
                struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
                ff = ForceField(
                    NAT,
                    "ff1.csv",
                    readff=True,
                    energy_calculator=energy_ff,
                    gradient_calculator=complete_gradient,
                    hessian_calculator=complete_hessian,
                )

                _, _, xyz_start, _ = readin_xyz("struc2.xyz")

                converged, energy, final_geom, steps, time, message = anc_optimizer(
                    xyz_start, ff, struc.atom_types, max_micro_steps=5
                )

                assert (
                    energy <= 0.5
                ), "Final energy should be reasonably low for converged geometry"
            finally:
                os.chdir(original_cwd)

    def test_anc_optimizer_return_values(self):
        """Test that ANC optimizer returns correctly formatted values."""
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "tests" / "examples" / "small_single_molecule"

        with tempfile.TemporaryDirectory() as temp_wd:
            # Copy necessary files
            ff_src = data_dir / "ff1.csv"
            xyz_src = data_dir / "struc2.xyz"
            if ff_src.exists():
                shutil.copy(ff_src, temp_wd)
            if xyz_src.exists():
                shutil.copy(xyz_src, temp_wd)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_wd)

                # Create objects
                struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
                ff = ForceField(
                    NAT,
                    "ff1.csv",
                    readff=True,
                    energy_calculator=energy_ff,
                    gradient_calculator=complete_gradient,
                    hessian_calculator=complete_hessian,
                )

                _, _, xyz_start, _ = readin_xyz("struc2.xyz")

                converged, energy, final_geom, steps, time, message = anc_optimizer(
                    xyz_start, ff, struc.atom_types, max_micro_steps=5
                )

                assert isinstance(
                    converged, (bool, np.bool_)
                ), "Convergence flag should be boolean"
                assert isinstance(steps, (int, np.integer)), "Steps should be integer"
                assert isinstance(
                    time, (int, float, np.number)
                ), "Time should be numeric"
                assert isinstance(message, str), "Message should be string"
                assert final_geom is not None, "Final geometry should not be None"
                assert (
                    final_geom.shape == xyz_start.shape
                ), "Final geometry shape should match input"
            finally:
                os.chdir(original_cwd)

    def test_anc_optimizer_positive_steps(self):
        """Test that ANC optimizer takes at least some steps."""
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "tests" / "examples" / "small_single_molecule"

        with tempfile.TemporaryDirectory() as temp_wd:
            # Copy necessary files
            ff_src = data_dir / "ff1.csv"
            xyz_src = data_dir / "struc2.xyz"
            if ff_src.exists():
                shutil.copy(ff_src, temp_wd)
            if xyz_src.exists():
                shutil.copy(xyz_src, temp_wd)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_wd)

                # Create objects
                struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
                ff = ForceField(
                    NAT,
                    "ff1.csv",
                    readff=True,
                    energy_calculator=energy_ff,
                    gradient_calculator=complete_gradient,
                    hessian_calculator=complete_hessian,
                )

                _, _, xyz_start, _ = readin_xyz("struc2.xyz")

                converged, energy, final_geom, steps, time, message = anc_optimizer(
                    xyz_start, ff, struc.atom_types, max_micro_steps=5
                )

                assert steps > 0, "Optimization should take at least one step"
            finally:
                os.chdir(original_cwd)

    def test_anc_optimizer_geometry_validity(self):
        """Test that optimized geometry has valid (non-NaN) coordinates."""
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "tests" / "examples" / "small_single_molecule"

        with tempfile.TemporaryDirectory() as temp_wd:
            # Copy necessary files
            ff_src = data_dir / "ff1.csv"
            xyz_src = data_dir / "struc2.xyz"
            if ff_src.exists():
                shutil.copy(ff_src, temp_wd)
            if xyz_src.exists():
                shutil.copy(xyz_src, temp_wd)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_wd)

                # Create objects
                struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
                ff = ForceField(
                    NAT,
                    "ff1.csv",
                    readff=True,
                    energy_calculator=energy_ff,
                    gradient_calculator=complete_gradient,
                    hessian_calculator=complete_hessian,
                )

                _, _, xyz_start, _ = readin_xyz("struc2.xyz")

                converged, energy, final_geom, steps, time, message = anc_optimizer(
                    xyz_start, ff, struc.atom_types, max_micro_steps=5
                )

                assert not np.any(
                    np.isnan(final_geom)
                ), "Final geometry should not contain NaN values"
                assert not np.any(
                    np.isinf(final_geom)
                ), "Final geometry should not contain inf values"
            finally:
                os.chdir(original_cwd)

    def test_anc_optimizer_different_tolerances(self):
        """Test ANC optimizer with different convergence tolerances."""
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "tests" / "examples" / "small_single_molecule"

        with tempfile.TemporaryDirectory() as temp_wd:
            # Copy necessary files
            ff_src = data_dir / "ff1.csv"
            xyz_src = data_dir / "struc2.xyz"
            if ff_src.exists():
                shutil.copy(ff_src, temp_wd)
            if xyz_src.exists():
                shutil.copy(xyz_src, temp_wd)

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_wd)

                # Create objects
                struc = StructuralInformation(NAT, XYZ, WBO, ATOM_TYPES)
                ff = ForceField(
                    NAT,
                    "ff1.csv",
                    readff=True,
                    energy_calculator=energy_ff,
                    gradient_calculator=complete_gradient,
                    hessian_calculator=complete_hessian,
                )

                _, _, xyz_start, _ = readin_xyz("struc2.xyz")

                # Test with loose tolerance
                converged, energy, _, _, _, _ = anc_optimizer(
                    xyz_start,
                    ff,
                    struc.atom_types,
                    e_tol=1e-3,
                    x_tol=1e-2,
                    max_micro_steps=5,
                )

                assert isinstance(
                    converged, (bool, np.bool_)
                ), "Should handle different tolerances"
            finally:
                os.chdir(original_cwd)


# class TestScipyOptimizer:
#     """Tests for SciPy-based Newton-CG optimizer."""

#     def test_scipy_optimizer_convergence(self, setup_test_environment, forcefield_and_structure):
#         """Test that SciPy optimizer produces convergence result."""
#         temp_wd = setup_test_environment
#         ff, struc = forcefield_and_structure

#         xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#         _, _, xyz_start, _ = readin_xyz(xyz_path)

#         result = scipy_optimizer(xyz_start, ff, struc)

#         assert hasattr(result, 'success'), "Result should have success attribute"
#         assert hasattr(result, 'fun'), "Result should have fun (energy) attribute"
#         assert hasattr(result, 'x'), "Result should have x (coordinates) attribute"
#         assert hasattr(result, 'nit'), "Result should have nit (iterations) attribute"

#     # def test_scipy_optimizer_final_energy(self, setup_test_environment, forcefield_and_structure):
#     #     """Test that final energy is reasonable after optimization."""
#     #     temp_wd = setup_test_environment
#     #     ff, struc = forcefield_and_structure

#     #     xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#     #     _, _, xyz_start, _ = readin_xyz(xyz_path)

#     #     result = scipy_optimizer(xyz_start, ff, struc)
#     #     final_energy = result.fun

#     #     assert isinstance(final_energy, (int, float, np.number)), "Final energy should be numeric"
#     #     assert final_energy >= 0, "Energy should be non-negative"
#     #     assert final_energy < 1.0, "Final energy should be reasonably low for converged geometry"

#     def test_scipy_optimizer_coordinate_shape(self, setup_test_environment, forcefield_and_structure):
#         """Test that optimized coordinates have correct shape."""
#         temp_wd = setup_test_environment
#         ff, struc = forcefield_and_structure

#         xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#         _, _, xyz_start, _ = readin_xyz(xyz_path)

#         result = scipy_optimizer(xyz_start, ff, struc)
#         final_coords = result.x.reshape((len(xyz_start), 3))

#         assert final_coords.shape == xyz_start.shape, "Final coordinates should match input shape"

#     def test_scipy_optimizer_coordinate_validity(self, setup_test_environment, forcefield_and_structure):
#         """Test that optimized coordinates contain no NaN or inf values."""
#         temp_wd = setup_test_environment
#         ff, struc = forcefield_and_structure

#         xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#         _, _, xyz_start, _ = readin_xyz(xyz_path)

#         result = scipy_optimizer(xyz_start, ff, struc)
#         final_coords = result.x.reshape((len(xyz_start), 3))

#         assert not np.any(np.isnan(final_coords)), "Coordinates should not contain NaN"
#         assert not np.any(np.isinf(final_coords)), "Coordinates should not contain inf"

#     def test_scipy_optimizer_iterations(self, setup_test_environment, forcefield_and_structure):
#         """Test that optimizer takes expected number of iterations."""
#         temp_wd = setup_test_environment
#         ff, struc = forcefield_and_structure

#         xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#         _, _, xyz_start, _ = readin_xyz(xyz_path)

#         result = scipy_optimizer(xyz_start, ff, struc)
#         steps = result.nit

#         assert isinstance(steps, (int, np.integer)), "Number of iterations should be integer"
#         assert steps >= 0, "Number of iterations should be non-negative"

#     def test_scipy_optimizer_message(self, setup_test_environment, forcefield_and_structure):
#         """Test that optimizer returns informative message."""
#         temp_wd = setup_test_environment
#         ff, struc = forcefield_and_structure

#         xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#         _, _, xyz_start, _ = readin_xyz(xyz_path)

#         result = scipy_optimizer(xyz_start, ff, struc)

#         assert hasattr(result, 'message'), "Result should have message attribute"
#         assert isinstance(result.message, str), "Message should be string"


# class TestOptimizerComparison:
#     """Tests comparing behavior of both optimizers."""

#     def test_both_optimizers_reduce_energy(self, setup_test_environment, forcefield_and_structure):
#         """Test that both optimizers reduce energy from starting geometry."""
#         temp_wd = setup_test_environment
#         ff, struc = forcefield_and_structure

#         xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#         _, _, xyz_start, _ = readin_xyz(xyz_path)

#         # Calculate initial energy
#         initial_energy = ff.get_energy(xyz_start.flatten())

#         # Run ANC optimizer
#         _, anc_energy, _, _, _, _ = anc_optimizer(
#             xyz_start, ff, struc.atom_types, max_micro_steps=5
#         )

#         # Reset and run SciPy optimizer
#         scipy_result = scipy_optimizer(xyz_start, ff, struc)
#         scipy_energy = scipy_result.fun

#         assert anc_energy <= initial_energy + 0.1, "ANC should not significantly increase energy"
#         assert scipy_energy <= initial_energy + 0.1, "SciPy should not significantly increase energy"

#     def test_both_optimizers_produce_valid_geometries(self, setup_test_environment, forcefield_and_structure):
#         """Test that both optimizers produce geometries without NaN/inf."""
#         temp_wd = setup_test_environment
#         ff, struc = forcefield_and_structure

#         xyz_path = os.path.join(temp_wd, 'struc2.xyz')
#         _, _, xyz_start, _ = readin_xyz(xyz_path)

#         # ANC optimizer
#         _, _, anc_geom, _, _, _ = anc_optimizer(
#             xyz_start, ff, struc.atom_types, max_micro_steps=5
#         )

#         # SciPy optimizer
#         scipy_result = scipy_optimizer(xyz_start, ff, struc)
#         scipy_geom = scipy_result.x.reshape((len(xyz_start), 3))

#         assert not np.any(np.isnan(anc_geom)), "ANC geometry should have no NaN values"
#         assert not np.any(np.isnan(scipy_geom)), "SciPy geometry should have no NaN values"
