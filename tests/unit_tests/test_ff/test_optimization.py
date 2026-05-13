from ffits.forcefield.python_interface.optimization import (
    optimize_with_forcefield,
    optimize_with_forcefield_from_file,
)
from ffits.external.molbar import anc_optimizer
from ffits.forcefield.python_interface.ff_energy import (
    complete_gradient,
    complete_hessian,
)
from ffits.datatype.structure_data import ForceField, StructuralInformation
from ffits.datatype.calculation_data import TSCalculationOptions
from ffits.io.reader import read_wbo_file, read_xtb_hessian, readin_xyz
from ffits.ts_guess.define_starting_parameters import fill_ff
from ffits.forcefield.python_interface.ff_energy import ForceField, energy_ff
from ffits.utils.temp_dir_manager import TempDirManager
import numpy as np
import pytest
import os
import tempfile
from pathlib import Path
from tests.test_utils import reactant_structure_main


@pytest.fixture
def test_molecule_data():
    """Load test molecule data (small_single_molecule example)."""
    data = reactant_structure_main()
    return {"info": data.info, "ff": data.ff, "nat": data.ff.nat}


def test_optimize_with_forcefield_without_calcoptions(test_molecule_data):
    """Test that optimize_with_forcefield works without given calculation options."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        converged, energy, final_geom = optimize_with_forcefield(
            test_molecule_data["info"],
            test_molecule_data["ff"],
            optimizer=anc_optimizer,
            trajectory_filename=f"{temp_path}/test_trajectory.xyz",
            final_geometry_filename=f"{temp_path}/test_optimized.xyz",
            opt_stdout_filename=f"{temp_path}/test_optimization.out",
        )

        assert converged is True, "Optimization did not converge."
        assert isinstance(energy, float), "Energy should be a float."
        assert final_geom.shape == (7, 3), "Final geometry has incorrect shape."
        assert energy <= 0.1, "Energy is unexpectedly high."


def test_optimize_with_forcefield_from_file_without_calcoptions(test_molecule_data):
    """Test that optimize_with_forcefield works without given calculation options."""

    path1 = os.path.join(os.getcwd(), "tests/examples/small_single_molecule")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        converged, energy, final_geom = optimize_with_forcefield_from_file(
            path1 + "/struc1.xyz",
            path1 + "/ff1.csv",
            optimizer=anc_optimizer,
            trajectory_filename=f"{temp_path}/test_trajectory.xyz",
            final_geometry_filename=f"{temp_path}/test_optimized.xyz",
            opt_stdout_filename=f"{temp_path}/test_optimization.out",
        )

        assert converged is True, "Optimization did not converge."
        assert isinstance(energy, float), "Energy should be a float."
        assert final_geom.shape == (7, 3), "Final geometry has incorrect shape."
        assert energy <= 0.1, "Energy is unexpectedly high."
