from ffits.ts_guess.guess import get_ts_guess
from ffits.datatype.structure_data import (
    ForceField,
    Structure,
)
import numpy as np
import pytest
import os
import tempfile
from pathlib import Path
from tests.test_utils import reactant_structure_main, product_structure_main


@pytest.fixture
def test_molecule_data1():
    """Load test molecule data (small_single_molecule example)."""
    struc = reactant_structure_main()
    return {"struc": struc, "info": struc.info, "ff": struc.ff, "nat": struc.ff.nat}


@pytest.fixture
def test_molecule_data2():
    """Load test molecule data (small_single_molecule example)."""
    struc = product_structure_main()
    return {"struc": struc, "info": struc.info, "ff": struc.ff, "nat": struc.ff.nat}


def test_get_ts_guess_withoutcalcdata(test_molecule_data1, test_molecule_data2):
    """Test that get_ts_guess runs without errors and returns a Structure object with the expected attributes."""
    struc1: Structure = test_molecule_data1["struc"]
    struc2: Structure = test_molecule_data2["struc"]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cwd = os.getcwd()
        os.chdir(temp_path)
        # pipe all output files to temp_path
        try:
            struc1.ff.ff_filename = str(temp_path / "ff1.csv")
            struc2.ff.ff_filename = str(temp_path / "ff2.csv")
            tsff, converged, energy, final_geom = get_ts_guess(struc1, struc2)
            assert isinstance(
                tsff, ForceField
            ), "get_ts_guess did not return a ForceField object."
            assert converged is True, "TS guess optimization did not converge."
            assert isinstance(
                energy, float
            ), "get_ts_guess did not return a float for energy."
            assert isinstance(
                final_geom, np.ndarray
            ), "get_ts_guess did not return a numpy array for final geometry."
            os.chdir(cwd)
        finally:
            os.chdir(cwd)
