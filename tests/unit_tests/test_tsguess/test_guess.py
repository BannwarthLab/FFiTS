from types import SimpleNamespace
from ffits.ts_guess.guess import get_ts_guess, _get_start_structure
from ffits.datatype.calculation_data import CalculationData
from ffits.io.reader import readin_xyz
from ffits.datatype.structure_data import (
    Structure,
)
from ffits.datatype.forcefield_data import ForceField
import numpy as np
import pytest
import os
import tempfile
from pathlib import Path
from tests.test_utils import (
    SMALL_MOLECULE_DIR,
    reactant_structure_main,
    product_structure_main,
)


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


# ---------------------------------------------------------------------------
# start structure of the TS optimization (ts_calc.start_structure)
# ---------------------------------------------------------------------------


def _start(struc1, struc2, start_structure="", perform_two=True, from_reactant=True):
    calcdata = CalculationData.from_default()
    calcdata.ts_calc.start_structure = start_structure
    calcdata.ts_calc.perform_two_optimizations = perform_two
    tsff = SimpleNamespace(start_from_reactant=from_reactant)
    return _get_start_structure(struc1, struc2, tsff, calcdata)


def test_start_structure_default_optimizes_both_sides(
    test_molecule_data1, test_molecule_data2
):
    assert _start(test_molecule_data1["struc"], test_molecule_data2["struc"]) is None


@pytest.mark.parametrize("from_reactant, side", [(True, 0), (False, 1)])
def test_start_structure_default_single_side(
    test_molecule_data1, test_molecule_data2, from_reactant, side
):
    strucs = (test_molecule_data1["struc"], test_molecule_data2["struc"])
    info = _start(*strucs, perform_two=False, from_reactant=from_reactant)
    assert info is strucs[side].info


@pytest.mark.parametrize("start, side", [("reactant", 0), ("product", 1)])
def test_start_structure_side_overrides_two_optimizations(
    test_molecule_data1, test_molecule_data2, start, side
):
    strucs = (test_molecule_data1["struc"], test_molecule_data2["struc"])
    assert _start(*strucs, start_structure=start) is strucs[side].info


def test_start_structure_from_xyz_file(test_molecule_data1, test_molecule_data2):
    nat, _, xyz, _ = readin_xyz(str(SMALL_MOLECULE_DIR / "struc2.xyz"))
    info = _start(
        test_molecule_data1["struc"],
        test_molecule_data2["struc"],
        start_structure=str(SMALL_MOLECULE_DIR / "struc2.xyz"),
    )
    assert info.nat == nat
    np.testing.assert_allclose(info.xyz, xyz)
    assert list(info.atom_types) == list(test_molecule_data1["info"].atom_types)


def test_start_structure_with_other_atoms_raises(
    test_molecule_data1, test_molecule_data2, tmp_path
):
    wrong = tmp_path / "wrong.xyz"
    wrong.write_text("1\n\nH 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="same atoms"):
        _start(
            test_molecule_data1["struc"],
            test_molecule_data2["struc"],
            start_structure=str(wrong),
        )


def test_get_ts_guess_from_start_file_runs_one_optimization(
    test_molecule_data1, test_molecule_data2, tmp_workdir
):
    """With a start file, only one optimization runs, even with perform_two_optimizations."""
    struc1: Structure = test_molecule_data1["struc"]
    struc2: Structure = test_molecule_data2["struc"]
    # product geometry with atom 0 displaced, so it differs from both endpoints
    nat, _, xyz, atom_types = readin_xyz(str(SMALL_MOLECULE_DIR / "struc2.xyz"))
    xyz = xyz.copy()
    xyz[0] += [0.15, -0.1, 0.05]
    (tmp_workdir / "start.xyz").write_text(
        f"{nat}\n\n"
        + "".join(
            f"{a} {x:.8f} {y:.8f} {z:.8f}\n" for a, (x, y, z) in zip(atom_types, xyz)
        )
    )
    calcdata = CalculationData.from_default()
    calcdata.ts_path.ff_filename = str(tmp_workdir / "tsff.csv")
    calcdata.ts_calc.start_structure = "start.xyz"
    assert calcdata.ts_calc.perform_two_optimizations

    _, _, _, final_geom = get_ts_guess(struc1, struc2, calcdata=calcdata)

    assert (tmp_workdir / "ts_optimization.out").exists()
    assert not (tmp_workdir / "ts_optimization_reactant.out").exists()
    assert not (tmp_workdir / "ts_optimization_product.out").exists()
    nat, _, start_xyz, _ = readin_xyz(str(tmp_workdir / "start.xyz"))
    frame_lines = (tmp_workdir / "trj_ts_guess.xyz").read_text().splitlines()
    first_frame = np.array(
        [line.split()[1:4] for line in frame_lines[2 : 2 + nat]], dtype=float
    )

    # molbar may translate/rotate the frame, so compare interatomic distances
    def distances(xyz):
        return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)

    np.testing.assert_allclose(distances(first_frame), distances(start_xyz), atol=1e-5)
    assert final_geom.shape == (struc1.info.nat, 3)
