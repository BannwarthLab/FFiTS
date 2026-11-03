"""Tests for ffits.ts_guess.rct_path.

The full reaction-path run (real xtb/molbar) is covered by
tests/integration_tests/test_rctpathmode.py; here the file helper and the
image loop of create_path() (with the TS guess and xtb faked) are tested.
"""

import numpy as np

from ffits.ts_guess.rct_path import _write_structure_to_xyz


def test_write_structure_to_xyz_basic(tmp_path):
    structure = {
        "nat": 2,
        "atom_types": ["C", "H"],
        "xyz": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    target = tmp_path / "struc.xyz"

    _write_structure_to_xyz(structure, str(target))

    lines = target.read_text().splitlines()
    assert lines[0] == "2"
    assert lines[1] == ""
    assert lines[2] == "C 0.00000000 0.00000000 0.00000000"
    assert lines[3] == "H 1.00000000 0.00000000 0.00000000"


def test_write_structure_to_xyz_includes_comment(tmp_path):
    structure = {
        "nat": 1,
        "atom_types": ["C"],
        "xyz": np.array([[0.0, 0.0, 0.0]]),
        "comment": "step 3",
    }
    target = tmp_path / "struc.xyz"

    _write_structure_to_xyz(structure, str(target))

    lines = target.read_text().splitlines()
    assert lines[0] == "1"
    assert lines[1] == "step 3"
    assert lines[2] == "C 0.00000000 0.00000000 0.00000000"


def test_create_path_sets_up_once_and_starts_from_previous_image(
    monkeypatch, tmp_workdir, capsys
):
    """Input and setup run once; every image is one optimization from the previous image."""
    import shutil

    from ffits.datatype.calculation_data import CalculationData
    from ffits.ts_guess import rct_path
    from tests.test_utils import SMALL_MOLECULE_DIR

    shutil.copy2(SMALL_MOLECULE_DIR / "struc1.xyz", "reac.xyz")
    shutil.copy2(SMALL_MOLECULE_DIR / "struc2.xyz", "prod.xyz")
    calls = {"input": 0, "setup": 0}
    images = []

    def fake_prepare_structures(cd):
        calls["setup"] += 1
        return "struc1", "struc2"

    def fake_get_ts_guess(struc1, struc2, calcdata):
        assert (struc1, struc2) == ("struc1", "struc2")
        assert calcdata.ts_calc.scale_one_sided_bonds
        images.append(
            (
                calcdata.ts_calc.start_structure,
                calcdata.ts_calc.perform_two_optimizations,
            )
        )
        shutil.copy2("reac.xyz", "ts_guess.xyz")
        open(calcdata.ts_path.ff_filename, "w").close()
        return None, True, 0.0, None

    energies = iter([-1.0, -0.99, -0.95, -0.97, -0.98, -1.01])

    class FakeXtb:
        def __init__(self, **kwargs):
            pass

        def singlepoint(self, xyz, name):
            return next(energies)

    monkeypatch.setattr(rct_path, "prepare_structures", fake_prepare_structures)
    monkeypatch.setattr(rct_path, "get_ts_guess", fake_get_ts_guess)
    monkeypatch.setattr(
        rct_path,
        "print_calculation_data",
        lambda cd: calls.__setitem__("input", calls["input"] + 1),
    )
    monkeypatch.setattr(rct_path, "Xtb", FakeXtb)

    calcdata = CalculationData.from_default()
    calcdata.reactant_path.xyz_filename = "reac.xyz"
    calcdata.product_path.xyz_filename = "prod.xyz"
    _, highest = rct_path.create_path(calcdata, steps=4)

    assert calls == {"input": 1, "setup": 1}
    assert images == [
        ("reactant", False),
        ("ts_guess_0.xyz", False),
        ("ts_guess_1.xyz", False),
        ("ts_guess_2.xyz", False),
    ]
    assert highest["energy"] == -0.95
    out = capsys.readouterr().out
    for step in range(4):
        assert f"Step {step + 1}/4 (ts_guess_{step}.xyz)" in out
    assert "Reaction Path Summary" in out
    assert "ts_guess_1.xyz" in out.split("<- TS guess")[0].splitlines()[-1]
