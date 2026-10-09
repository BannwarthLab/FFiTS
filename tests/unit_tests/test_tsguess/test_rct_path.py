"""Tests for the pure-I/O helper in ffits.ts_guess.rct_path.

get_one_image() and create_path() are tightly coupled to real xtb/molbar
calculations and are exercised by the tmp_workdir-based integration tests
instead; _write_structure_to_xyz() is plain file I/O and is covered here.
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
