"""Tests for ffits.io.file_writer (structure/hessian/trajectory writers)."""

import networkx as nx
import numpy as np
import pytest

from ffits.io.file_writer import (
    print_box,
    generate_xtb_fix_input,
    write_all_molecules_to_file,
    write_string2file,
    write_xyz_to_file,
    write_hessian_to_orcahessfile,
    write_trajectory_to_xyz,
)


def test_print_box_smoke(capsys):
    print_box("hello", width=20)
    out = capsys.readouterr().out
    assert "hello" in out
    # top/middle/bottom border lines
    assert out.count("\n") == 3


def test_generate_xtb_fix_input_single_atom():
    result = generate_xtb_fix_input({0})
    assert result == "$fix\n   atoms: 0\nend\n"


def test_generate_xtb_fix_input_multiple_atoms():
    result = generate_xtb_fix_input({0, 1, 2})
    assert result.startswith("$fix\n   atoms: ")
    assert result.endswith("\nend\n")
    # every requested atom index appears in the atoms line
    atoms_line = result.splitlines()[1]
    for atom in (0, 1, 2):
        assert str(atom) in atoms_line


def test_write_string2file(tmp_path):
    target = tmp_path / "out.txt"
    write_string2file("hello world", target)
    assert target.read_text() == "hello world"


def test_write_string2file_overwrites_existing_content(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old content")
    write_string2file("new content", target)
    assert target.read_text() == "new content"


def test_write_xyz_to_file(tmp_path):
    target = tmp_path / "struc.xyz"
    xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    write_xyz_to_file(xyz, str(target))

    lines = target.read_text().splitlines()
    assert lines[0] == "2"
    assert lines[1] == ""
    assert lines[2] == "0.000000 0.000000 0.000000"
    assert lines[3] == "1.500000 0.000000 0.000000"


def test_write_hessian_to_orcahessfile(tmp_path):
    target = tmp_path / "out.hess"
    nat = 2
    hessian = np.arange(36, dtype=float).reshape(6, 6)
    xyz_with_masses = ["C 12.011 0.0 0.0 0.0 6", "H 1.008 1.0 0.0 0.0 1"]
    write_hessian_to_orcahessfile(nat, hessian, xyz_with_masses, str(target))

    content = target.read_text()
    assert content.startswith("$orca_hessian_file\n")
    assert "$atoms\n" in content
    assert "$hessian\n" in content
    assert content.rstrip().endswith("$end")
    # atom count and hessian dimension are both written as their own line
    assert "\n2\n" in content
    assert "\n6\n" in content


def test_write_trajectory_to_xyz(tmp_path):
    target = tmp_path / "trajectory.xyz"
    trajectory = [
        {
            "nat": 2,
            "atom_types": np.array(["C", "H"]),
            "xyz": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "energy": -1.23456789,
        },
        {
            "nat": 2,
            "atom_types": np.array(["C", "H"]),
            "xyz": np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]),
            # no "energy" key -- should default to 0
        },
    ]
    write_trajectory_to_xyz(trajectory, target)

    lines = target.read_text().splitlines()
    assert lines[0] == "2"
    assert lines[1] == "-1.23456789"
    assert lines[2] == "C 0.00000000 0.00000000 0.00000000"
    assert lines[3] == "H 1.00000000 0.00000000 0.00000000"
    assert lines[4] == "2"
    assert lines[5] == "0.00000000"


def test_write_all_molecules_to_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    xyz = {1: "C 0.0 0.0 0.0\n", 2: "H 1.0 0.0 0.0\n"}

    write_all_molecules_to_file([graph], xyz, "mol")

    output = tmp_path / "mol1.xyz"
    assert output.exists()
    lines = output.read_text().splitlines()
    assert lines[0] == "2"
    assert lines[1] == ""
    assert "C 0.0 0.0 0.0" in lines[2]
    assert "H 1.0 0.0 0.0" in lines[3]
