import os
import pytest
import numpy as np
from tempfile import TemporaryDirectory
from src.io.reader import readin_xyz  


def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_readin_xyz_valid(tmp_path):
    xyz_content = """3
Water molecule
O  0.000  0.000  0.000
H  0.758  0.000  0.504
H -0.758  0.000  0.504
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "water.xyz")
        write_file(file_path, xyz_content)

        nat, comment, coords, atoms = readin_xyz(file_path)

        assert nat == 3
        assert comment == "Water molecule"
        assert atoms == ["O", "H", "H"]
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (3, 3)
        np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0])


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        readin_xyz("nonexistent.xyz")


def test_invalid_first_line(tmp_path):
    xyz_content = """notanint
comment
O 0.0 0.0 0.0
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "bad.xyz")
        write_file(file_path, xyz_content)

        with pytest.raises(ValueError, match="First line of XYZ file must be an integer"):
            readin_xyz(file_path)


def test_atom_count_mismatch(tmp_path):
    xyz_content = """2
Too few atoms
O 0.0 0.0 0.0
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "mismatch.xyz")
        write_file(file_path, xyz_content)

        with pytest.raises(ValueError, match="Atom count mismatch"):
            readin_xyz(file_path)


def test_malformed_line(tmp_path):
    xyz_content = """1
Broken line
O 0.0 0.0
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "broken.xyz")
        write_file(file_path, xyz_content)

        with pytest.raises(ValueError, match="Malformed line"):
            readin_xyz(file_path)
