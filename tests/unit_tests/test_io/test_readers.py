"""
Tests for IO readers (XYZ, WBO, Hessian).
Consolidated test suite for all file reading functionality.
"""

import os
import pytest
import numpy as np
from tempfile import TemporaryDirectory
from ffits.io.reader import readin_xyz, read_wbo_file, read_xtb_hessian


def write_file(path, content):
    """Helper to write a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ============================================================================
# XYZ READER TESTS
# ============================================================================


class TestXyzReader:
    """Tests for readin_xyz function."""

    def test_readin_xyz_valid(self, tmp_path):
        """Test reading a valid XYZ file."""
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

    def test_readin_xyz_file_not_found(self):
        """Test handling of missing XYZ file."""
        with pytest.raises(FileNotFoundError):
            readin_xyz("nonexistent.xyz")

    def test_readin_xyz_invalid_first_line(self, tmp_path):
        """Test error handling for non-integer first line."""
        xyz_content = """notanint
comment
O 0.0 0.0 0.0
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "bad.xyz")
            write_file(file_path, xyz_content)

            with pytest.raises(
                ValueError, match="First line of XYZ file must be an integer"
            ):
                readin_xyz(file_path)

    def test_readin_xyz_atom_count_mismatch(self, tmp_path):
        """Test error handling for atom count mismatch."""
        xyz_content = """2
Too few atoms
O 0.0 0.0 0.0
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "mismatch.xyz")
            write_file(file_path, xyz_content)

            with pytest.raises(ValueError, match="Atom count mismatch"):
                readin_xyz(file_path)

    def test_readin_xyz_malformed_line(self, tmp_path):
        """Test error handling for malformed atom lines."""
        xyz_content = """1
Broken line
O 0.0 0.0
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "broken.xyz")
            write_file(file_path, xyz_content)

            with pytest.raises(ValueError, match="Malformed line"):
                readin_xyz(file_path)

    def test_readin_xyz_large_molecule(self, tmp_path):
        """Test reading a larger molecule."""
        xyz_content = """7
Ethanol-like
C -2.333 3.312 0.201
C -0.916 2.859 -0.043
O  0.063 3.559 0.089
H -2.926 3.184 -0.713
H -2.797 2.680 0.968
H -0.815 1.797 -0.367
H -2.351 4.357 0.516
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "ethanol.xyz")
            write_file(file_path, xyz_content)

            nat, comment, coords, atoms = readin_xyz(file_path)

            assert nat == 7
            assert len(atoms) == 7
            assert coords.shape == (7, 3)

    def test_readin_emptylastline(self, tmp_path):
        """Test reading a larger molecule."""
        xyz_content = """7
Ethanol-like
C -2.333 3.312 0.201
C -0.916 2.859 -0.043
O  0.063 3.559 0.089
H -2.926 3.184 -0.713
H -2.797 2.680 0.968
H -0.815 1.797 -0.367
H -2.351 4.357 0.516

"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "ethanol.xyz")
            write_file(file_path, xyz_content)

            nat, comment, coords, atoms = readin_xyz(file_path)

            assert nat == 7
            assert len(atoms) == 7
            assert coords.shape == (7, 3)


# ============================================================================
# WBO READER TESTS
# ============================================================================


class TestWboReader:
    """Tests for read_wbo_file function."""

    def test_read_wbo_file_valid(self):
        """Test reading a valid WBO file."""
        content = """1 2 0.95
2 3 1.12
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.wbo")
            write_file(file_path, content)

            result = read_wbo_file(file_path)
            expected = {(0, 1): 0.95, (1, 2): 1.12}

            assert result == expected

    def test_read_wbo_file_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        content = """1 2 0.95

2 3 1.12

"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_empty.wbo")
            write_file(file_path, content)

            result = read_wbo_file(file_path)
            expected = {(0, 1): 0.95, (1, 2): 1.12}

            assert result == expected

    def test_read_wbo_file_file_not_found(self):
        """Test handling of missing WBO file."""
        with pytest.raises(FileNotFoundError):
            read_wbo_file("nonexistent.wbo")

    def test_read_wbo_file_malformed_line(self):
        """Test error handling for malformed lines."""
        content = """1 2 0.95
bad line
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "bad.wbo")
            write_file(file_path, content)

            with pytest.raises(ValueError, match="Malformed line"):
                read_wbo_file(file_path)

    def test_read_wbo_file_invalid_numeric_values(self):
        """Test error handling for non-numeric values."""
        content = """1 2 notanumber
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "bad_num.wbo")
            write_file(file_path, content)

            with pytest.raises(ValueError, match="Invalid numeric values"):
                read_wbo_file(file_path)

    def test_read_wbo_file_sorted_bond_pairs(self):
        """Test that bond pairs are sorted."""
        content = """2 1 0.95
3 1 0.88
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "unsorted.wbo")
            write_file(file_path, content)

            result = read_wbo_file(file_path)

            # Keys should be sorted tuples
            for bond in result.keys():
                assert bond[0] < bond[1]

    def test_read_wbo_file_large_molecule(self):
        """Test reading WBO data for a larger molecule."""
        content = """1 2 1.027
2 3 1.928
1 4 0.956
1 5 0.956
2 6 0.934
1 7 0.983
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "ethanol.wbo")
            write_file(file_path, content)

            result = read_wbo_file(file_path)

            assert len(result) == 6
            assert result[(0, 1)] == 1.027
            assert result[(1, 2)] == 1.928


# ============================================================================
# HESSIAN READER TESTS
# ============================================================================


class TestHessianReader:
    """Tests for read_hessian function."""

    def test_read_hessian_valid_1atom(self):
        """Test reading a 3x3 Hessian (1 atom)."""
        content = """$hessian
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "hess.txt")
            write_file(file_path, content)

            hess = read_xtb_hessian(file_path)
            assert hess.shape == (3, 3)
            np.testing.assert_allclose(hess, np.eye(3))

    def test_read_hessian_valid_2atoms(self):
        """Test reading a 6x6 Hessian (2 atoms)."""
        content = """$hessian
1 0 0 0 0 0
0 1 0 0 0 0
0 0 1 0 0 0
0 0 0 1 0 0
0 0 0 0 1 0
0 0 0 0 0 1
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "hess2.txt")
            write_file(file_path, content)

            hess = read_xtb_hessian(file_path)
            assert hess.shape == (6, 6)
            np.testing.assert_allclose(hess, np.eye(6))

    def test_read_hessian_non_square_error(self):
        """Test error for non-square Hessian dimensions."""
        content = "1 2 3 4 5"
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "bad.txt")
            write_file(file_path, content)

            with pytest.raises(ValueError, match="does not form a square matrix"):
                read_xtb_hessian(file_path)

    def test_read_hessian_invalid_dimensions_error(self):
        """Test error for Hessian dimensions not divisible by 3."""
        content = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "bad_dims.txt")
            write_file(file_path, content)

            with pytest.raises(ValueError, match="not divisible by 3"):
                read_xtb_hessian(file_path)

    def test_read_hessian_symmetric(self):
        """Test that returned Hessian is symmetric."""
        content = """$hessian
1.0 0.5 0.2
0.5 2.0 0.3
0.2 0.3 1.5
"""
        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sym.txt")
            write_file(file_path, content)

            hess = read_xtb_hessian(file_path)
            np.testing.assert_array_almost_equal(hess, hess.T)

    def test_read_hessian_large_molecule(self):
        """Test reading Hessian for a larger molecule (7 atoms)."""
        # 21x21 Hessian for 7 atoms
        size = 21
        data = np.eye(size).flatten().tolist()
        content = "$hessian\n" + " ".join(map(str, data))

        with TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "large.txt")
            write_file(file_path, content)

            hess = read_xtb_hessian(file_path)
            assert hess.shape == (21, 21)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
