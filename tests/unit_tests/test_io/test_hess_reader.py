import os
import pytest
import numpy as np
from tempfile import TemporaryDirectory
from ffits.io.reader import read_hessian  # replace with your actual module name


def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_read_hessian_valid_1atom():
    # 3x3 Hessian for 1 atom
    content = """$hessian
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "hess.txt")
        write_file(file_path, content)

        hess = read_hessian(file_path)
        assert hess.shape == (3, 3)
        np.testing.assert_allclose(hess, np.eye(3))


def test_read_hessian_valid_2atoms():
    # 6x6 Hessian for 2 atoms
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

        hess = read_hessian(file_path)
        assert hess.shape == (6, 6)


def test_read_hessian_non_square():
    # 5 elements -> cannot form square
    content = "1 2 3 4 5"
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "nonsquare.txt")
        write_file(file_path, content)

        with pytest.raises(ValueError, match="does not form a square matrix"):
            read_hessian(file_path)


def test_read_hessian_not_divisible_by_3():
    # 4x4 Hessian -> 16 elements, 16 is not divisible by 3
    content = """1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "notdiv3.txt")
        write_file(file_path, content)

        with pytest.raises(ValueError, match="not divisible by 3"):
            read_hessian(file_path)


def test_read_hessian_empty_file():
    content = ""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "empty.txt")
        write_file(file_path, content)

        with pytest.raises(ValueError):
            read_hessian(file_path)
