import os
import pytest
from tempfile import TemporaryDirectory
from ffits.io.reader import read_wbo_file  # <-- replace with your actual module name


def write_file(path: str, content: str):
    """Helper to write a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_read_wbo_file_valid():
    content = """1 2 0.95
2 3 1.12
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.wbo")
        write_file(file_path, content)

        result = read_wbo_file(file_path)
        expected = {(1, 2): 0.95, (2, 3): 1.12}

        assert result == expected


def test_read_wbo_file_skips_empty_lines():
    content = """1 2 0.95

2 3 1.12
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_empty.wbo")
        write_file(file_path, content)

        result = read_wbo_file(file_path)
        expected = {(1, 2): 0.95, (2, 3): 1.12}

        assert result == expected


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_wbo_file("nonexistent.wbo")


def test_malformed_line():
    content = """1 2 0.95
bad line
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "malformed.wbo")
        write_file(file_path, content)

        with pytest.raises(ValueError, match="Malformed line"):
            read_wbo_file(file_path)


def test_invalid_numbers():
    content = """1 2 notanumber
"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "invalid.wbo")
        write_file(file_path, content)

        with pytest.raises(ValueError, match="Invalid numeric values"):
            read_wbo_file(file_path)
