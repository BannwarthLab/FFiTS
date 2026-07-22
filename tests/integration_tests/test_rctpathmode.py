from tests.test_utils import run_ffits_as_subprocess, SMALL_MOLECULE_DIR
import shutil
from pathlib import Path
import tempfile
import os


def test_standard_rctpath_run():
    """
    Test that ffits creates optimized.xyz, ff.csv, etc when run with reac.xyz and prod.xyz with standard configuration
    """
    examples_dir = SMALL_MOLECULE_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copy2(examples_dir / "struc1.xyz", temp_path / "reac.xyz")
        shutil.copy2(examples_dir / "struc2.xyz", temp_path / "prod.xyz")

        original_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            stdout, stderr, returncode = run_ffits_as_subprocess(
                ["reac.xyz", "prod.xyz", "--rctpath", "5"]
            )
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            assert (
                returncode == 0
            ), f"ffits failed with return code {returncode}\nstderr: {stderr}"

            output = temp_path / "path_trj.xyz"
            assert output.exists(), f"path_trj.xyz was not created in {temp_path}"

            optimized_output = temp_path / "optimized.xyz"
            assert optimized_output.exists(), f"optimized.xyz was not created in {temp_path}"

            with open(temp_path / "path_trj.xyz", "r", encoding="utf-8") as f:
                trajectory_lines = [line.strip() for line in f.readlines() if line.strip()]

            frames = []
            idx = 0
            while idx < len(trajectory_lines):
                nat = int(trajectory_lines[idx])
                energy = float(trajectory_lines[idx + 1])
                atom_types = []
                coordinates = []
                for pos in range(idx + 2, idx + 2 + nat):
                    parts = trajectory_lines[pos].split()
                    atom_types.append(parts[0])
                    coordinates.append([float(value) for value in parts[1:4]])
                frames.append((energy, atom_types, coordinates))
                idx += nat + 2

            highest_energy_frame = max(frames, key=lambda frame: frame[0])
            with open(optimized_output, "r", encoding="utf-8") as f:
                optimized_lines = [line.strip() for line in f.readlines() if line.strip()]

            optimized_nat = int(optimized_lines[0])
            optimized_atom_types = []
            optimized_coordinates = []
            for pos in range(2, 2 + optimized_nat):
                parts = optimized_lines[pos].split()
                optimized_atom_types.append(parts[0])
                optimized_coordinates.append([float(value) for value in parts[1:4]])

            assert optimized_nat == nat
            assert optimized_atom_types == highest_energy_frame[1]
            assert optimized_coordinates == highest_energy_frame[2]

            ff_csv_file = temp_path / "tsff_2.csv"
            assert ff_csv_file.exists(), f"tsff.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "ff1.csv"
            assert ff_csv_file.exists(), f"ff1.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "ff2.csv"
            assert ff_csv_file.exists(), f"ff2.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "wbo1"
            assert ff_csv_file.exists(), f"wbo1 was not created in {temp_path}"
            ff_csv_file = temp_path / "wbo2"
            assert ff_csv_file.exists(), f"wbo2 was not created in {temp_path}"
            ff_csv_file = temp_path / "struc1.hess"
            assert ff_csv_file.exists(), f"struc1.hess was not created in {temp_path}"
            ff_csv_file = temp_path / "struc2.hess"
            assert ff_csv_file.exists(), f"struc2.hess was not created in {temp_path}"
        finally:
            os.chdir(original_cwd)
