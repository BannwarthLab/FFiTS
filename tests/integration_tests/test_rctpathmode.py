import re
import shutil

import numpy as np
import pytest

from ffits.io.reader import readin_xyz
from tests.test_utils import run_ffits_as_subprocess, SMALL_MOLECULE_DIR


def _read_trajectory(path):
    """Reads a multi-frame xyz file -> list of (relative energy, atom_types, xyz)."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    frames = []
    idx = 0
    while idx < len(lines):
        nat = int(lines[idx])
        assert lines[idx + 1].endswith("kcal/mol (xtb, relative to reactant)")
        energy = float(lines[idx + 1].split()[0])
        rows = [lines[pos].split() for pos in range(idx + 2, idx + 2 + nat)]
        atom_types = [row[0] for row in rows]
        xyz = np.array([row[1:4] for row in rows], dtype=float)
        frames.append((energy, atom_types, xyz))
        idx += nat + 2
    return frames


def _run_rctpath(tmp_workdir, frames):
    shutil.copy2(SMALL_MOLECULE_DIR / "struc1.xyz", tmp_workdir / "reac.xyz")
    shutil.copy2(SMALL_MOLECULE_DIR / "struc2.xyz", tmp_workdir / "prod.xyz")
    stdout, stderr, returncode = run_ffits_as_subprocess(
        ["reac.xyz", "prod.xyz", "--rctpath", str(frames)]
    )
    assert (
        returncode == 0
    ), f"ffits failed with return code {returncode}\nstderr: {stderr}"
    return stdout


def test_standard_rctpath_run(tmp_workdir):
    """Standard 5-frame run: output files, endpoints, TS guess = highest-energy frame."""
    frames = 5
    stdout = _run_rctpath(tmp_workdir, frames)

    for name in ("path_trj.xyz", "ts_guess.xyz", "ff1.csv", "ff2.csv", "wbo1", "wbo2"):
        assert (tmp_workdir / name).exists(), f"{name} was not created"
    for name in ("struc1.hess", "struc2.hess"):
        assert (tmp_workdir / name).exists(), f"{name} was not created"
    for step in range(frames - 2):
        assert (tmp_workdir / f"ts_guess_{step}.xyz").exists()
        assert (tmp_workdir / f"tsff_{step}.csv").exists()

    trajectory = _read_trajectory(tmp_workdir / "path_trj.xyz")
    assert len(trajectory) == frames
    assert trajectory[0][0] == 0.0  # energies relative to the reactant

    nat, _, reac_xyz, reac_types = readin_xyz(str(tmp_workdir / "reac.xyz"))
    _, _, prod_xyz, _ = readin_xyz(str(tmp_workdir / "prod.xyz"))
    for _, atom_types, xyz in trajectory:
        assert atom_types == list(reac_types)
        assert xyz.shape == (nat, 3)
    np.testing.assert_allclose(trajectory[0][2], reac_xyz, atol=1e-6)
    np.testing.assert_allclose(trajectory[-1][2], prod_xyz, atol=1e-6)

    # ts_guess.xyz is the highest-energy frame of the path
    highest = max(trajectory, key=lambda frame: frame[0])
    _, _, ts_xyz, ts_types = readin_xyz(str(tmp_workdir / "ts_guess.xyz"))
    assert list(ts_types) == highest[1]
    np.testing.assert_allclose(ts_xyz, highest[2], atol=1e-6)

    # every image starts from the previous one, the first from the reactant
    starts = re.findall(r"Starting TS guess optimization from (.*)\.\n", stdout)
    assert starts == ["reactant geometry reac.xyz"] + [
        f"geometry ts_guess_{step}.xyz" for step in range(frames - 3)
    ]


@pytest.mark.parametrize(
    "frames, product_factors",
    [
        (3, ["0.50"]),  # single image, used to divide by zero
        (4, ["0.25", "0.75"]),
    ],
)
def test_rctpath_number_of_images(tmp_workdir, frames, product_factors):
    """Short paths: right number of frames and images and the expected mixing factors."""
    stdout = _run_rctpath(tmp_workdir, frames)

    trajectory = _read_trajectory(tmp_workdir / "path_trj.xyz")
    assert len(trajectory) == frames
    assert (tmp_workdir / "ts_guess.xyz").exists()
    for step in range(frames - 2):
        assert (tmp_workdir / f"ts_guess_{step}.xyz").exists()
    assert not (tmp_workdir / f"ts_guess_{frames - 2}.xyz").exists()

    assert re.findall(r"factor_product = (\d\.\d\d)", stdout) == product_factors
