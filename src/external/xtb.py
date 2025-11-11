#!/bin/python

import subprocess
import os
from src.io.reader import read_wbo_file, read_hessian, readin_xyz
import subprocess
import tempfile
import shutil
from pathlib import Path


class Xtb:
    """
    xTB program caller and wrapper, calculations are performed in temporary directories
    """
    #TODO add lömi

    def __init__(self, chrg: int, mult: int, xtb_path: str = "xtb") -> None:
        self.xtb_path = xtb_path
        self.chrg = chrg
        self.uhf = mult - 1 # multiplicity = number of unpaired electrons + 1
        self._check_xtb_loaded()
        print(f"[INFO] xTB will be run with uhf = {self.uhf}, chrg = {self.chrg}")

    # ------------------------------------------------------------------
    # --- UTILITIES ----------------------------------------------------
    # ------------------------------------------------------------------

    def _check_xtb_loaded(self):
        """Check if xTB executable is available."""
        result = subprocess.run(
            ["which", self.xtb_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not result.stdout.strip():
            raise RuntimeError(f"xTB executable '{self.xtb_path}' not found in PATH.")

    def _run_xtb(self, command: str, cwd: Path):
        """Run an xTB command inside cwd and handle errors."""
        stdout = cwd / "xtb.out"
        stderr = cwd / "xtb_err.out"
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            stdout=stdout.open("w"),
            stderr=stderr.open("w"),
        )
        if result.returncode != 0:
            raise RuntimeError(f"xTB failed with code {result.returncode}. See {stderr}")
        return result.returncode

    def _get_command(self, input_xyz: Path, keyword: str) -> str:
        """Build xTB command string."""
        return f"{self.xtb_path} {input_xyz} --uhf {self.uhf} --chrg {self.chrg} {keyword}"

    # ------------------------------------------------------------------
    # --- CORE CALCULATIONS --------------------------------------------
    # ------------------------------------------------------------------

    def singlepoint(self, input_xyz: str, output_name: str) -> float:
        """Run xTB singlepoint calculation in a temporary directory."""
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(input_xyz, tmp / Path(input_xyz).name)
            command = self._get_command(Path(input_xyz).name, "")
            self._run_xtb(command, tmp)

            # Save result outputs
            energy = self._find_energy_in_output(tmp / "xtb.out")
            shutil.copy(tmp / "xtb.out", f"{output_name}_singlepoint.out")
            return energy

    def geomopt(self, input_xyz: str, output_name: str):
        """Run xTB geometry optimization and save optimized geometry."""
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            xyz_name = Path(input_xyz).name
            shutil.copy(input_xyz, tmp / xyz_name)

            command = self._get_command(xyz_name, "--opt")
            self._run_xtb(command, tmp)

            optimized_xyz = tmp / "xtbopt.xyz"
            optimized_log = tmp / "xtbopt.log"
            if not optimized_xyz.exists():
                raise RuntimeError("No optimized geometry found (xtbopt.xyz missing).")

            shutil.copy(optimized_xyz, f"{output_name}")
            shutil.copy(optimized_log, f"trj_{output_name}")
            (tmp / "xtbrestart").unlink(missing_ok=True)
            print(f'[INFO] Geometry optimization of {input_xyz} to {output_name} finished successfully.')

            return f"{output_name}"

    def hesscalc(self, input_xyz: str, output_name: str):
        """Run xTB Hessian calculation."""
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(input_xyz, tmp / Path(input_xyz).name)
            command = self._get_command(Path(input_xyz).name, "--hess")
            self._run_xtb(command, tmp)

            hess_file = tmp / "hessian"
            if not hess_file.exists():
                raise RuntimeError("No Hessian file generated.")
            shutil.copy(hess_file, f"{output_name}")
            print(f'[INFO] Hessian calculation for {input_xyz} finished successfully.')
            return read_hessian(f"{output_name}")

    def wbocalc(self, input_xyz: str, output_name: str):
        """Run xTB WBO calculation and return parsed WBO data."""
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(input_xyz, tmp / Path(input_xyz).name)
            command = self._get_command(Path(input_xyz).name, "--wbo")
            self._run_xtb(command, tmp)

            wbo_file = tmp / "wbo"
            if not wbo_file.exists():
                raise RuntimeError("No WBO file generated.")
            shutil.copy(wbo_file, f"{output_name}")
            print(f'[INFO] WBO calculation for {input_xyz} finished successfully.')
            return read_wbo_file(f"{output_name}")
        
    def geomopt_with_topology_check(self, input_xyz: str, output_basename: str, wbo_output_basename: str, threshold: int = 0.3) -> str:
        """
        Perform geometry optimization and check if topology (WBOs) changed.
        Returns True if significant WBO difference is detected, else False.
        """
        wbo_before = self.wbocalc(input_xyz, f"before_{wbo_output_basename}")
        opt_xyz = self.geomopt(input_xyz, f"opt_{output_basename}")
        wbo_after = self.wbocalc(opt_xyz, f"{wbo_output_basename}")

        for bond, before_val in wbo_before.items():
            after_val = wbo_after.get(bond, 0.0)
            if abs(after_val - before_val) > threshold:
                print(f'[WARNING] Topology changed significantly during geometry optimization in bond {bond} with a change in bond order of {abs(after_val - before_val)}.')
                return f"{opt_xyz}" 

        return f"{opt_xyz}", wbo_after  

    # ------------------------------------------------------------------
    # --- ANALYSIS -----------------------------------------------------
    # ------------------------------------------------------------------

    def _find_energy_in_output(self, output_filename: Path) -> float:
        """Parse total energy from an xTB output file."""
        with output_filename.open("r") as f:
            for line in reversed(f.readlines()):
                if "TOTAL ENERGY" in line:
                    return float(line.split()[3])
        raise ValueError(f"No total energy found in {output_filename}")
