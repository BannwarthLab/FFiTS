#!/bin/python

import subprocess
import os
from ffits.io.reader import read_wbo_file, read_xtb_hessian, readin_xyz
import subprocess
import tempfile
import shutil
from pathlib import Path
from importlib import resources
from typing import Tuple, List, Dict
import numpy as np

def get_xtb_path() -> Path:
    """
    Returns a filesystem path to the xtb binary inside the package.
    Safe for pip-installed packages, wheels, or editable installs.
    """
    xtb_file = resources.files("ffits") / "bin" / "xtb"
    with resources.as_file(xtb_file) as path:
        return path


class Xtb:
    """
    xTB program caller and wrapper, calculations are performed in temporary directories
    """
    #TODO add lömi

    def __init__(self, chrg: int, mult: int, xtb_path: str = get_xtb_path()) -> None:
        self.xtb_path = xtb_path
        self.chrg = chrg
        self.uhf = mult - 1 # multiplicity = number of unpaired electrons + 1
        self.xtb_path = xtb_path
        # self._check_xtb_loaded()
        print(f"[INFO] xTB will be run with uhf = {self.uhf}, chrg = {self.chrg}")

    # ------------------------------------------------------------------
    # --- UTILITIES ----------------------------------------------------
    # ------------------------------------------------------------------

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
        print(stderr.read_text())
        if result.returncode != 0:
            raise RuntimeError(f"xTB command {command} failed with code {result.returncode}. See {stderr}")
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

            energy = self._find_energy_in_output(tmp / "xtb.out")
            shutil.copy(tmp / "xtb.out", f"{output_name}_singlepoint.out")
            return energy

    def geomopt(self, input_xyz: str, output_filename: str, output_dir: str | None = None) -> Tuple[int, str, np.ndarray, List[str]]:
        """
        Run xTB geometry optimization and save optimized geometry.
        If output_dir is provided, save output there, otherwise use a temporary directory.
        input:
            input_xyz: path to input geometry file (xyz format)
            output_filename: name of the optimized geometry file to be saved
            output_dir: optional directory to save all xtb output files (including trajectory and logs)
        returns:
            tuple containing optimized geometry data and a dictionary of bond orders
        """
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

            output_path = Path(output_filename)
            shutil.copy(optimized_xyz, output_path)

            trj_path = output_path.parent / f"trj_{output_path.name}"
            shutil.copy(optimized_log, trj_path)

            # copy whole temp directory to output_dir if specified
            if output_dir is not None:
                shutil.copytree(tmp, Path(output_dir), dirs_exist_ok=True)

            (tmp / "xtbrestart").unlink(missing_ok=True)
            print(f'[INFO] Geometry optimization of {input_xyz} to {output_filename} finished successfully.')

            return readin_xyz(f"{output_filename}")

    def hesscalc(self, input_xyz: str, output_name: str, output_dir: str | None = None):
        """
        Run xTB hessian calculation and save hessian.
        If output_dir is provided, save output there, otherwise use a temporary directory.
        input:
            input_xyz: path to input geometry file (xyz format)
            output_filename: name of the hessian file to be saved
            output_dir: optional directory to save all xtb output files 
        returns: 
            parsed Hessian matrix as a numpy array
        """
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(input_xyz, tmp / Path(input_xyz).name)
            command = self._get_command(Path(input_xyz).name, "--hess")
            self._run_xtb(command, tmp)

            hess_file = tmp / "hessian"
            if not hess_file.exists():
                raise RuntimeError("No Hessian file generated.")
            shutil.copy(hess_file, f"{output_name}")
            # copy whole temp directory to output_dir if specified
            if output_dir is not None:
                shutil.copytree(tmp, Path(output_dir), dirs_exist_ok=True)

            print(f'[INFO] Hessian calculation for {input_xyz} finished successfully.')
            return read_xtb_hessian(f"{output_name}")

    def wbocalc(self, input_xyz: str, output_name: str, output_dir: str | None = None)  -> Dict[Tuple[int, int], float]:
        """
        Run xTB geometry optimization and save optimized geometry.
        If output_dir is provided, save output there, otherwise use a temporary directory.
        input:
            input_xyz: path to input geometry file (xyz format)
            output_name: name of the WBO file to be saved
            output_dir: optional directory to save all xtb output files 
        returns: 
            parsed WBO data as a dictionary with keys as tuples of atom indices and values as WBOs
        """
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(input_xyz, tmp / Path(input_xyz).name)
            command = self._get_command(Path(input_xyz).name, "--wbo")
            self._run_xtb(command, tmp)

            wbo_file = tmp / "wbo"
            if not wbo_file.exists():
                raise RuntimeError("No WBO file generated.")
            if not wbo_file.exists():
                raise RuntimeError("No WBO file generated.")
            shutil.copy(wbo_file, f"{output_name}")
            print(f'[INFO] WBO calculation for {input_xyz} finished successfully.')
            return read_wbo_file(f"{output_name}")
        
    def geomopt_with_topology_check(self, input_xyz: str, output_basename: str, wbo_output_basename: str, threshold: float = 0.2) -> Tuple[str, Dict[Tuple[int, int], float]]:
        """
        Perform geometry optimization and check if topology (WBOs) changed.
        
        Args:
            input_xyz: path to input geometry file (xyz format)
            output_basename: path/basename for the optimized geometry file
            wbo_output_basename: path/basename for WBO output files
            threshold: threshold for WBO change to flag topology change
            
        Returns:
            Tuple of (path to optimized xyz file, dictionary of final WBOs)
            If topology changes significantly, prints warning but still returns both values.
        """
        output_base_path = Path(output_basename)
        wbo_base_path = Path(wbo_output_basename)
        
        output_dir = output_base_path.parent if output_base_path.parent != Path(".") else Path(".")
        wbo_dir = wbo_base_path.parent if wbo_base_path.parent != Path(".") else Path(".")
        
        wbo_before_file = str(wbo_dir / f"before_{wbo_base_path.name}")
        wbo_before = self.wbocalc(input_xyz, wbo_before_file)
        
        opt_filename = str(output_dir / f"opt_{output_base_path.name}")
        nat, comment, coordinates, atom_types = self.geomopt(input_xyz, opt_filename)
        
        wbo_after_file = str(wbo_dir / f"after_{wbo_base_path.name}")
        wbo_after = self.wbocalc(opt_filename, wbo_after_file)

        topology_changes = []
        for bond, before_val in wbo_before.items():
            after_val = wbo_after.get(bond, 0.0)
            if abs(after_val - before_val) > threshold:
                topology_changes.append(f"Bond {bond}: {before_val:.4f} -> {after_val:.4f} (change: {abs(after_val - before_val):.4f})")
        
        new_bonds = []
        for bond, after_val in wbo_after.items():
            if bond not in wbo_before:
                new_bonds.append(f"Bond {bond}: formed with WBO {after_val:.4f}")
        
        disappeared_bonds = []
        for bond, before_val in wbo_before.items():
            if bond not in wbo_after:
                disappeared_bonds.append(f"Bond {bond}: disappeared (was {before_val:.4f})")
        
        if topology_changes:
            print(wbo_after, wbo_before)
            for change in topology_changes:
                print(f'[WARNING] Topology changed significantly during geometry optimization: {change}')
        
        if new_bonds:
            print(f'[WARNING] New bonds formed during geometry optimization:')
            for bond in new_bonds:
                print(f'  {bond}')
        
        if disappeared_bonds:
            print(f'[WARNING] Bonds disappeared during geometry optimization:')
            for bond in disappeared_bonds:
                print(f'  {bond}')
        
        return readin_xyz(opt_filename)

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
