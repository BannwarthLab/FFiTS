#!/bin/python

import logging
import subprocess
from ffits.io.reader import read_wbo_file, read_xtb_hessian, readin_xyz
from ffits.utils.temp_dir_manager import TempDirManager
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Xtb:
    """Class to interface with program xtb"""

    def __init__(
        self,
        chrg: int,
        mult: int,
        xtb_alpb_solvent: str | None = None,
        xtb_input_name: str | None = None,
        xtb_path: str = "xtb",
        temp_dir_manager: Optional[TempDirManager] = None,
    ) -> None:
        """_summary_

        Args:
            chrg (int): charge of the system
            mult (int): multiplicity of the system
            xtb_alpb_solvent (str | None, optional): _description_. Defaults to None.
            xtb_input_name (str | None, optional): _description_. Defaults to None.
            xtb_path (str, optional): _description_. Defaults to "xtb".
            temp_dir_manager (Optional[TempDirManager], optional): _description_. Defaults to None.
        """
        # If xtb_path not provided, try to find it

        self.xtb_path = xtb_path
        self.chrg = chrg
        self.uhf = mult - 1  # multiplicity = number of unpaired electrons + 1
        self.xtb_input_name = xtb_input_name
        self.xtb_alpb_solvent = xtb_alpb_solvent
        self.temp_dir_manager = temp_dir_manager
        self._check_xtb_loaded()
        logger.info(f"xTB will be run with uhf = {self.uhf}, chrg = {self.chrg}")
        logger.info(f"Using xTB executable: {self.xtb_path}")
        if self.xtb_alpb_solvent is not None:
            logger.info(
                f"Using ALPB solvent model with solvent: {self.xtb_alpb_solvent}"
            )
        if self.xtb_input_name is not None:
            logger.info(f"Using xtb input file: {self.xtb_input_name}")

    # ------------------------------------------------------------------
    # --- UTILITIES ----------------------------------------------------
    # ------------------------------------------------------------------

    def _check_xtb_loaded(self):
        """Check if xTB executable is available."""
        try:
            result = subprocess.run(
                [self.xtb_path, "--help"],
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            error_msg = (
                f"xTB executable '{self.xtb_path}' not found or not working.\n"
                f"Make sure xTB is installed and accessible.\n"
                f"You can:\n"
                f"  1. Load the module: module load xtb\n"
                f"  2. Add xtb to PATH: export PATH=/path/to/xtb/bin:$PATH\n"
                f"  3. Change variable [system] >> xtb_path in config file to the full path of the xtb executable\n"
                f"Error: {e}"
            )
            raise RuntimeError(error_msg) from e

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
            raise RuntimeError(
                f"xTB command {command} failed with code {result.returncode}. See {stderr}"
            )
        return result.returncode

    def _get_command(self, input_xyz: Path, keyword: str) -> str:
        """Build xTB command string."""
        if self.xtb_alpb_solvent is not None:
            keyword += f" --alpb {self.xtb_alpb_solvent}"
        if self.xtb_input_name is not None:
            keyword += f" --input {self.xtb_input_name}"
        return (
            f"{self.xtb_path} {input_xyz} --uhf {self.uhf} --chrg {self.chrg} {keyword}"
        )

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

    def geomopt(
        self, input_xyz: str, output_filename: str, output_dir: str | None = None
    ) -> Tuple[int, str, np.ndarray, List[str]]:
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

            # Copy to DEBUG directory if debug logging is enabled
            if self.temp_dir_manager is not None:
                step_name = f"xtb_opt_{Path(output_filename).stem}"
                self.temp_dir_manager.copy_temp_to_debug(tmp, step_name)

            (tmp / "xtbrestart").unlink(missing_ok=True)
            logger.info(
                f"Geometry optimization of {input_xyz} to {output_filename} finished successfully."
            )

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
        cwd = Path(input_xyz).parent
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(input_xyz, tmp / Path(input_xyz).name)
            command = self._get_command(Path(input_xyz).name, "--hess")
            self._run_xtb(command, tmp)

            hess_file = tmp / "hessian"
            if not hess_file.exists():
                raise RuntimeError("No Hessian file generated.")
            # copy whole temp directory to output_dir if specified
            if output_dir is not None:
                shutil.copytree(tmp, Path(output_dir), dirs_exist_ok=True)

            # Copy to DEBUG directory if debug logging is enabled
            if self.temp_dir_manager is not None:
                step_name = f"xtb_hess_{Path(output_name).stem}"
                self.temp_dir_manager.copy_temp_to_debug(tmp, step_name)

            shutil.copy(
                hess_file, cwd / output_name
            )  # copy hessian to original directory for later reading
            logger.info(f"Hessian calculation for {input_xyz} finished successfully.")
            return read_xtb_hessian(f"{cwd / output_name}")

    def wbocalc(
        self, input_xyz: str, output_name: str, output_dir: str | None = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Run xTB geometry optimization and save optimized geometry.
        If output_dir is provided, save output there, otherwise use a temporary directory.
        
        Args:
            input_xyz: path to input geometry file (xyz format)
            output_name: name of the WBO file to be saved
            output_dir: optional directory to save all xtb output files

        Returns:
            parsed WBO data as a dictionary with keys as tuples of atom indices and values as WBOs
        """
        cwd = Path(input_xyz).parent
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(input_xyz, tmp / Path(input_xyz).name)
            command = self._get_command(Path(input_xyz).name, "--wbo")
            self._run_xtb(command, tmp)
            if output_dir is not None:
                shutil.copytree(tmp, Path(output_dir), dirs_exist_ok=True)

            # Copy to DEBUG directory if debug logging is enabled
            if self.temp_dir_manager is not None:
                step_name = f"xtb_wbo_{Path(output_name).stem}"
                self.temp_dir_manager.copy_temp_to_debug(tmp, step_name)

            wbo_file = tmp / "wbo"
            if not wbo_file.exists():
                raise RuntimeError("No WBO file generated.")
            if not wbo_file.exists():
                raise RuntimeError("No WBO file generated.")
            shutil.copy(wbo_file, cwd / output_name)
            logger.info(f"WBO calculation for {input_xyz} finished successfully.")
            return read_wbo_file(f"{output_name}")

    def geomopt_with_topology_check(
        self,
        input_xyz: str,
        output_basename: str,
        wbo_output_basename: str,
        threshold: float = 0.2,
    ) -> Tuple[str, Dict[Tuple[int, int], float]]:
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

        output_dir = (
            output_base_path.parent
            if output_base_path.parent != Path(".")
            else Path(".")
        )
        wbo_dir = (
            wbo_base_path.parent if wbo_base_path.parent != Path(".") else Path(".")
        )

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
                topology_changes.append(
                    f"Bond {bond}: {before_val:.4f} -> {after_val:.4f} (change: {abs(after_val - before_val):.4f})"
                )

        new_bonds = []
        for bond, after_val in wbo_after.items():
            if bond not in wbo_before:
                new_bonds.append(f"Bond {bond}: formed with WBO {after_val:.4f}")

        disappeared_bonds = []
        for bond, before_val in wbo_before.items():
            if bond not in wbo_after:
                disappeared_bonds.append(
                    f"Bond {bond}: disappeared (was {before_val:.4f})"
                )

        if topology_changes:
            logger.debug(f"WBOs before optimization:\n{wbo_before}")
            logger.debug(f"WBOs after optimization:\n{wbo_after}")
            for change in topology_changes:
                logger.warning(
                    f"Significant topology change detected during geometry optimization: {change}"
                )

        if new_bonds:
            logger.warning(f"New bonds formed during geometry optimization:")
            for bond in new_bonds:
                logger.warning(f"  {bond}")

        if disappeared_bonds:
            logger.warning(f"Bonds disappeared during geometry optimization:")
            for bond in disappeared_bonds:
                logger.warning(f"  {bond}")

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
