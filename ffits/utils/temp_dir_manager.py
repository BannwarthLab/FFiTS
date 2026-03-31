"""
Temporary directory management for FFiTS calculations.

When DEBUG logging is enabled, automatically preserves intermediate
calculation outputs in a structured DEBUG directory for inspection.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class TempDirManager:
    """
    Manages temporary directory outputs when DEBUG logging is enabled.

    Creates subdirectories in a DEBUG/ directory for each calculation step,
    organizing outputs for easy inspection and debugging.

    Attributes
        debug_dir : Path
            Path to the DEBUG directory (created on first use if DEBUG enabled)
        enabled : bool
            True if DEBUG logging is enabled

    """
    def __init__(self, work_dir: Optional[Path] = None):
        """
        Initialize the temporary directory manager.

        Parameters
        ----------
        work_dir : Path, optional
            Working directory where DEBUG/ will be created. Defaults to current directory.
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.debug_dir = self.work_dir / "DEBUG"
        self.enabled = self._is_debug_logging_enabled()

    def _is_debug_logging_enabled(self) -> bool:
        """Check if DEBUG logging level is enabled."""
        root_logger = logging.getLogger("ffits")
        return root_logger.isEnabledFor(logging.DEBUG)

    def get_debug_subdir(self, step_name: str) -> Optional[Path]:
        """
        Get (and create) a subdirectory in DEBUG/ for a specific calculation step.

        If DEBUG is not enabled, returns None. Does not create directories.

        Parameters
        ----------
        step_name : str
            Name of the calculation step (e.g., 'xtb_opt_reactant', 'xtb_hess_product')

        Returns
        -------
        Path or None
            Path to the subdirectory (created if it didn't exist), or None if DEBUG disabled.

        Examples
        --------
        >>> manager = TempDirManager()
        >>> debug_path = manager.get_debug_subdir('xtb_opt_reactant')
        >>> # Returns: DEBUG/xtb_opt_reactant/
        """
        if not self.enabled:
            return None

        subdir = self.debug_dir / step_name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def copy_temp_to_debug(self, temp_path: Path, step_name: str) -> Optional[Path]:
        """
        Copy contents from a temporary directory to a DEBUG subdirectory.

        Only performs copy if DEBUG logging is enabled. Creates the DEBUG
        subdirectory structure automatically.

        Parameters
        ----------
        temp_path : Path
            Path to the temporary directory to copy from
        step_name : str
            Name of the calculation step for the subdirectory name

        Returns
        -------
        Path or None
            Path to the DEBUG subdirectory where files were copied, or None if DEBUG disabled.

        Examples
        --------
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     manager = TempDirManager()
        ...     debug_path = manager.copy_temp_to_debug(Path(tmpdir), 'xtb_opt_reactant')
        """
        if not self.enabled:
            return None

        if not temp_path.exists():
            logger.warning(f"Temporary directory {temp_path} does not exist, skipping copy")
            return None

        debug_subdir = self.debug_dir / step_name
        debug_subdir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy all contents from temp_path to debug_subdir
            for item in temp_path.iterdir():
                dest = debug_subdir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            logger.debug(f"Copied temporary directory contents to {debug_subdir}")
            return debug_subdir

        except Exception as e:
            logger.warning(f"Failed to copy temporary directory to DEBUG: {e}")
            return None

    def is_enabled(self) -> bool:
        """
        Check if DEBUG directory preservation is enabled.

        Returns
        -------
        bool
            True if DEBUG logging is active.
        """
        return self.enabled
