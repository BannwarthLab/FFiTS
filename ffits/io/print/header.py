import datetime
import textwrap
import subprocess
import os
from pathlib import Path
from importlib import metadata

import tomllib


def _get_build_date():
    """
    Get the build/installation date from the ffits package directory.
    
    Returns
    -------
    str
        Build date as "YYYY-MM-DD HH:MM" based on package modification time.
    """
    try:
        # Get the ffits package directory
        ffits_dir = Path(__file__).parent.parent
        # Use the modification time of the package directory
        mtime = os.path.getmtime(ffits_dir)
        build_datetime = datetime.datetime.fromtimestamp(mtime)
        return build_datetime.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def _get_last_commit_date():
    """
    Get the date of the last commit from git.
    
    Returns
    -------
    str or None
        Last commit date as "YYYY-MM-DD HH:MM" or None if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Format: "2024-02-09 10:30:45 +0100" -> extract "2024-02-09 10:30"
            commit_date = result.stdout.strip()[:16]
            return commit_date
    except Exception:
        pass
    return None


def _load_pyproject_metadata():
    """
    Load metadata from installed package.
    
    Returns
    -------
    dict
        Dictionary with 'license', 'authors', and 'version' keys.
    """
    metadata_dict = {
        "license": "MIT",
        "authors": "Daria Babushkina",
        "version": "-"
    }
    
    try:
        # Get version from installed package metadata (fast, reliable on clusters)
        version = metadata.version("ffits")
        metadata_dict["version"] = version
    except Exception:
        pass
    
    return metadata_dict


def print_program_header(version: str = None):
    """
    Prints a formatted header for the FFiTS program.

    Parameters
    ----------
    version : str, optional
        Program version string. If None, will be loaded from pyproject.toml.
    """

    metadata = _load_pyproject_metadata()
    if version is None:
        version = metadata["version"]

    build_date = _get_build_date()
    
    last_commit_date = _get_last_commit_date()

    logo = r"""
                ███████╗███████╗██╗████████╗███████╗                
                ██╔════╝██╔════╝╚═╝╚══██╔══╝██╔════╝                
                █████╗  █████╗  ██╗   ██║   ███████╗                
                ██╔══╝  ██╔══╝  ██║   ██║   ╚════██║                
                ██║     ██║     ██║   ██║   ███████║                
                ╚═╝     ╚═╝     ╚═╝   ╚═╝   ╚══════╝                
             Force Field-interpolated Transition States             
"""

    # --- Print header ---
    print(logo)
    print("=" * 70)
    print(f" Program:     FFiTS  —  Force Field-interpolated Transition States")
    print(f" Version:     {version}")
    print(f" Build Date:  {build_date}")
    if last_commit_date:
        print(f" Last Commit: {last_commit_date}")
    print(f" Authors:     {metadata['authors']}")
    print(f" License:     {metadata['license']}")
    print("-" * 70)

    # --- Reference / Citation Info TODO ---
    citation = textwrap.dedent("""
        If you use FFiTS in your work, please cite:
        [1] A future paper 
    """).strip()

    print(citation)
    print("=" * 70 + "\n")
