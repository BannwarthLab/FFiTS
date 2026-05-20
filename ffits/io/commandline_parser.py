import argparse
import os
import sys
import logging

logger = logging.getLogger(__name__)


def _get_version():
    """
    Get version from pyproject.toml.

    Returns
        version (str): Version string from pyproject.toml, or "unknown" if not found.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    from pathlib import Path

    try:
        # Find pyproject.toml in parent directories
        current_path = Path(__file__).parent
        while current_path != current_path.parent:
            pyproject_path = current_path / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                if "project" in data and "version" in data["project"]:
                    return data["project"]["version"]
            current_path = current_path.parent
    except Exception:
        pass

    return "unknown"


def parse_args():
    """
    Parse command-line arguments and return them as a dictionary.
    """
    parser = argparse.ArgumentParser(
        description="FFiTS TS calculation workflow. Provide structures and input file."
    )

    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="Print version and exit.",
    )
    if "--version" in sys.argv or "-v" in sys.argv:
        version = _get_version()
        print(f"FFiTS version: {version}")
        sys.exit(0)

    # Positional arguments: multiple structure files
    parser.add_argument(
        "structures", nargs="*", help="Structure files (e.g., struc1.xyz struc2.xyz)"
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to the config TOML file (optional).",
    )
    parser.add_argument(
        "--charge",
        "-chrg",
        type=int,
        default=None,
        help="Charge for the system (default: 0).",
    )
    parser.add_argument(
        "--multiplicity",
        "-mult",
        type=int,
        default=None,
        help="Multiplicity of the system (default: 1).",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default=None,
        help="Run optimizer mode with specified csv ff.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with detailed logging output and DEBUG directory creation.",
    )

    args = parser.parse_args()

    if not args.opt and len(args.structures) != 2 and args.version == False:
        parser.error("Exactly two structure files must be provided for TS guess mode.")
    if args.opt and len(args.structures) != 1:
        parser.error("Exactly one structure file must be provided for optimizer mode.")

    # Validate structure files
    for f in args.structures:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Structure file not found: {f}")

    # Validate config file
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.config is not None and args.multiplicity is not None:
        logger.warning(
            f"Multiplicity given in the input file differs from command line multiplicity."
        )
    if args.config is not None and args.charge is not None:
        logger.warning(
            f"Charge given in the input file differs from command line charge."
        )

    return {
        "structures": args.structures,
        "config_file": args.config,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "optff": args.opt,
        "debug": args.debug,
    }
