import argparse
import os

def parse_args():
    """
    Parse command-line arguments and return them as a dictionary.
    """
    parser = argparse.ArgumentParser(
        description="FFITS TS calculation workflow. Provide structures and input file."
    )

    # Positional arguments: multiple structure files
    parser.add_argument(
        "structures",
        nargs="+",
        help="Structure files (e.g., struc1.xyz struc2.xyz)"
    )

    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to the config TOML file (optional)."
    )
    parser.add_argument(
        "--charge", "-chrg",
        type=int,
        default=None,
        help="Charge for the system (default: 0)."
    )
    parser.add_argument(
        "--multiplicity", "-mult",
        type=int,
        default=None,
        help="Multiplicity of the system (default: 1)."
    )
    parser.add_argument(
        "--opt",
        type=str,
        default=None,
        help="Run optimizer mode with specified csv ff."
    )

    args = parser.parse_args() 

    # Validate structure files
    for f in args.structures:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Structure file not found: {f}")

    # Validate config file
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
    
    if args.config is not None and args.multiplicity is not None:
        print(f'[WARNING] Multiplicity given in the input file differs from command line multiplicity.')
    if args.config is not None and args.charge is not None:
        print(f'[WARNING] Charge given in the input file differs from command line charge.')

    return {
        "structures": args.structures,
        "config_file": args.config,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "optff": args.opt
    }

