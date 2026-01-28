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
        "--input", "-i",
        type=str,
        default=None,
        help="Path to the input TOML file (optional)."
    )
    parser.add_argument(
        "--charge", "-c",
        type=int,
        default=None,
        help="Charge for the system (default: 0)."
    )
    parser.add_argument(
        "--multiplicity", "-m",
        type=int,
        default=None,
        help="Multiplicity of the system (default: 1)."
    )
    parser.add_argument(
        "--opt",
        type=str,
        default=None,
        help="Run optimizer mode with specified optimizer type (e.g., 'ff')."
    )

    args = parser.parse_args() 

    # Validate structure files
    for f in args.structures:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Structure file not found: {f}")

    # Validate input file
    if args.input is not None:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
    
    if args.input is not None and args.multiplicity is not None:
        print(f'[WARNING] Multiplicity given in the input file differs from command line multiplicity.')
    if args.input is not None and args.charge is not None:
        print(f'[WARNING] Charge given in the input file differs from command line charge.')

    return {
        "structures": args.structures,
        "input_file": args.input,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "opt_mode": args.opt
    }

