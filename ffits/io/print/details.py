"""Console printouts for individual stages of a run (FF fitting, optimization, TS guess)."""


def print_ff_fitting(struc: str):
    """Print a section header announcing FF fitting for the given structure."""
    print("\n" + "\n" + "=" * 60)
    print(f" FF Fitting for structure {struc}")
    print("=" * 60)


def print_ts_optimization_start():
    """Print the "TS Guess Construction" section header."""
    print("\n" + "=" * 60)
    print(" TS Guess Construction")
    print("=" * 60)


def print_optimization_start():
    """Print the "Optimization" section header."""
    print("\n" + "=" * 60)
    print(" Optimization ")
    print("=" * 60)


def print_optimization_end(converged, energy, final_geom, steps, time, message):
    """Print a summary of a completed geometry optimization."""
    print("\n" + "-" * 60)
    print(" Geometry Optimization Summary")
    print("\n")

    # --- Results block ---
    status = "CONVERGED" if converged else "NOT CONVERGED"
    print(f"  Status:         {status}")
    print(f"  Steps taken:    {steps}")
    print(f"  Final energy:   {energy: .6f}")
    print(f"  Total time:     {time: .2f} s")

    if message:
        print(f"  Message:        {message}")
