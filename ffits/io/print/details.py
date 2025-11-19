

def print_ff_fitting(struc: str):
    print("\n" + "\n" + "=" * 60)
    print(f" FF Fitting for structure {struc}")
    print("=" * 60)

def print_optimization_start():
    print("\n" + "=" * 60)
    print(" TS Guess Construction")
    print("=" * 60)

def print_optimization_end(converged, energy, final_geom, steps, time, message):
    print("\n" + "-" * 60)
    print(" Geometry Optimization Summary")
    print('\n')

    # --- Results block ---
    status = "CONVERGED" if converged else "NOT CONVERGED"
    print(f"  Status:         {status}")
    print(f"  Steps taken:    {steps}")
    print(f"  Final energy:   {energy: .6f}")
    print(f"  Total time:     {time: .2f} s")

    if message:
        print(f"  Message:        {message}")
