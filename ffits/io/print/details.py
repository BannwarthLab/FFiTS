"""Console printouts for individual stages of a run (FF fitting, optimization, TS guess)."""

HARTREE2KCAL = 627.5094740631


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


def print_rct_path_start(n_images: int):
    """Print the "Reaction Path" section header."""
    print("\n" + "=" * 60)
    print(f" Reaction Path ({n_images} image(s) between reactant and product)")
    print("=" * 60)


def print_rct_path_image(step: int, n_images: int, fact1: float, fact2: float):
    """Print the header of one reaction-path image (``step`` is 0-based, printed as step/n_images)."""
    print("\n" + "-" * 60)
    print(
        f" Step {step + 1}/{n_images} (ts_guess_{step}.xyz): factor_reactant = {fact1:.2f}, factor_product = {fact2:.2f}"
    )
    print("-" * 60)


def print_rct_path_summary(
    names: list[str], product_factors: list[float], energies: list[float], highest: int
):
    """Print the frames of the reaction path with their xtb energies relative to the reactant."""
    width = max(len("Structure"), *(len(name) for name in names))
    print("\n" + "=" * 60)
    print(" Reaction Path Summary")
    print("=" * 60)
    print(
        f"  {'Frame':>5}  {'Structure':<{width}}  {'f_product':>9}  {'E(xtb) / Eh':>14}  {'dE / kcal/mol':>13}"
    )
    for i, (name, fact2, energy) in enumerate(zip(names, product_factors, energies)):
        marker = "  <- TS guess" if i == highest else ""
        print(
            f"  {i:>5}  {name:<{width}}  {fact2:>9.2f}  {energy:>14.8f}  "
            f"{(energy - energies[0]) * HARTREE2KCAL:>13.2f}{marker}"
        )
    print(f"\n  Highest-energy frame {highest} ({names[highest]}) written to ts_guess.xyz")
