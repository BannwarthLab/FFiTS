"""Console printouts summarizing calculation configuration."""
from ffits.datatype.calculation_data import (
    CalculationData,
    PathData,
    CalculationOptions,
)


def print_header_setup():
    """Print the "SETUP" section header."""
    print("\n" + "\n" + "=" * 60)
    print(" SETUP ")
    print("=" * 60)


def print_calculation_data(calc_data: CalculationData):
    """
    Nicely formatted printout summarizing the setup of a TS calculation.
    """
    print("\n" + "=" * 60)
    print(" INPUT ")
    print("=" * 60)

    # --- System section ---
    system = calc_data.system
    print("\n[ System ]")
    print(f"  Charge:       {system.charge}")
    print(f"  Multiplicity: {system.multiplicity}")
    print(f"  XTB path:     {system.xtb_path}")
    print(f"  XTB input:    {system.xtb_input_name or '—'}")
    print(f"  XTB ALPB solvent: {system.xtb_alpb_solvent or '—'}")

    # --- Paths section ---
    print("\n[ Reactant Paths ]")
    _print_path(calc_data.reactant_path)

    print("\n[ Product Paths ]")
    _print_path(calc_data.product_path)

    print("\n[ TS Paths ]")
    _print_path(calc_data.ts_path)

    # --- Calculation options ---
    print("\n[ Reactant Calculation Options ]")
    _print_calc_options(calc_data.reactant_calc)

    print("\n[ Product Calculation Options ]")
    _print_calc_options(calc_data.product_calc)

    print("\n[ TS Calculation Options ]")
    ts = calc_data.ts_calc
    print(f"  Optimizer:                        {ts.optimizer}")
    print(f"  Factor (Reactant):                {ts.factor_reactant:.2f}")
    print(f"  Factor (Product):                 {ts.factor_product:.2f}")
    # print(f"  Energy threshold (two opts):      {ts.energy_threshold_two_optimizations:.3f}")
    # print(f"  Perform two optimizations:        {ts.perform_two_optimizations}")
    # print(f"  Molbar optimizer E tolerance:     {ts.molbar_optimizer_e_tol}")
    # print(f"  Molbar optimizer X tolerance:     {ts.molbar_optimizer_x_tol}")
    # print(f"  Molbar optimizer max micro steps: {ts.molbar_optimizer_max_micro_steps}")

    # --- Postprocessing section ---
    print("\n[ Postprocessing ]")
    print(f"  Relaxation: {calc_data.postprocessing.relaxation}")


def _print_path(path: PathData):
    """Helper to print file paths neatly."""
    print(f"  XYZ file:       {path.xyz_filename or '—'}")
    print(f"  WBO file:       {path.wbo_filename or '—'}")
    print(f"  Hessian file:   {path.hessian_filename or '—'}")
    print(f"  FF file:        {path.ff_filename or '—'}")


def _print_calc_options(opt: CalculationOptions):
    """Helper to print booleans with Yes/No formatting."""
    print(
        f"  Geometry optimization:            {'Yes' if opt.geometry_optimization else 'No'}"
    )
    print(f"  WBO calculation:                  {'Yes' if opt.wbo_calc else 'No'}")
    print(f"  Hessian calculation:              {'Yes' if opt.hessian_calc else 'No'}")
    print(
        f"  FF parameterization:              {'Yes' if opt.ff_parameterization else 'No'}"
    )
    print(
        f"  Test parameterization:            {'Yes' if opt.test_parameterization else 'No'}"
    )
    # print(f"  FF parameter repulsion:           {opt.ff_parameter_repulsion}")
    # print(f"  FF parameterization max iter:     {opt.ff_parameterization_maxiteration}")
    # print(f"  FF parameterization step size:    {opt.ff_parameterization_stepsize}")
    # print(f"  FF parameterization threshold:    {opt.ff_parameterization_threshold}")
    # print(f"  FF constant repulsion:            {'Yes' if opt.ff_parameterization_constant_repulsion else 'No'}")
