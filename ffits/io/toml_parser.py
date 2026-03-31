import logging
import os
from ffits.datatype.calculation_data import CalculationData
import argparse
import tomllib


logger = logging.getLogger(__name__)



def load_toml(path: str) -> dict:
    """Load a TOML file from the given path."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_update(base: dict, updates: dict) -> dict:
    """
    Recursively update a nested dict (base) with another (updates).
    Keeps unspecified defaults while overwriting provided keys.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(user_path: str | None = None) -> dict:
    """Load the default TOML config and update with user overrides."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, ".."))  # go from src/io → src
    default_path = os.path.join(project_root, "data", "default_config.toml")
    # Load defaults
    config = load_toml(default_path)

    # Load and merge user config if provided
    if user_path and os.path.exists(user_path):
        user_config = load_toml(user_path)
        config = deep_update(config, user_config)
    else:
        logger.info("No user config found, using defaults.")

    return config

def load_calculation_data(user_path: str | None = None, test: bool = False) -> CalculationData:
    """
    Load configuration data from a TOML file (if provided) and fill
    a CalculationData instance with defaults for missing values.
    """
    data = load_config(user_path)
    cd = CalculationData()  # start with defaults

    # =======================
    # System
    # =======================
    sys_data = data.get("system", {})
    cd.system.charge = sys_data.get("charge", cd.system.charge)
    cd.system.multiplicity = sys_data.get("multiplicity", cd.system.multiplicity)
    cd.system.xtb_path = sys_data.get("xtb_path", cd.system.xtb_path)
    cd.system.xtb_input_name = sys_data.get("xtb_input_name", cd.system.xtb_input_name)
    cd.system.xtb_alpb_solvent = sys_data.get("xtb_alpb_solvent", cd.system.xtb_alpb_solvent)

    # =======================
    # Reactant
    # =======================
    reactant = data.get("reactant", {})

    # Reactant path
    react_path = reactant.get("path", {})
    cd.reactant_path.xyz_filename = '' # will be set from commandline
    cd.reactant_path.wbo_filename = react_path.get("wbo_filename", cd.reactant_path.wbo_filename)
    cd.reactant_path.hessian_filename = react_path.get("hessian_filename", cd.reactant_path.hessian_filename)
    cd.reactant_path.ff_filename = react_path.get("ff_filename", cd.reactant_path.ff_filename)

    # Reactant calculation
    react_calc = reactant.get("calculation", {})
    cd.reactant_calc.geometry_optimization = react_calc.get("geometry_optimization", cd.reactant_calc.geometry_optimization)
    cd.reactant_calc.wbo_calc = react_calc.get("wbo_calc", cd.reactant_calc.wbo_calc)
    cd.reactant_calc.hessian_calc = react_calc.get("hessian_calc", cd.reactant_calc.hessian_calc)
    cd.reactant_calc.ff_parameterization = react_calc.get("ff_parameterization", cd.reactant_calc.ff_parameterization)
    cd.reactant_calc.test_parameterization = react_calc.get("test_parameterization", cd.reactant_calc.test_parameterization)
    cd.reactant_calc.ff_parameterization_maxiteration = react_calc.get("ff_parameterization_maxiteration", cd.reactant_calc.ff_parameterization_maxiteration)
    cd.reactant_calc.ff_parameter_repulsion = react_calc.get("ff_parameter_repulsion", cd.reactant_calc.ff_parameter_repulsion)
    cd.reactant_calc.ff_parameterization_stepsize = react_calc.get("ff_parameterization_stepsize", cd.reactant_calc.ff_parameterization_stepsize)
    cd.reactant_calc.ff_parameterization_threshold = react_calc.get("ff_parameterization_threshold", cd.reactant_calc.ff_parameterization_threshold)
    cd.reactant_calc.ff_parameterization_constant_repulsion = react_calc.get("ff_parameterization_constant_repulsion", cd.reactant_calc.ff_parameterization_constant_repulsion)

    # =======================
    # Product
    # =======================
    product = data.get("product", {})

    # Product path
    prod_path = product.get("path", {})
    cd.product_path.xyz_filename = '' # will be set from commandline
    cd.product_path.wbo_filename = prod_path.get("wbo_filename", cd.product_path.wbo_filename)
    cd.product_path.hessian_filename = prod_path.get("hessian_filename", cd.product_path.hessian_filename)
    cd.product_path.ff_filename = prod_path.get("ff_filename", cd.product_path.ff_filename)

    # Product calculation
    prod_calc = product.get("calculation", {})
    cd.product_calc.geometry_optimization = prod_calc.get("geometry_optimization", cd.product_calc.geometry_optimization)
    cd.product_calc.wbo_calc = prod_calc.get("wbo_calc", cd.product_calc.wbo_calc)
    cd.product_calc.hessian_calc = prod_calc.get("hessian_calc", cd.product_calc.hessian_calc)
    cd.product_calc.ff_parameterization = prod_calc.get("ff_parameterization", cd.product_calc.ff_parameterization)
    cd.product_calc.test_parameterization = prod_calc.get("test_parameterization", cd.product_calc.test_parameterization)
    cd.product_calc.ff_parameterization_maxiteration = prod_calc.get("ff_parameterization_maxiteration", cd.product_calc.ff_parameterization_maxiteration)
    cd.product_calc.ff_parameterization_stepsize = prod_calc.get("ff_parameterization_stepsize", cd.product_calc.ff_parameterization_stepsize)
    cd.product_calc.ff_parameterization_threshold = prod_calc.get("ff_parameterization_threshold", cd.product_calc.ff_parameterization_threshold)
    cd.product_calc.ff_parameterization_constant_repulsion = prod_calc.get("ff_parameterization_constant_repulsion", cd.product_calc.ff_parameterization_constant_repulsion)

    # =======================
    # TS Guess calculation
    # =======================
    ts_guess = data.get("ts_guess_calculation", {})

    # TS path
    ts_path = ts_guess.get("path", {})
    cd.ts_path.ff_filename = ts_path.get("ff_filename", cd.ts_path.ff_filename)
    cd.ts_path.hessian_filename = ts_path.get("hessian_filename", cd.ts_path.hessian_filename)

    # TS calculation
    ts_calc = ts_guess.get("calculation", {})
    cd.ts_calc.factor_reactant = ts_calc.get("factor_reactant", cd.ts_calc.factor_reactant)
    cd.ts_calc.factor_product = ts_calc.get("factor_product", cd.ts_calc.factor_product)
    cd.ts_calc.optimizer = ts_calc.get("optimizer", cd.ts_calc.optimizer)
    cd.ts_calc.energy_threshold_two_optimizations = ts_calc.get(
        "energy_threshold_two_optimizations", cd.ts_calc.energy_threshold_two_optimizations
    )
    cd.ts_calc.perform_two_optimizations = ts_calc.get("perform_two_optimizations", cd.ts_calc.perform_two_optimizations)
    cd.ts_calc.molbar_optimizer_e_tol = ts_calc.get("molbar_optimizer_e_tol", cd.ts_calc.molbar_optimizer_e_tol)
    cd.ts_calc.molbar_optimizer_x_tol = ts_calc.get("molbar_optimizer_x_tol", cd.ts_calc.molbar_optimizer_x_tol)
    cd.ts_calc.molbar_optimizer_max_micro_steps = ts_calc.get("molbar_optimizer_max_micro_steps", cd.ts_calc.molbar_optimizer_max_micro_steps)

    # =======================
    # Postprocessing
    # =======================
    post = data.get("postprocessing", {})
    cd.postprocessing.relaxation = post.get("relaxation", cd.postprocessing.relaxation)

    # for testing purposes, return the CalculationData object without performing sanity checks, to allow testing of error handling in those checks
    if test:
        return cd
    # =======================
    # Sanity checks
    # =======================
    if not cd.reactant_calc.hessian_calc and not os.path.exists(cd.reactant_path.hessian_filename) :
        raise FileNotFoundError(f'No Hessian calculation requested, but file {cd.reactant_path.hessian_filename} not found.')
    
    if not cd.product_calc.hessian_calc and not os.path.exists(cd.product_path.hessian_filename) :
        raise FileNotFoundError(f'No Hessian calculation requested, but file {cd.product_path.hessian_filename} not found.')
    
    if cd.system.multiplicity > 3 or cd.system.multiplicity < 1:
        raise ValueError(f'Multiplicity of {cd.system.multiplicity} is chemically unreasonable on this theory level.')
    
    # if cd.system.charge > 3 or cd.system.charge < -3:
    #     print(f'[WARNING] Charge of {cd.system.charge} may be a bit much. Are you certain this is correct?')
    
    # Ensure factors sum to 1.0
    if abs(cd.ts_calc.factor_reactant + cd.ts_calc.factor_product - 1.0) > 1e-8:
        raise ValueError("Error: factor_reactant + factor_product must equal 1.0")

    # Ensure optimization option is valid
    valid_optimization = ["molbar-optimizer", "scipy-optimizer"]
    if cd.ts_calc.optimizer not in valid_optimization:
        raise ValueError(f"Invalid optimization option: {cd.ts_calc.optimizer}")

    # Ensure relaxation option is valid
    valid_relaxations = ["None", "gfn2-xtb", "pbeh-3c"]
    if cd.postprocessing.relaxation not in valid_relaxations:
        raise ValueError(f"Invalid relaxation option: {cd.postprocessing.relaxation}")

    return cd

def overwrite_from_commandline(calcdata: CalculationData, multiplicity: int | None, charge: int | None, structures: list | None = None):
    '''overwrites information in CalculationData object if given via commandline'''
    if multiplicity:
        if multiplicity != calcdata.system.multiplicity:
            logger.info(f"Multiplicity of {calcdata.system.multiplicity} is being overwritten by {multiplicity}.")
        calcdata.system.multiplicity = multiplicity
    if charge:
        if charge != calcdata.system.charge:
            logger.info(f"Charge of {calcdata.system.charge} is being overwritten by {charge}.")
        calcdata.system.charge = charge
    if structures:
        if len(structures) >= 1:
            calcdata.reactant_path.xyz_filename = structures[0]
            logger.info(f"Reactant XYZ file set to: {structures[0]}")
        if len(structures) >= 2:
            calcdata.product_path.xyz_filename = structures[1]
            logger.info(f"Product XYZ file set to: {structures[1]}")
        if len(structures) > 2:
            logger.warning(f"More than 2 structure files provided. Only the first two will be used.")

