"""Dataclasses describing calculation configuration for a full ffits run."""
from dataclasses import dataclass, field, fields
import os
import logging
import tomllib

logger = logging.getLogger(__name__)


def load_toml(path: str) -> dict:
    """Loads a TOML file and returns its contents as a dictionary.

    Args:
        path (str): The file path to the TOML file.

    Returns:
        dict: The parsed contents of the TOML file.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def _apply_section(obj, section: dict, skip: tuple = ()) -> None:
    """Overwrites obj's dataclass fields from a TOML section dict, in place.

    Every field of ``obj`` not listed in ``skip`` is read from ``section``:
    a key present in ``section`` overwrites the field's current value, a key
    absent from ``section`` leaves it untouched. Because this walks
    ``obj``'s actual dataclass fields instead of a hand-maintained name
    list, any key written under the matching TOML table -- present or
    added later -- is picked up automatically, with no separate wiring
    step required. This is the single piece of boilerplate behind the
    section-by-section overwriting done throughout
    :meth:`CalculationData.from_config`.

    Args:
        obj: Dataclass instance whose fields are updated.
        section (dict): TOML section dict to read values from.
        skip (tuple[str, ...], optional): Field names to leave untouched
            even if present in ``section`` -- for fields that are
            deliberately not sourced from the config file.
    """
    for f in fields(obj):
        if f.name in skip:
            continue
        setattr(obj, f.name, section.get(f.name, getattr(obj, f.name)))


@dataclass
class System:
    """
    General information, which are relevant during the whole calculation.

    Attributes:
        charge (int): Total charge of the system.
        multiplicity (int): Spin multiplicity of the system.
        xtb_path (str): Path to the xtb executable.
        xtb_input_name (str | None): Optional name for xtb input files.
        xtb_alpb_solvent (str | None): Optional solvent specification for xtb ALPB calculations.
    """

    charge: int = 0
    multiplicity: int = 1
    xtb_path: str = "xtb"
    use_gxtb: bool = False
    xtb_input_name: str | None = None
    xtb_alpb_solvent: str | None = None


@dataclass
class PathData:
    """
    Information about the paths to the different files for one structure.

    Attributes:
        xyz_filename (str): Path to the XYZ file containing the structure's coordinates.
        wbo_filename (str): Path to the file containing Wiberg bond orders for the structure.
        hessian_filename (str): Path to the file containing the Hessian matrix for the structure.
        ff_filename (str): Path to the file containing the force field parameters for the structure
    """

    xyz_filename: str = ""
    wbo_filename: str = ""
    hessian_filename: str = ""
    ff_filename: str = ""


@dataclass
class CalculationOptions:
    """
    Information for changing different parameters during the preliminary calculations and FF parameterization of energy minima.

    Attributes:
        geometry_optimization (bool): Whether to perform a geometry optimization before calculating WBOs and Hessian.
        wbo_calc (bool): Whether to calculate WBOs with xtb or read them from file.
        hessian_calc (bool): Whether to calculate the Hessian with xtb or read it from file.
        bo_treshold (float): Threshold for defining bonds based on bond orders during the generation of the TS guess.
        ff_parameterization (bool): Whether to perform FF parameterization or read FF parameters from file.
        test_parameterization (bool): Whether to perform a test of the FF parameterization by calculating the energy and gradient with the FF and comparing it to the reference xtb values.
        ff_parameter_repulsion (float): Starting value for the repulsion parameters during FF parameterization.
        ff_parameterization_maxiteration (int): Maximum number of iterations for FF parameterization.
        ff_parameterization_stepsize (float): Step size for FF parameterization.
        ff_parameterization_threshold (float): Threshold for convergence of FF parameterization based on the change in the RMSD between FF and Ref Hessian between iterations.
        ff_parameterization_constant_repulsion (bool): If set to true, the ff parameters of the repulsive terms are not changed during fitting. This can be useful to prevent overfitting of the FF to the reference data.

    """

    geometry_optimization: bool = False
    wbo_calc: bool = True
    hessian_calc: bool = True
    bo_treshold: float = 0.0
    only_proper_dihedrals: bool = True
    ff_parameterization: bool = True
    test_parameterization: bool = False
    ff_parameter_repulsion: float = 0.01
    ff_parameterization_maxiteration: int = 1000
    ff_parameterization_stepsize: float = 0.15
    ff_parameterization_threshold: float = 0.0005
    # the lower the threshold, the more prone to overfitting the model becomes, the higher the less precize it gets
    ff_parameterization_constant_repulsion: bool = True
    # If set to true, the ff parameters of the repulsive terms are not changed during fitting


@dataclass
class TSCalculationOptions:
    """
    Information for changing different parameters during the TS guess generation and optimization with the TSFF.

    Attributes:
        factor_reactant (float): Reactant weight when mixing into the TS force field.
        factor_product (float): Product weight when mixing into the TS force field.
        optimizer (str): Optimizer to use ("molbar-optimizer" or "scipy-optimizer").
        average_with_hess_weight (bool): Weight reactant/product averaging by Hessian similarity.
        hess_weight_sharpness (float): Sharpness of the Hessian-similarity weighting.
        perform_two_optimizations (bool): Whether to run a second TS optimization pass.
        molbar_optimizer_e_tol (float): Energy convergence tolerance for the molbar optimizer.
        molbar_optimizer_x_tol (float): Coordinate convergence tolerance for the molbar optimizer.
        molbar_optimizer_max_micro_steps (int): Max micro-steps per molbar optimizer iteration.
        energy_threshold_two_optimizations (float): Threshold deciding if a second optimization runs. Not checked currently.
    """

    factor_reactant: float = 0.5
    factor_product: float = 0.5
    optimizer: str = "molbar-optimizer"
    average_with_hess_weight: bool = True
    hess_weight_sharpness: float = 0.8
    perform_two_optimizations: bool = False
    molbar_optimizer_e_tol: float = 1e-4
    molbar_optimizer_x_tol: float = 1e-2
    molbar_optimizer_max_micro_steps: int = 1
    energy_threshold_two_optimizations: float = 0.15


@dataclass
class Postprocessing:
    """
    Information for Postprocessing, eg somehow changing the generated TS guess. Not used currently.

    Attributes:
        relaxation (str): Postprocessing relaxation method to apply to the
            TS guess ("None", "gfn2-xtb", or "pbeh-3c").
    """

    relaxation: str = "None"


@dataclass
class CalculationData:
    """
    Summary class for all calculation information.

    Attributes:
        system (System): General system information (charge, multiplicity, xtb settings).
        reactant_path (PathData): File paths for the reactant structure.
        product_path (PathData): File paths for the product structure.
        reactant_calc (CalculationOptions): Calculation options for the reactant.
        product_calc (CalculationOptions): Calculation options for the product.
        ts_calc (TSCalculationOptions): Calculation options for the TS guess generation/optimization.
        ts_path (PathData): File paths for the TS guess.
        postprocessing (Postprocessing): Postprocessing options applied to the TS guess.
    """

    system: System = field(default_factory=System)
    reactant_path: PathData = field(default_factory=PathData)
    product_path: PathData = field(default_factory=PathData)
    reactant_calc: CalculationOptions = field(default_factory=CalculationOptions)
    product_calc: CalculationOptions = field(default_factory=CalculationOptions)
    ts_calc: TSCalculationOptions = field(default_factory=TSCalculationOptions)
    ts_path: PathData = field(default_factory=PathData)
    postprocessing: Postprocessing = field(default_factory=Postprocessing)

    @staticmethod
    def from_default():
        """Creates a CalculationData object with default values. This can be useful for testing the TS guess generation process with a standard set of parameters when no specific calculation data is provided.

        Returns:
            CalculationData: A CalculationData object initialized with default values.
        """
        cd = CalculationData()
        cd.reactant_path.xyz_filename = "struc1.xyz"
        cd.reactant_path.wbo_filename = "wbo1"
        cd.reactant_path.hessian_filename = "struc1.hess"
        cd.reactant_path.ff_filename = "ff1.csv"
        cd.product_path.xyz_filename = "struc2.xyz"
        cd.product_path.wbo_filename = "wbo2"
        cd.product_path.hessian_filename = "struc2.hess"
        cd.product_path.ff_filename = "ff2.csv"
        cd.ts_path.ff_filename = "tsff.csv"
        cd.ts_path.hessian_filename = "ts.hess"
        cd.ts_path.wbo_filename = "wbo_ts"
        return cd

    @staticmethod
    def from_config(path_to_config: str, test: bool = False) -> "CalculationData":
        """Builds a CalculationData object from a TOML config file, layered over the defaults.

        Starts from :meth:`from_default` and overwrites values found in the
        TOML file. Unless ``test`` is True, also runs sanity checks on the
        result (required files exist, options are consistent and valid).

        Args:
            path_to_config (str): Path to the TOML config file.
            test (bool, optional): If True, skip the sanity checks. Defaults to False. Only for testing.

        Raises:
            FileNotFoundError: If the config file or a required Hessian file is missing.
            ValueError: If a setting fails validation (see the sanity checks below).

        Returns:
            CalculationData: CalculationData object initialized with values from the config file.
        """
        cd: CalculationData = CalculationData.from_default()
        if not os.path.exists(path_to_config):
            raise FileNotFoundError(f"Config file {path_to_config} not found.")
        logger.info(f"Loading calculation data from config file: {path_to_config}")
        data = load_toml(path_to_config)
        # =======================
        # System
        # =======================
        sys_data = data.get("system", {})
        _apply_section(cd.system, sys_data)

        # =======================
        # Reactant
        # =======================
        reactant = data.get("reactant", {})

        # Reactant path
        react_path = reactant.get("path", {})
        cd.reactant_path.xyz_filename = ""  # will be set from commandline
        _apply_section(cd.reactant_path, react_path, skip=("xyz_filename",))

        # Reactant calculation
        react_calc = reactant.get("calculation", {})
        _apply_section(cd.reactant_calc, react_calc)

        # =======================
        # Product
        # =======================
        product = data.get("product", {})

        # Product path
        prod_path = product.get("path", {})
        cd.product_path.xyz_filename = ""  # will be set from commandline
        _apply_section(cd.product_path, prod_path, skip=("xyz_filename",))

        # Product calculation
        prod_calc = product.get("calculation", {})
        _apply_section(cd.product_calc, prod_calc)

        # =======================
        # TS Guess calculation
        # =======================
        ts_guess = data.get("ts_guess", {})

        # TS path
        ts_path = ts_guess.get("path", {})
        _apply_section(cd.ts_path, ts_path)

        # TS calculation
        ts_calc = ts_guess.get("calculation", {})
        _apply_section(cd.ts_calc, ts_calc)

        # =======================
        # Postprocessing
        # =======================
        post = data.get("postprocessing", {})
        _apply_section(cd.postprocessing, post)

        # for testing purposes, return the CalculationData object without performing sanity checks, to allow testing of error handling in those checks
        if test:
            return cd
        # =======================
        # Sanity checks
        # =======================
        if not cd.reactant_calc.hessian_calc and not os.path.exists(
            cd.reactant_path.hessian_filename
        ):
            raise FileNotFoundError(
                f"No Hessian calculation requested, but file {cd.reactant_path.hessian_filename} not found."
            )

        if not cd.product_calc.hessian_calc and not os.path.exists(
            cd.product_path.hessian_filename
        ):
            raise FileNotFoundError(
                f"No Hessian calculation requested, but file {cd.product_path.hessian_filename} not found."
            )
        if cd.system.multiplicity > 3 or cd.system.multiplicity < 1:
            raise ValueError(
                f"Multiplicity of {cd.system.multiplicity} is chemically unreasonable on this theory level."
            )
        if (
            cd.reactant_calc.only_proper_dihedrals
            != cd.product_calc.only_proper_dihedrals
        ):
            raise ValueError(
                "The option only_proper_dihedrals is set differently for the reactant and product structures. This option must be the same for both structures, as it affects the TS FF creation and optimization."
            )
        if (
            not cd.reactant_calc.ff_parameterization
            and not cd.product_calc.ff_parameterization
        ):
            logger.warning(
                "FF parameterization is not enabled for either the reactant or product structure. This means that the TS FF will be created using the default FF parameters and no comparison of reactant and product structures will be performed. Thus the option only_proper_dihedrals will be ignored if it is in the config file."
            )

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
            raise ValueError(
                f"Invalid relaxation option: {cd.postprocessing.relaxation}"
            )

        return cd
