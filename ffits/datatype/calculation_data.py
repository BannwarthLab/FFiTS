from dataclasses import dataclass, field
import os
import logging
import tomllib

logger = logging.getLogger(__name__)


def load_toml(path: str) -> dict:
    """Loads a TOML file and returns its contents as a dictionary.

    Args:
        path (str): The file path to the TOML file.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass
class System:
    """
    General information, which are relevant during the whole calculation.

    Variables:
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

    Variables:
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

    Variables:
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
    """

    factor_reactant: float = 0.5
    factor_product: float = 0.5
    optimizer: str = "molbar-optimizer"
    energy_threshold_two_optimizations: float = 0.15
    average_with_hess_weight: bool = True
    perform_two_optimizations: bool = False
    molbar_optimizer_e_tol: float = 1e-4
    molbar_optimizer_x_tol: float = 1e-3
    molbar_optimizer_max_micro_steps: int = 1
    energy_threshold_two_optimizations: float = 0.15


@dataclass
class Postprocessing:
    """
    Information for Postprocessing, eg somehow changing the generated TS guess.
    """

    relaxation: str = "None"


@dataclass
class CalculationData:
    """
    Summary class for all calculation information.
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
        """_summary_

        Args:
            path_to_config (str): _description_
            test (bool, optional): _description_. Defaults to False.

        Raises:
            FileNotFoundError: _description_
            FileNotFoundError: _description_
            FileNotFoundError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

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
        cd.system.charge = sys_data.get("charge", cd.system.charge)
        cd.system.multiplicity = sys_data.get("multiplicity", cd.system.multiplicity)
        cd.system.use_gxtb = sys_data.get("use_gxtb", cd.system.use_gxtb)
        cd.system.xtb_path = sys_data.get("xtb_path", cd.system.xtb_path)
        cd.system.xtb_input_name = sys_data.get(
            "xtb_input_name", cd.system.xtb_input_name
        )
        cd.system.xtb_alpb_solvent = sys_data.get(
            "xtb_alpb_solvent", cd.system.xtb_alpb_solvent
        )

        # =======================
        # Reactant
        # =======================
        reactant = data.get("reactant", {})

        # Reactant path
        react_path = reactant.get("path", {})
        cd.reactant_path.xyz_filename = ""  # will be set from commandline
        cd.reactant_path.wbo_filename = react_path.get(
            "wbo_filename", cd.reactant_path.wbo_filename
        )
        cd.reactant_path.hessian_filename = react_path.get(
            "hessian_filename", cd.reactant_path.hessian_filename
        )
        cd.reactant_path.ff_filename = react_path.get(
            "ff_filename", cd.reactant_path.ff_filename
        )

        # Reactant calculation
        react_calc = reactant.get("calculation", {})
        cd.reactant_calc.geometry_optimization = react_calc.get(
            "geometry_optimization", cd.reactant_calc.geometry_optimization
        )
        cd.reactant_calc.wbo_calc = react_calc.get(
            "wbo_calc", cd.reactant_calc.wbo_calc
        )
        cd.reactant_calc.hessian_calc = react_calc.get(
            "hessian_calc", cd.reactant_calc.hessian_calc
        )
        cd.reactant_calc.only_proper_dihedrals = react_calc.get(
            "only_proper_dihedrals", cd.reactant_calc.only_proper_dihedrals
        )    
        cd.reactant_calc.ff_parameterization = react_calc.get(
            "ff_parameterization", cd.reactant_calc.ff_parameterization
        )
        cd.reactant_calc.test_parameterization = react_calc.get(
            "test_parameterization", cd.reactant_calc.test_parameterization
        )
        cd.reactant_calc.ff_parameterization_maxiteration = react_calc.get(
            "ff_parameterization_maxiteration",
            cd.reactant_calc.ff_parameterization_maxiteration,
        )
        cd.reactant_calc.ff_parameter_repulsion = react_calc.get(
            "ff_parameter_repulsion", cd.reactant_calc.ff_parameter_repulsion
        )
        cd.reactant_calc.ff_parameterization_stepsize = react_calc.get(
            "ff_parameterization_stepsize",
            cd.reactant_calc.ff_parameterization_stepsize,
        )
        cd.reactant_calc.ff_parameterization_threshold = react_calc.get(
            "ff_parameterization_threshold",
            cd.reactant_calc.ff_parameterization_threshold,
        )
        cd.reactant_calc.ff_parameterization_constant_repulsion = react_calc.get(
            "ff_parameterization_constant_repulsion",
            cd.reactant_calc.ff_parameterization_constant_repulsion,
        )
        cd.reactant_calc.bo_treshold = react_calc.get(
            "bo_treshold",
            cd.reactant_calc.bo_treshold,
        )

        # =======================
        # Product
        # =======================
        product = data.get("product", {})

        # Product path
        prod_path = product.get("path", {})
        cd.product_path.xyz_filename = ""  # will be set from commandline
        cd.product_path.wbo_filename = prod_path.get(
            "wbo_filename", cd.product_path.wbo_filename
        )
        cd.product_path.hessian_filename = prod_path.get(
            "hessian_filename", cd.product_path.hessian_filename
        )
        cd.product_path.ff_filename = prod_path.get(
            "ff_filename", cd.product_path.ff_filename
        )

        # Product calculation
        prod_calc = product.get("calculation", {})
        cd.product_calc.geometry_optimization = prod_calc.get(
            "geometry_optimization", cd.product_calc.geometry_optimization
        )
        cd.product_calc.wbo_calc = prod_calc.get("wbo_calc", cd.product_calc.wbo_calc)
        cd.product_calc.hessian_calc = prod_calc.get(
            "hessian_calc", cd.product_calc.hessian_calc
        )
        cd.product_calc.only_proper_dihedrals = prod_calc.get(
            "only_proper_dihedrals", cd.product_calc.only_proper_dihedrals
        )
        cd.product_calc.ff_parameterization = prod_calc.get(
            "ff_parameterization", cd.product_calc.ff_parameterization
        )
        cd.product_calc.test_parameterization = prod_calc.get(
            "test_parameterization", cd.product_calc.test_parameterization
        )
        cd.product_calc.ff_parameterization_maxiteration = prod_calc.get(
            "ff_parameterization_maxiteration",
            cd.product_calc.ff_parameterization_maxiteration,
        )
        cd.product_calc.ff_parameter_repulsion = prod_calc.get(
            "ff_parameter_repulsion", cd.product_calc.ff_parameter_repulsion
        )
        cd.product_calc.ff_parameterization_stepsize = prod_calc.get(
            "ff_parameterization_stepsize", cd.product_calc.ff_parameterization_stepsize
        )
        cd.product_calc.ff_parameterization_threshold = prod_calc.get(
            "ff_parameterization_threshold",
            cd.product_calc.ff_parameterization_threshold,
        )
        cd.product_calc.ff_parameterization_constant_repulsion = prod_calc.get(
            "ff_parameterization_constant_repulsion",
            cd.product_calc.ff_parameterization_constant_repulsion,
        )
        cd.product_calc.bo_treshold = prod_calc.get(
            "bo_treshold",
            cd.product_calc.bo_treshold,
        )

        # =======================
        # TS Guess calculation
        # =======================
        ts_guess = data.get("ts_guess_calculation", {})

        # TS path
        ts_path = ts_guess.get("path", {})
        cd.ts_path.ff_filename = ts_path.get("ff_filename", cd.ts_path.ff_filename)
        cd.ts_path.hessian_filename = ts_path.get(
            "hessian_filename", cd.ts_path.hessian_filename
        )

        # TS calculation
        ts_calc = ts_guess.get("calculation", {})
        cd.ts_calc.factor_reactant = ts_calc.get(
            "factor_reactant", cd.ts_calc.factor_reactant
        )
        cd.ts_calc.factor_product = ts_calc.get(
            "factor_product", cd.ts_calc.factor_product
        )
        cd.ts_calc.optimizer = ts_calc.get("optimizer", cd.ts_calc.optimizer)
        cd.ts_calc.energy_threshold_two_optimizations = ts_calc.get(
            "energy_threshold_two_optimizations",
            cd.ts_calc.energy_threshold_two_optimizations,
        )
        cd.ts_calc.average_with_hess_weight = ts_calc.get(
            "average_with_hess_weight", cd.ts_calc.average_with_hess_weight
        )
        cd.ts_calc.perform_two_optimizations = ts_calc.get(
            "perform_two_optimizations", cd.ts_calc.perform_two_optimizations
        )
        cd.ts_calc.molbar_optimizer_e_tol = ts_calc.get(
            "molbar_optimizer_e_tol", cd.ts_calc.molbar_optimizer_e_tol
        )
        cd.ts_calc.molbar_optimizer_x_tol = ts_calc.get(
            "molbar_optimizer_x_tol", cd.ts_calc.molbar_optimizer_x_tol
        )
        cd.ts_calc.molbar_optimizer_max_micro_steps = ts_calc.get(
            "molbar_optimizer_max_micro_steps",
            cd.ts_calc.molbar_optimizer_max_micro_steps,
        )

        # =======================
        # Postprocessing
        # =======================
        post = data.get("postprocessing", {})
        cd.postprocessing.relaxation = post.get(
            "relaxation", cd.postprocessing.relaxation
        )

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
        if cd.reactant_calc.only_proper_dihedrals != cd.product_calc.only_proper_dihedrals:
            raise ValueError(
                "The option only_proper_dihedrals is set differently for the reactant and product structures. This option must be the same for both structures, as it affects the TS FF creation and optimization."
            )
        if ((not cd.reactant_calc.ff_parameterization and not cd.product_calc.ff_parameterization)):
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
