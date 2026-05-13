from dataclasses import dataclass, field


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
