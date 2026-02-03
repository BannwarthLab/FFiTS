from dataclasses import dataclass, field
@dataclass
class System:
    charge: int = 0
    multiplicity: int = 1


@dataclass
class PathData:
    xyz_filename: str = ""
    wbo_filename: str = ""
    hessian_filename: str = ""
    ff_filename: str = ""


@dataclass
class CalculationOptions:
    geometry_optimization: bool = False
    wbo_calc: bool = True
    hessian_calc: bool = True
    ff_parameterization: bool = False
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
    factor_reactant: float = 0.5
    factor_product: float = 0.5
    optimizer: str = "molbar-optimizer"
    energy_threshold_two_optimizations: float = 0.15
    perform_two_optimizations: bool = False
    molbar_optimizer_e_tol: float = 1e-8
    molbar_optimizer_x_tol: float = 1e-3
    molbar_optimizer_max_micro_steps: int = 1
    energy_threshold_two_optimizations: float = 0.15


@dataclass
class Postprocessing:
    relaxation: str = "None"

@dataclass
class CalculationData:
    system: System = field(default_factory=System)
    reactant_path: PathData = field(default_factory=PathData)
    product_path: PathData = field(default_factory=PathData)
    reactant_calc: CalculationOptions = field(default_factory=CalculationOptions)
    product_calc: CalculationOptions = field(default_factory=CalculationOptions)
    ts_calc: TSCalculationOptions = field(default_factory=TSCalculationOptions)
    ts_path: PathData = field(default_factory=PathData)
    postprocessing: Postprocessing = field(default_factory=Postprocessing)


