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


@dataclass
class TSCalculationOptions:
    factor_reactant: float = 0.5
    factor_product: float = 0.5
    optimizer: str = "molbar-optimizer"
    energy_threshold_two_optimizations: float = 0.15
    perform_two_optimizations: bool = False


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


