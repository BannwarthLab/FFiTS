from interface.call_xtb import *
# from interface.call_crest import *
from datatype.calculation_data import *
from datatype.structure_data import *

# class Calculation:
#     def __init__(self, input_xyz: str, calculation_data: CalculationParameters):
#         self.input_xyz = input_xyz
#         self.calc = calculation_data

# @dataclass
class GeometryOptimization:
        # TODO add a wbo calc and check to check whether topology changes and issue a warning

    @staticmethod
    def with_xtb(input_structure: StructurePath, calc: CalculationParameters, xtb_path='xtb'):
        if not calc.perform_geometryoptimization:
            print(f'> No geometryoptimization will be performed because perform_geometryoptimization is set to {calc.perform_geometryoptimization}.')
            return None
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        energy = xtb.geomopt(input_xyz)
        print(f'> xTB geometry optimization of {input_xyz} finished with a final energy of {round(energy, 5)}.')
        print(f'> New xTB geometry is written to {input_xyz}.')
        return energy


class HessianCalculation:

    @staticmethod
    def with_xtb(input_structure: StructurePath, calc: CalculationParameters, xtb_path='xtb'):
        if not calc.perform_hesscalculation:
            print(f'> No Hessian Calculation will be performed because perform_geometryoptimization is set to {calc.perform_hesscalculation}.')
            return None
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        xtb.hesscalc(input_structure.hess_filename, input_xyz)
        print(f'> xTB Hessian calculation for {input_xyz} finished.')
        print(f'> Hessian is written to {input_structure.hess_filename}.')
        return None


class WboCalculation:

    @staticmethod
    def with_xtb(input_structure: StructurePath, calc: CalculationParameters, xtb_path='xtb'):
        if not calc.perform_wbocalculation:
            print(f'> No WBO Calculation will be performed because perform_geometryoptimization is set to {calc.perform_wbocalculation}.')
            return None
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        wbo_dict = xtb.wbocalc(input_structure.wbo_filename, input_xyz)
        print(f'> xTB WBO calculation for {input_xyz} finished.')
        print(f'> WBO is written to {input_structure.wbo_filename}.')
        return wbo_dict
