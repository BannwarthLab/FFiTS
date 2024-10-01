from interface.call_xtb import *
# from interface.call_crest import *
from datatype.calculation_data import *

# class Calculation:
#     def __init__(self, input_xyz: str, calculation_data: CalculationParameters):
#         self.input_xyz = input_xyz
#         self.calc = calculation_data

# @dataclass
class GeometryOptimization:
        # TODO add a wbo calc and check to check whether topology changes and issue a warning

    @staticmethod
    def with_xtb(input_xyz: str, calc: CalculationParameters):
        if not calc.perform_geometryoptimization:
            print(f'> No geometryoptimization will be performed because perform_geometryoptimization is set to {calc.perform_geometryoptimization}.')
            return None
        xtb = Xtb()
        energy = xtb.geomopt(input_xyz)
        print(f'> xTB geometry optimization of {input_xyz} finished with a final energy of {round(energy, 5)}.')
        print(f'> New xTB geometry is written to {input_xyz}')
        return energy


# class HessianCalculation:

#     @staticmethod
#     def with_xtb(input_xyz: str, calc: CalculationParameters):
#         if not calc.perform_hesscalculation:
#             print(f'> No geometryoptimization will be performed because perform_geometryoptimization is set to {calc.perform_hesscalculation}.')
#             return None
#         xtb = Xtb()
#         energy = xtb.hesscalc(input_xyz)
#         print(f'> xTB geometry optimization of {input_xyz} finished with a final energy of {energy}.')
#         print(f'> New xTB geometry is written to {input_xyz}')
#         return energy