from interface.call_xtb import *
# from interface.call_crest import *
from datatype.calculation_data import *
from datatype.structure_data import *
from input_library import *
from interface.call_crest import *

# class Calculation:
#     def __init__(self, input_xyz: str, calculation_data: CalculationParameters):
#         self.input_xyz = input_xyz
#         self.calc = calculation_data

# def 

#TODO make all new names a class with staticmethods so that they are fixed

# @dataclass
class GeometryOptimization:
        # TODO add a wbo calc and check to check whether topology changes and issue a warning
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path='xtb'):
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        nat, energy, xyz = xtb.geomopt(input_xyz)
        print(f'> xTB geometry optimization of {input_xyz} finished with a final energy of {round(energy, 5)}.')
        print(f'> New xTB geometry is written to {input_xyz}.')
        return nat, energy, xyz

    @staticmethod
    def with_ff_potential_crest(input_structure: StructurePath, ff: ForceField):
        crest = Crest()
        new_ff_name = f'{input_structure.ff_filename}_modified'
        ff.write_force_field(ff, new_ff_name)
        inpu_struc = input_structure.deepcopy()
        inpu_struc.ff_filename = new_ff_filename
        input = input_ff_optimization(starting_struc=inpu_struc.xyz_filename, useff=True, struc=inpu_struc)
        crest_input_filename = 'input_ff_opt.toml'
        crest.write_input2file(input, crest_input_filename)
        rc = crest.run_input(crest_input_filename, 'ff_fit')
        print(f'> FF based geometry optimization for {structure_for_fit.xyz_filename} finished.')
        print(f'> FF is written to {structure_for_fit.ff_filename}.')
        return rc




class HessianCalculation:
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path='xtb'):
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        xtb.hesscalc(input_structure.hess_filename, input_xyz)
        print(f'> xTB Hessian calculation for {input_xyz} finished.')
        print(f'> Hessian is written to {input_structure.hess_filename}.')
        return None


class WboCalculation:
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path='xtb'):
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        wbo_dict = xtb.wbocalc(input_structure.wbo_filename, input_xyz)
        print(f'> xTB WBO calculation for {input_xyz} finished.')
        print(f'> WBO is written to {input_structure.wbo_filename}.')
        return wbo_dict

class FittedFfGeneration:
    @staticmethod
    def with_crest(nat, start_structure: str, structure_for_fit: StructurePath):
        crest = Crest()
        input = input_ff_optimization(starting_struc=start_structure, useff=False,struc=structure_for_fit)
        crest_input_filename = 'input_ff_fit.toml'
        crest.write_input2file(input, crest_input_filename)
        rc = crest.run_input(crest_input_filename, 'ff_fit')
        print(f'> FF Fitting for {structure_for_fit.xyz_filename} finished.')
        print(f'> FF is written to {structure_for_fit.ff_filename}.')
        return ForceField(nat, structure_for_fit.ff_filename)


        