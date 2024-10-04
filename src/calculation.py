from src.interface.xtb import *
from src.datatype.calculation_data import *
from src.datatype.structure_data import *
from src.input_library import *
from src.interface.crest import *
import copy
import shutil

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
    def with_ff_potential_crest(input_structure: StructurePath, ff: ForceField, output_xyz=''):
        if output_xyz == '':
            output_xyz = input_structure.xyz_filename
        crest = Crest()
        new_ff_name = Name.modified_ff(input_structure.ff_filename)
        ff.write_force_field(new_ff_name)
        inpu_struc = copy.deepcopy(input_structure)
        inpu_struc.ff_filename = new_ff_name
        input = input_ff_optimization(starting_struc=input_structure.xyz_filename, useff=True, struc=input_structure)
        crest_input_filename = 'input_ff_opt.toml'
        crest.write_input2file(input, crest_input_filename)
        rc = crest.run_input(crest_input_filename, 'ff_fit')
        shutil.copy('crestopt.xyz', output_xyz)
        print(f'> FF based geometry optimization for {input_structure.xyz_filename} finished.')
        return rc




class HessianCalculation:
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path='xtb'):
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        xtb.hesscalc(input_structure.hess_filename, input_xyz)
        print(f'> xTB Hessian calculation for {input_xyz} finished.')
        print(f'> Hessian is written to {input_structure.hess_filename}.')


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
        print(f'> FF is written to {Name.fitted_ff(structure_for_fit.ff_filename)}.')
        return ForceField(nat, Name.fitted_ff(structure_for_fit.ff_filename))


class TsGuessCalculation:
    @staticmethod
    def with_crest(start_struc: str, calc_params):
        crest = Crest()
        input = input_ts_search(start_struc, calc)
        crest_input_filename = 'input_tssearch.toml'
        crest.write_input2file(input, crest_input_filename)
        rc = crest.run_input(crest_input_filename, 'tssearch')
        # TODO postprocessing 
        