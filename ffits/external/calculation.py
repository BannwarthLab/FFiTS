from ffits.io.xtb import *
from ffits.datatype.calculation_data import *
from ffits.datatype.structure_data import *
from ffits.external.input_library import *
from ffits.io.crest import *
import copy
import shutil
import os
from ffits.io.exceptions import ConvergenceError
import subprocess

# @dataclass
class GeometryOptimization: 
    ''' 
    performs a geometry optimization with the specified method. 
    '''
        # TODO add a wbo calc and check to check whether topology changes and issue a warning
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path='xtb'):
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        nat, energy, xyz = xtb.geomopt(input_xyz)
        print(f'> xTB geometry optimization of {input_xyz} finished with a final energy of {round(energy, 5)}.')
        print(f'> New xTB geometry is written to {input_xyz}.')
        if os.path.exists('xtbrestart'):
            os.remove('xtbrestart')
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
        rc = crest.run_input(crest_input_filename, 'ff_opt') # NAME
        if crest.check_convergence('ff_opt.out'):  
            shutil.copy('crestopt.xyz', output_xyz)
            print(f'> FF based geometry optimization for {input_structure.xyz_filename} finished.')
            return rc
        print(f'!!> ERROR FF based geometry optimization for {input_structure.xyz_filename} did not converge. Last structure of crestopt.log is used.')
        p = subprocess.Popen(f'tail crestopt.log -n {str(ff.nat+2)} > {input_structure.xyz_filename}', shell=True) #NAME
        p.wait()
        raise ConvergenceError('Not converged')

    @staticmethod
    def constrained_with_xtb(input_filename,  xtb_path='xtb'):
        xtb = Xtb(xtb_path=xtb_path)
        nat, energy, xyz = xtb.geomopt(f'--input xtb.inp {input_filename}')
        os.rename(f'--input xtb.inp {input_filename}',input_filename)
        print(f'> Constrained xTB geometry optimization of {input_filename}  with constrained atoms  finished with a final energy of {round(energy, 5)}.')
        print(f'> New xTB geometry is written to {input_filename}.')
        if os.path.exists('xtbrestart'):
            os.remove('xtbrestart')
        return nat, energy, xyz


class HessianCalculation:
    ''' 
    performs a Hessian calculation with the specified method. 
    Output is in the xtb type format.
    '''
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path='xtb'):
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        xtb.hesscalc(input_structure.hessian_filename, input_xyz)
        print(f'> xTB Hessian calculation for {input_xyz} finished.')
        print(f'> Hessian is written to {input_structure.hessian_filename}.')


class WboCalculation:
    ''' 
    performs a WBO calculation with the specified method. 
    Output is in the xtb type format.
    '''
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path='xtb'):
        input_xyz = input_structure.xyz_filename
        xtb = Xtb(xtb_path=xtb_path)
        wbo_dict = xtb.wbocalc(input_structure.wbo_filename, input_xyz)
        print(f'> xTB WBO calculation for {input_xyz} finished.')
        print(f'> WBO is written to {input_structure.wbo_filename}.')
        return wbo_dict

class FittedFfGeneration:
    '''
    generates a force field file, containing information on the fitted FF
    '''
    @staticmethod
    def with_crest(nat, start_structure: str, structure_for_fit: StructurePath):
        crest = Crest()
        print(f'> FF Fitting for {structure_for_fit.xyz_filename} starts ...')
        input = input_ff_optimization(starting_struc=start_structure, useff=False,struc=structure_for_fit)
        crest_input_filename = 'input_ff_fit.toml'
        crest.write_input2file(input, crest_input_filename)
        rc = crest.run_input(crest_input_filename, 'ff_fit')
        print(f'> FF Fitting for {structure_for_fit.xyz_filename} finished.')
        print(f'> FF is written to {Name.fitted_ff(structure_for_fit.ff_filename)}.')
        return ForceField(nat, Name.fitted_ff(structure_for_fit.ff_filename))


class TsGuessCalculation:
    '''
    performs TS guess calculation with specified method.
    '''
    @staticmethod
    def with_crest(start_struc: str, calc: CalculationParameters):
        crest = Crest()
        input = input_ts_search(start_struc, calc)
        crest_input_filename = 'input_tssearch.toml'
        print(f'> TS guess Calculation from {start_struc} starts.')
        crest.write_input2file(input, crest_input_filename)
        rc = crest.run_input(crest_input_filename, 'tssearch')  # NAME
        converged = crest.check_convergence('tssearch.out') 
        if converged: 
            print(f'> TS guess Calculation from {start_struc} converged.')
            shutil.copy('crestopt.xyz', 'ts.xyz')  # NAME
            return readin_xyz('ts.xyz', rdenergy=False)
        else: 
            print(f'!!> Run did not converge.') 
            raise ConvergenceError('Not converged') 
            