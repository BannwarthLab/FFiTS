
import subprocess
import os
import shutil
from src.calculation import TsGuessCalculation, GeometryOptimization, HessianCalculation, WboCalculation
from src.datatype.calculation_data import CalculationParameters, Reaction
from src.datatype.structure_data import Structure, ForceField
from src.interface.exceptions import ConvergenceError
from src.interface.reader import readin_xyz
from src.interface.printer import generate_xtb_fix_input, write_string2file


def constrained_optimization(reac: Reaction, calc: CalculationParameters):
     if not calc.perform_relaxation:
          return None
     write_string2file(string=generate_xtb_fix_input(reac.unique_atoms), filename='xtb.inp')
     shutil.copy('final.xyz', 'tsff_final.xyz')
     GeometryOptimization.constrained_with_xtb('final.xyz')


def search_ts(struc1: Structure, struc2: Structure, calc: CalculationParameters):
     try: #TODO i don't check wether i want to do the calculation
          nat, energy, xyz = TsGuessCalculation.with_crest(struc1.path.xyz_filename, calc)
          ts_ff = ForceField(nat, calc.ts.ff_filename)
          os.rename('crestopt.log', 'crestopt1.log')
     except ConvergenceError: #NAME
          os.rename('crestopt.log', 'crestopt1.log')
          os.rename('tssearch.out', 'tssearch1.out')
          os.rename('tssearch_err.out', 'tssearch_err1.out')
          try: 
               nat, energy, xyz = TsGuessCalculation.with_crest(struc2.path.xyz_filename, calc)
               ts_ff = ForceField(nat, calc.ts.ff_filename)
               os.rename('crestopt.log', 'crestopt2.log')
          except ConvergenceError:
               os.rename('crestopt.log', 'crestopt2.log')
               p = subprocess.Popen(f'tail crestopt1.log -n {str(struc1.info.nat+2)} > ts.xyz', shell=True) #NAME
               p.wait()
               nat, energy, xyz = readin_xyz('ts.xyz', False)
               os.rename('ts.xyz', calc.ts.xyz_filename)
               print(f'!!> Both CREST runs did not converge. Last structure of crestopt1.log is used for crestopt.xyz.') 
               ts_ff = ForceField(nat, calc.ts.ff_filename)

     print(f"> TS Guess is optimized a final time with TS FF potential in {calc.ts.ff_filename}.")
     os.rename('ts.xyz', calc.ts.xyz_filename)
     HessianCalculation.with_xtb(calc.ts)
     WboCalculation.with_xtb(calc.ts)
     GeometryOptimization.with_ff_potential_crest(calc.ts, ts_ff)

     

     return ts_ff
          
     



