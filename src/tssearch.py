
import subprocess
import os
from src.calculation import TsGuessCalculation, GeometryOptimization, HessianCalculation, WboCalculation
from src.datatype.calculation_data import CalculationParameters
from src.datatype.structure_data import Structure, ForceField
from src.interface.exceptions import ConvergenceError
from src.interface.reader import readin_xyz



def search_ts(struc1: Structure, struc2: Structure, calc: CalculationParameters):
     try: 
          nat, energy, xyz = TsGuessCalculation.with_crest(struc1.path.xyz_filename, calc)
          ts_ff = ForceField(nat, calc.ts.ff_filename)
          os.rename('crestopt.log', 'crestopt1.log')
     except ConvergenceError: #NAME
          try: 
               nat, energy, xyz = TsGuessCalculation.with_crest(struc2.path.xyz_filename, calc)
               ts_ff = ForceField(nat, calc.ts.ff_filename)
               os.rename('crestopt.log', 'crestopt2.log')
          except ConvergenceError:
               p = subprocess.Popen(f'tail crestopt.log -n {str(struc1.info.nat+2)} > ts.xyz', shell=True) #NAME
               p.wait()
               nat, energy, xyz = readin_xyz('ts.xyz', False)
               os.rename('ts.xyz', calc.ts.xyz_filename)
               print(f'!!> CREST run did not converge. Last structure of crestopt.log is used for final.xyz.')
               return ForceField(nat, calc.ts.ff_filename)

     os.rename('ts.xyz', calc.ts.xyz_filename)
     HessianCalculation.with_xtb(calc.ts)
     WboCalculation.with_xtb(calc.ts)

     GeometryOptimization.with_ff_potential_crest(calc.ts, ts_ff)     
     

     return ts_ff
          
     



