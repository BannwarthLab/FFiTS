#!/bin/python

import subprocess
import os 
from input_library import input_avff1, input_avff2, input_ff_calc, input_start1, input_start2, input_ff_opt1, input_ff_opt2, input_avff_optimize


    
class Crest:
    def __init__(self) -> None:
        pass
    

    def get_rmsd(self, xyz1: str, xyz2: str) -> float:
        print('rmsd calc')
        command = "crest --rmsd " + xyz1 + " " + xyz2 
        with open('rmsd.out', 'w') as stdout_file, open('rmsd_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            if rc == 0:
                with open('rmsd.out', 'r') as f:
                    lines = f.readlines()
                return float(lines[-1].split(' ')[-1])
            else:
                raise Exception("An Error happend during the RMSD calculation, with the return code ", rc)

    def write_input2file(self, input: str, filename):
        with open(filename, 'w') as file:
            file.write(input)

    def calc_fitted_ff(self, input_filename: str):
        command = "/home/guests/dbabushkina/master_thesis/crest/_build/crest -i " + input_filename
        with open('fitff.out', 'w') as stdout_file, open('fitff_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            if rc == 0:
                print('Fitted FF calculated to from ', input_filename)
                return rc
            else:
                raise Exception("An Error happend during the FF fitting ", rc)

    def run_ts_search(self, startstruc: int, path_to_crest: str, input_filename: str):
        if startstruc == 1:
            self.write_input2file(input_avff1, input_filename)
        elif startstruc == 2:
            self.write_input2file(input_avff2, input_filename)
        elif startstruc == -1:
            self.write_input2file(input_start1, input_filename)
        elif startstruc == -2:
            self.write_input2file(input_start2, input_filename)
        elif startstruc == 0:
            self.write_input2file(input_avff_optimize, input_filename)
        else:
            raise Exception("Startstruc can only be 1, 2, -1 or -2.")
        
        command = path_to_crest + " -i " + input_filename
        with open('tssearch.out', 'w') as stdout_file, open('tssearch_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
        if os.path.exists('crestopt.xyz'):
            print('IT DOES EXIST')
            return 0
        else:
            return 20
                # raise Exception("An Error happend during the TS search, with the return code ", rc)
        

### testing

# if __name__ == '__main__':
#     d = Crest()
#     d.write_input2file(input_ff_calc, 'inptest.toml')

