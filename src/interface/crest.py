#!/bin/python

import subprocess
import os 
# from src.input_library import input_ff_optimization, input_ts_search


    
class Crest:
    '''
    CREST caller and CREST specific operations.
    '''
    def __init__(self, crest_path="/home/dbabushkina/1_ts_search2024/crest/_build/crest") -> None:
        self.crest_path = crest_path
    

    def check_convergence(self, filepath: str, keyword="geometry successfully optimized"):
        '''
        checks convergence by finding keyword in output saved in 'filepath'
        '''
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if keyword in line:
                return True
        return False

    def get_rmsd(self, xyz1: str, xyz2: str) -> float:
        '''
        runs an RMSD calculation and returns the RMSD between structure in paths 'xyz1' and 'xyz2'
        '''
        print('rmsd calc')
        command = f"{self.crest_path} --rmsd {xyz1} {xyz2}"
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
        '''
        writes input string 'input' to 'filename'.
        '''
        with open(filename, 'w') as file:
            file.write(input)

    def run_input(self, input_file: str, output_name: str):
        '''
        runs CREST input from 'input_file' and saves resulting output of the calculation in 'output_name'.out and 'output_name'_err.out
        '''
        command = f"{self.crest_path} -i {input_file}"
        with open(f'{output_name}.out', 'w') as stdout_file, open(f'{output_name}_err.out', 'w') as stderr_file:           
            if os.path.exists('.UHF'):
                os.rename('.UHF', 'tuhf')
            if os.path.exists('.CHRG'):
                os.rename('.CHRG', 'tchrg')

            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode  
                 
            if os.path.exists('tuhf'):
                os.rename('tuhf','.UHF')
            if os.path.exists('tchrg'):
                os.rename( 'tchrg','.CHRG')
            if rc == 0:
                return rc
            else:
                raise Exception(f"An Error happend during the CREST calculation with rc {rc}.\n For further information look in {output_name}_err.out.")

        
