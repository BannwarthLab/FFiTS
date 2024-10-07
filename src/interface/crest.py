#!/bin/python

import subprocess
import os 
# from src.input_library import input_ff_optimization, input_ts_search


    
class Crest:
    def __init__(self, crest_path="/home/guests/dbabushkina//1_ts_search2024/crest/_build/crest") -> None:
        self.crest_path = crest_path
    

    def check_convergence(self, filepath: str):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "geometry successfully optimized" in line:
                return True
        return False

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

    def run_input(self, input_file: str, output_name: str):
        command = f"{self.crest_path} -i {input_file}"
        with open(f'{output_name}.out', 'w') as stdout_file, open(f'{output_name}_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            if rc == 0:
                return rc
            else:
                raise Exception(f"An Error happend during the CREST calculation with rc {rc}.\n For further information look in {output_name}_err.out.")

        
