#!/bin/python

import subprocess
import os
from src.interface.readin import read_wbo_file, readin_xyz

class Xtb:
    def __init__(self, xtb_path='xtb') -> None:
        # maybe define all names here instead of giving them in func
        self.xtb_path = xtb_path

    def get_command(self, input, keyword: str) -> str:
        return f"{self.xtb_path} {input} {keyword}"
    
    def find_energy_in_output(self, output_filename: str) -> float:
        with open(output_filename, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "TOTAL ENERGY" in line:
                return float(line.split()[3])
        raise Exception("No Energy was found in output of the xTB singlepoint calculation.")
        
        
    def singlepoint(self, input_xyz: str) -> float:
        command = "xtb " + input_xyz 
        with open('xtb_single.out', 'w') as stdout_file, open('xtb_single_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            return self.find_energy_in_output("xtb_single.out")
            if rc != 0:
                raise Exception("An Error happend during the xTB geometry optimization, with the return code ", rc)
    

    def geomopt(self, input_xyz: str) -> float:
        #return xtb energy 
        command = self.get_command(input_xyz, '--opt') 
        with open('xtb.out', 'w') as stdout_file, open('xtb_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            if rc == 0:
                os.rename('xtbopt.xyz', input_xyz)
                return readin_xyz(input_xyz)
            else:
                raise Exception("An Error happend during the xTB geometry optimization, with the return code ", rc)
        
    def hesscalc(self, hess_filename, input_xyz: str) -> None:
        #creates hess file after xtb hess call
        command = self.get_command(input_xyz, '--hess') 
        with open('xtbhess.out', 'w') as stdout_file, open('xtbhess_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            if rc == 0:
                if os.path.isfile("hessian"):
                    os.rename('hessian', hess_filename)   
                    # print("Hessian for calculation of", input_xyz, "was generated to file", hess_filename)
                else:
                    raise Exception("No Hessian was generated")
            else:
                raise Exception("An Error happend during the xTB Hessian calculation, with the return code ", rc)
    


    def wbocalc(self, wbo_filename, input_xyz: str) -> dict:
        # creates wbo file
        command = self.get_command(input_xyz, '--wbo') 
        with open('xtbwbo.out', 'w') as stdout_file, open('xtbwbo_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            if rc == 0:
                if os.path.isfile("wbo"):
                    os.rename("wbo", wbo_filename)
                else:
                    raise Exception("No WBO was generated")
            else:
                raise Exception("An Error happend during the WBO calculation, with the return code ", rc)
            return read_wbo_file(wbo_filename)
            
    def get_negative_frequencies(self, vibspectrum_filename, input_xyz: str) -> list:
        command = "xtb " + input_xyz + " --hess"
        with open('xtbhess.out', 'w') as stdout_file, open('xtbhess_err.out', 'w') as stderr_file:
            p = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, shell=True)
            p.wait()
            rc = p.returncode
            if rc == 0:
                if os.path.isfile("vibspectrum"):
                    os.rename('vibspectrum', vibspectrum_filename)   
                    with open(vibspectrum_filename, 'r') as f:
                        lines = f.readlines()
                    neg_freq = []
                    for i in range(3, len(lines)):
                        val = float(lines[i].split()[2])
                        if val < -10:
                            neg_freq.append(val)
                            print(val)
                        elif val > 10: 
                            break 
                    return neg_freq
                else:
                    raise Exception("No Vibspectrum was generated")
            else:
                raise Exception("An Error happend during the xTB Hessian and vibrations calculation, with the return code ", rc)
    
