import subprocess
import sys
import tempfile
import shutil
import os
from pathlib import Path
import pytest

# Code which calls ffits as a subprocess
def run_ffits_as_subprocess(args):
    command = 'ffits ' + ' '.join(args)
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode



def test_ffits_standard_tsguess_run():
    """
    Test that ffits creates optimized.xyz, ff.csv, etc when run with reac.xyz and prod.xyz with standard configuration
    """  
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for item in examples_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
        
        original_cwd = os.getcwd()
        os.chdir(temp_path)
        os.rename(temp_path / "struc1.xyz", temp_path / "reac.xyz")
        os.rename(temp_path / "struc2.xyz", temp_path / "prod.xyz")

        try:
            stdout, stderr, returncode = run_ffits_as_subprocess(['reac.xyz', 'prod.xyz'])
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            assert returncode == 0, f"ffits failed with return code {returncode}\nstderr: {stderr}"
            
            optimized_file = temp_path / "optimized.xyz"
            assert optimized_file.exists(), f"optimized.xyz was not created in {temp_path}"
            
            ff_csv_file = temp_path / "tsff.csv"
            assert ff_csv_file.exists(), f"tsff.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "ff1.csv"
            assert ff_csv_file.exists(), f"ff1.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "ff2.csv"
            assert ff_csv_file.exists(), f"ff2.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "wbo1"
            assert ff_csv_file.exists(), f"wbo1 was not created in {temp_path}"
            ff_csv_file = temp_path / "wbo2"
            assert ff_csv_file.exists(), f"wbo2 was not created in {temp_path}"
            ff_csv_file = temp_path / "struc1.hess"
            assert ff_csv_file.exists(), f"struc1.hess was not created in {temp_path}"
            ff_csv_file = temp_path / "struc2.hess"
            assert ff_csv_file.exists(), f"struc2.hess was not created in {temp_path}"
        finally:
            os.chdir(original_cwd)


def test_ffits_with_filename_change_config():
    """
    Test that ffits creates optimized.xyz, ff.csv, etc when run with reac.xyz and prod.xyz with custom toml input
    """  
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for item in examples_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
        
        original_cwd = os.getcwd()
        os.chdir(temp_path)
        os.rename(temp_path / "struc1.xyz", temp_path / "reac.xyz")
        os.rename(temp_path / "struc2.xyz", temp_path / "prod.xyz")
        
        custom_input = """
[ts_calc]
factor_reactant = 0.7
factor_product = 0.3

[reactant.path]
xyz_filename = "struc1.xyz"
wbo_filename = "wbooo1"
hessian_filename = "struc1test.hess"
ff_filename = "ff1test.csv"

[product.path]
xyz_filename = "struc2.xyz"
wbo_filename = "wbooo2"
hessian_filename = "struc2test.hess"
ff_filename = "ff2test.csv"

"""
        with open(temp_path / "custom_config.toml", 'w') as f:
            f.write(custom_input)
        try:
            stdout, stderr, returncode = run_ffits_as_subprocess(['reac.xyz', 'prod.xyz', '--config', 'custom_config.toml'])
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            assert returncode == 0, f"ffits failed with return code {returncode}\nstderr: {stderr}"
            
            optimized_file = temp_path / "optimized.xyz"
            assert optimized_file.exists(), f"optimized.xyz was not created in {temp_path}"
            
            ff_csv_file = temp_path / "tsff.csv"
            assert ff_csv_file.exists(), f"tsff.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "ff1test.csv"
            assert ff_csv_file.exists(), f"ff1test.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "ff2test.csv"
            assert ff_csv_file.exists(), f"ff2test.csv was not created in {temp_path}"
            ff_csv_file = temp_path / "wbooo1"
            assert ff_csv_file.exists(), f"wbooo1 was not created in {temp_path}"
            ff_csv_file = temp_path / "wbooo2"
            assert ff_csv_file.exists(), f"wbooo2 was not created in {temp_path}"
            ff_csv_file = temp_path / "struc1test.hess"
            assert ff_csv_file.exists(), f"struc1test.hess was not created in {temp_path}"
            ff_csv_file = temp_path / "struc2test.hess"
            assert ff_csv_file.exists(), f"struc2test.hess was not created in {temp_path}"
        finally:
            os.chdir(original_cwd)



        
        # # Create malformed XYZ files (missing atom count or invalid format)
        # bad_xyz1 = temp_path / "bad1.xyz"
        # bad_xyz2 = temp_path / "bad2.xyz"
        
        # bad_xyz1.write_text("not_a_number\ncomment line\nH 0 0 0\n")
        # bad_xyz2.write_text("2\ncomment\nH 0 0 0\n")  # Says 2 atoms but only has 1
        
        # original_cwd = os.getcwd()
        # os.chdir(temp_path)
        
        # try:
        #     stdout, stderr, returncode = run_ffits_as_subprocess(['bad1.xyz', 'bad2.xyz'])
        #     assert returncode != 0, "ffits should fail with malformed XYZ files"
        # finally:
        #     os.chdir(original_cwd)


# def test_ffits_preserves_atom_count():
#     """
#     Test that the optimized structure has the same number of atoms as input
#     """
#     examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_path = Path(temp_dir)
#         for item in examples_dir.iterdir():
#             if item.is_file():
#                 shutil.copy2(item, temp_path / item.name)
#             elif item.is_dir():
#                 shutil.copytree(item, temp_path / item.name)
        
#         original_cwd = os.getcwd()
#         os.chdir(temp_path)
        
#         try:
#             # Read input atom count
#             with open('struc1.xyz') as f:
#                 input_natoms = int(f.readline().strip())
            
#             stdout, stderr, returncode = run_ffits_as_subprocess(['struc1.xyz', 'struc2.xyz'])
#             assert returncode == 0
            
#             # Read output atom count
#             with open(temp_path / "optimized.xyz") as f:
#                 output_natoms = int(f.readline().strip())
            
#             assert output_natoms == input_natoms, \
#                 f"Atom count changed: input {input_natoms}, output {output_natoms}"
#         finally:
#             os.chdir(original_cwd)



# 34 40 67


# TODO dieser Test wird fertig gemacht, wenn die temporary directories printbar sind 
# def test_ffits_charge_multiplicity_passed_to_xtb():
#     """
#     Test that ffits correctly passes charge and multiplicity to xTB
#     by reading the xTB output file and verifying the values.
#     """
#     examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_path = Path(temp_dir)
#         for item in examples_dir.iterdir():
#             if item.is_file():
#                 shutil.copy2(item, temp_path / item.name)
#             elif item.is_dir():
#                 shutil.copytree(item, temp_path / item.name)
        
#         original_cwd = os.getcwd()
#         os.chdir(temp_path)
        
#         try:
#             # Run ffits with specific charge and multiplicity
#             charge = 2
#             multiplicity = 2
#             stdout, stderr, returncode = run_ffits_as_subprocess([
#                 'struc1.xyz', 'struc2.xyz', 
#                 '--charge', str(charge), 
#                 '--multiplicity', str(multiplicity)
#             ])
            
#             if returncode == 0:
#                 # Look for xTB output file (may be in subdirectories or current dir)
#                 xtb_out_file = None
#                 for root, dirs, files in os.walk(temp_path):
#                     if 'xtb.out' in files:
#                         xtb_out_file = Path(root) / 'xtb.out'
#                         break
                
#                 with open(Path(temp_path, 'xtb.out'), 'r') as f:
#                     content = f.read()
                    
#                     # Extract net charge from xTB output
#                     net_charge = None
#                     unpaired_electrons = None
                    
#                     for line in content.split('\n'):
#                         if ':  net charge' in line:
#                             parts = line.split()
#                             net_charge = int(parts[-2])
#                         if ':  unpaired electrons' in line:
#                             parts = line.split()
#                             unpaired_electrons = int(parts[-2])
#                     print(net_charge, unpaired_electrons)
#                     assert False
#                     assert net_charge is not None, "Could not find net charge in xtb.out"
#                     assert net_charge == charge, f"Expected charge {charge}, but xTB got {net_charge}"
                    
#                     expected_unpaired = multiplicity - 1
#                     assert unpaired_electrons is not None, "Could not find unpaired electrons in xtb.out"
#                     assert unpaired_electrons == expected_unpaired, \
#                         f"Expected unpaired electrons {expected_unpaired} (multiplicity {multiplicity}), but xTB got {unpaired_electrons}"
#         finally:
#             os.chdir(original_cwd)

