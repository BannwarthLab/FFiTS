
import subprocess
import sys
import tempfile
import shutil
import os
from pathlib import Path

# Code which calls ffits as a subprocess
def run_ffits_as_subprocess(args):
    command = 'ffits ' + ' '.join(args)
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def test_ffits_creates_optimized_xyz():
    """
    Test that ffits creates optimized.xyz when run with struc1.xyz and struc2.xyz
    """  
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for item in examples_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, temp_path / item.name)
        
        original_cwd = os.getcwd()
        os.chdir(temp_path)
        
        try:
            stdout, stderr, returncode = run_ffits_as_subprocess(['struc1.xyz', 'struc2.xyz'])
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            assert returncode == 0, f"ffits failed with return code {returncode}\nstderr: {stderr}"
            
            optimized_file = temp_path / "optimized.xyz"
            assert optimized_file.exists(), f"optimized.xyz was not created in {temp_path}"
        finally:
            # Change back to the original directory
            os.chdir(original_cwd)