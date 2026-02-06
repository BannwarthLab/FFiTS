import subprocess
import tempfile
import shutil
import os
from pathlib import Path
import pytest
from ffits.io.reader import readin_xyz
from ffits.utils.rmsd import kabsch_rmsd

# Code which calls ffits as a subprocess
def run_ffits_as_subprocess(args):
    command = 'ffits ' + ' '.join(args)
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode




def test_ffits_optmode():
    """
    Test which first calculates with present hessian and then without hessian and wbo calculation but reading in the files created in the first run, by changing the filenames in the config file
    """  
    examples_dir = Path(__file__).parent.parent / "examples" / "small_single_molecule"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copy2(examples_dir / 'struc1.xyz', temp_path / "reac.xyz")
        shutil.copy2(examples_dir / 'struc2.xyz', temp_path / "prod.xyz")
        
        original_cwd = os.getcwd()
        os.chdir(temp_path)
        
        try:
            # run ffits normally to generate ff files
            stdout, stderr, returncode = run_ffits_as_subprocess(['reac.xyz', 'prod.xyz'])
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            assert returncode == 0, f"ffits failed with return code {returncode}\nstderr: {stderr}"
            _, _, prod_xyz, _ = readin_xyz(temp_path / "prod.xyz")
            _, _, reac_xyz, _ = readin_xyz(temp_path / "reac.xyz")
            
            os.remove(temp_path / "optimized.xyz")

            # list every file in temp_path
            for file in temp_path.iterdir():
                print(file.name)

            # opt from reac to prod
            stdout, stderr, returncode = run_ffits_as_subprocess(['reac.xyz', '--opt', 'ff2.csv'])
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            _, _, optprod_xyz, _ = readin_xyz(temp_path / "optimized.xyz")
            assert kabsch_rmsd(prod_xyz, optprod_xyz) < 5e-2, "Optimized geometries differ between runs with and without hessian calculation."


            # with config and readin
            stdout, stderr, returncode = run_ffits_as_subprocess(['prod.xyz', '--opt', 'ff1.csv'])
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            _, _, optreac_xyz, _ = readin_xyz(temp_path / "optimized.xyz")
            
            # compare both ts guess through rmsd
            assert kabsch_rmsd(reac_xyz, optreac_xyz) < 5e-2, "Optimized geometries differ between runs with and without hessian calculation."

        finally:
            os.chdir(original_cwd)
