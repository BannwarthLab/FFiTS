from ffits.io.file_writer import write_hessian_to_orcahessfile
import numpy as np
import tempfile


def test_write_hessian_to_orcahessfile():

    with tempfile.TemporaryDirectory() as temp_wd:
        nat = 7
        hessian = np.array(
            [[0.1 * i * j for j in range(nat * 3)] for i in range(nat * 3)]
        )
        xyz_with_masses = np.array(
            [
                "7",
                "C 12.01 0.000 0.000 0.000",
                "H 1.008 0.000 0.000 1.089",
                "H 1.008 1.026 0.000 -0.363",
                "H 1.008 -0.513 -0.889 -0.363",
                "O 15.999 1.200 0.000 0.000",
                "H 1.008 1.200 0.758 0.584",
                "H 1.008 1.200 -0.758 0.584",
            ]
        )
        filename = f"{temp_wd}/test_hessian.orcahess"
        write_hessian_to_orcahessfile(nat, hessian, xyz_with_masses, filename)
        with open(filename, "r") as f:
            content = f.read()
        assert "$orca_hessian_file" in content
        assert "$atoms" in content
        assert "$hessian" in content
        assert "$end" in content
        # todo reader für orca hess
