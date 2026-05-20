import numpy as np


def compute_com(coordinates):
    """Compute geometric center (center of mass if all atoms have equal mass)."""
    return coordinates.mean(axis=0)


## From COMET
def kabsch_rmsd(xyz1, xyz2):
    xyz1 -= xyz1.mean(axis=0)
    xyz2 -= xyz2.mean(axis=0)

    R = xyz1.transpose() @ xyz2
    U, _, Vt = np.linalg.svd(R)
    V = Vt.transpose()
    Ut = U.transpose()
    d = int(np.sign(np.linalg.det(V @ Ut)))
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
    R = V @ mat @ Ut
    xyz2_aligned = xyz2 @ R
    rmsd = np.sqrt(np.mean(np.sum((xyz1 - xyz2_aligned) ** 2, axis=1)))
    return rmsd
