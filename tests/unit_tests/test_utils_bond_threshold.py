import numpy as np

from ffits.utils.bond_threshold import get_bo_threshold_matrix


def test_light_pairs_use_fixed_threshold_others_use_default():
    m = get_bo_threshold_matrix(["C", "H", "S", "Ne", "Na"], default_threshold=0.1)
    assert m[0, 1] == 0.3  # C-H
    assert m[0, 3] == 0.3  # C-Ne (period 2)
    assert m[0, 2] == 0.1  # C-S (period 3)
    assert m[2, 4] == 0.1  # S-Na
    assert m[1, 4] == 0.1  # H-Na
    assert np.allclose(m, m.T)


def test_default_zero_and_case_insensitive():
    m = get_bo_threshold_matrix(["c", "CL"], default_threshold=0.0)
    assert m[0, 1] == 0.0
    assert m[0, 0] == 0.3
