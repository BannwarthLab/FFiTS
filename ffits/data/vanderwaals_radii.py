import numpy as np
from ffits.data.elements import atom_symbol_to_number
from ffits.utils.geometry import angstrom2bohr

VANDER_VALUES = np.array(
    [
        0.91,
        0.92,  # H, He
        0.75,
        1.28,
        1.35,
        1.32,
        1.27,
        1.22,
        1.17,
        1.13,  # Li-Ne
        1.04,
        1.24,
        1.49,
        1.56,
        1.55,
        1.53,
        1.49,
        1.45,  # Na-Ar
        1.35,
        1.34,  # K, Ca
        1.42,
        1.42,
        1.42,
        1.42,
        1.42,  # Sc-Zn
        1.42,
        1.42,
        1.42,
        1.42,
        1.42,
        1.50,
        1.57,
        1.60,
        1.61,
        1.59,
        1.57,  # Ga-Kr
        1.48,
        1.46,  # Rb, Sr
        1.49,
        1.49,
        1.49,
        1.49,
        1.49,  # Y-Cd
        1.49,
        1.49,
        1.49,
        1.49,
        1.49,
        1.52,
        1.64,
        1.71,
        1.72,
        1.72,
        1.71,  # In-Xe
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,  # La-Yb
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,  # Lu-Hg
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,
        2.00,  # Tl-Rn
    ]
)


def get_vander_matrix(at: np.ndarray, vander_values=VANDER_VALUES, factor=1):
    """
    Build van der Waals interaction matrix. Automatically scales to bohr.

    Args:
        at (np.ndarray): Array of atomic symbols (length nat).
        vander_values (np.ndarray): Reference van der Waals radii (length 86).
        factor (float): Scaling factor.

    Returns
    -------
    vander_matrix : np.ndarray (nat x nat)
    """
    # Convert atomic numbers to 0-based indices
    radii = np.array(
        [angstrom2bohr(vander_values[atom_symbol_to_number(sym) - 1]) * factor for sym in at]
    )

    # Build the full symmetric matrix (outer sum)
    vander_matrix = radii[:, None] + radii[None, :]

    return vander_matrix
