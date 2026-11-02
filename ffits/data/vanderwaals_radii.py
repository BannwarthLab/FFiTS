"""Van der Waals radii data and helpers for building pairwise radii matrices.

Holds the ``VANDER_VALUES`` reference table (elements 1-86) and
:func:`get_vander_matrix`, which expands it into a per-atom-pair matrix
for a given structure.
"""

import numpy as np
from ffits.data.elements import atom_symbol_to_number
from ffits.utils.geometry import angstrom2bohr
# COVALENT RADII  taken from excprep based on
# "Atomic Radii of the Elements, M. Mantina, R. Valero, C. J. Cramer, and D. G. Truhlar,
# in CRC Handbook of Chemistry and Physics, 91st Edition (2010-2011),
# edited by W. M. Haynes (CRC Press, Boca Raton, FL, 2010), pages 9-49-9-50;
# corrected Nov. 17, 2010 for the 92nd edition.
ATOMIC_RADII_TRUHLAR = np.array(
    [  # 1 to 118 from excprep
        0.32,
        0.37,
        1.30,
        0.99,
        0.84,
        0.75,
        0.71,
        0.64,
        0.60,
        0.62,
        1.60,
        1.40,
        1.24,
        1.14,
        1.09,
        1.04,
        1.00,
        1.01,
        2.00,
        1.74,
        1.59,
        1.48,
        1.44,
        1.30,
        1.29,
        1.24,
        1.18,
        1.17,
        1.22,
        1.20,
        1.23,
        1.20,
        1.20,
        1.18,
        1.17,
        1.16,
        2.15,
        1.90,
        1.76,
        1.64,
        1.56,
        1.46,
        1.38,
        1.36,
        1.34,
        1.30,
        1.36,
        1.40,
        1.42,
        1.40,
        1.40,
        1.37,
        1.36,
        1.36,
        2.38,
        2.06,
        1.94,
        1.84,
        1.90,
        1.88,
        1.86,
        1.85,
        1.83,
        1.82,
        1.81,
        1.80,
        1.79,
        1.77,
        1.77,
        1.78,
        1.74,
        1.64,
        1.58,
        1.50,
        1.41,
        1.36,
        1.32,
        1.30,
        1.30,
        1.32,
        1.44,
        1.45,
        1.50,
        1.42,
        1.48,
        1.46,
        2.42,
        2.11,
        2.01,
        1.90,
        1.84,
        1.83,
        1.80,
        1.80,
        1.73,
        1.68,
        1.68,
        1.68,
        1.65,
        1.67,
        1.73,
        1.76,
        1.61,
        1.57,
        1.49,
        1.43,
        1.41,
        1.34,
        1.29,
        1.28,
        1.21,
        1.22,
        1.36,
        1.43,
        1.62,
        1.75,
        1.65,
        1.57,
    ],
    dtype=float,
)

# van der Waals radii in Angstroms for elements 1-86 (H to Rn) are taken from the program CREST
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

    Returns:
        vander_matrix (np.ndarray): The van der Waals interaction matrix (nat,nat).
    """
    # Convert atomic numbers to 0-based indices
    radii = np.array(
        [
            angstrom2bohr(vander_values[atom_symbol_to_number(sym) - 1]) * factor
            for sym in at
        ]
    )

    # Build the full symmetric matrix (outer sum)
    vander_matrix = radii[:, None] + radii[None, :]

    return vander_matrix
