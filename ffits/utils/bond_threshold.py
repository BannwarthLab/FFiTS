"""Atom-type dependent bond order thresholds.

Bonds between two atoms of the first or second period (Z <= 10, i.e. H to Ne)
use a fixed, stricter threshold. All other pairs (at least one atom from the
third period or higher) use the threshold supplied by the caller, usually the
``bo_treshold`` value from the config file.
"""

import numpy as np

from ffits.data.elements import atom_symbol_to_number

LIGHT_ATOM_MAX_Z = 10  # Ne, last element of the second period
LIGHT_ATOM_BO_THRESHOLD = 0.3  # was 0.8, which cut real O-H bonds (WBO ~0.79)


def get_bo_threshold_matrix(
    atom_types,
    default_threshold: float = 0.0,
    light_threshold: float = LIGHT_ATOM_BO_THRESHOLD,
) -> np.ndarray:
    """Build a (nat, nat) matrix with the bond order threshold for every atom pair.

    Args:
        atom_types (Sequence[str]): Element symbols of all atoms.
        default_threshold (float, optional): Threshold for pairs where at least one atom
            is from the third period or higher (normally the config value). Defaults to 0.0.
        light_threshold (float, optional): Threshold for pairs where both atoms are from
            the first or second period. Defaults to 0.3.

    Returns:
        np.ndarray: Symmetric matrix of thresholds, shape (nat, nat).
    """
    light = np.array(
        [atom_symbol_to_number(str(a)) <= LIGHT_ATOM_MAX_Z for a in atom_types],
        dtype=bool,
    )
    both_light = np.outer(light, light)
    return np.where(both_light, light_threshold, default_threshold).astype(float)
