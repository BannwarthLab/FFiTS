"""Atom-type dependent bond order thresholds.

Bonds between two atoms of the first or second period (Z <= 10, i.e. H to Ne)
use a fixed, stricter threshold. All other pairs (at least one atom from the
third period or higher) use the threshold supplied by the caller, usually the
``bo_treshold`` value from the config file.
"""

import numpy as np
import logging
from ffits.data.elements import atom_symbol_to_number

logger = logging.getLogger(__name__)


def get_bo_threshold_matrix(
    atom_types,
    default_threshold: float = 0.0,
    light_threshold: float = 0.3,
    light_atom_max_z: int = 10,
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
        [atom_symbol_to_number(str(a)) <= light_atom_max_z for a in atom_types],
        dtype=bool,
    )
    logger.debug(
        f"BO threshold matrix is build with light_treshold of {light_threshold} and bo_threshold of {default_threshold}."
    )
    both_light = np.outer(light, light)
    return np.where(both_light, light_threshold, default_threshold).astype(float)
