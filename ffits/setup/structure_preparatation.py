"""Structure preparation helpers (currently just coordinate randomization)."""
import logging


import numpy as np

logger = logging.getLogger(__name__)


def randomize_coordinates(
    xyz: np.ndarray, displacement: float = 0.01, random_seed: int = 42
) -> np.ndarray:
    """Randomly perturbs each atom's coordinates by a fixed displacement (random sign only).

    Useful for testing the robustness of TS guess generation against small
    perturbations in the input geometry.

    Args:
        xyz (np.ndarray): Original coordinates of the structure (shape: [nat, 3]).
        displacement (float, optional): Displacement magnitude for each atom. Defaults to 0.01.
        random_seed (int, optional): Seed for reproducibility of randomization. Defaults to 42.

    Returns:
        np.ndarray: Randomized coordinates of the structure (shape: [nat, 3]).
    """
    np.random.seed(random_seed)
    # randomize only the sign of the displacement, but the displacement itself is fixed to ensure consistency
    random_signs = np.random.choice([-1, 1], size=xyz.shape)  # Adjust scale as needed
    randomized_xyz = xyz + random_signs * displacement
    return randomized_xyz
