import numpy as np
import pandas as pd
from src.forcefield.fortran_energy.geometry_calc import bondlength, angle, dihedral_angle

def combine_ff_terms(df1: pd.DataFrame, df2: pd.DataFrame, term_type: str = "bond") -> pd.Series:
    """
    Combine two force field term DataFrames (bonds, angles, dihedrals, LJ, etc.)
    without duplicating entries based on their atom connectivity.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Force field term DataFrames with an 'atoms' column (list or tuple of atom indices).
    term_type : str
        One of {"bond", "angle", "dihedral"}.
        Determines how atom tuples are compared (ordering vs symmetry).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with duplicate entries removed.
    """

    def canonical_tuple(atoms):
        """This function COULD normalize atom ordering, but decided not do that. """
        atoms = tuple(atoms)
        if term_type == "bond":
            return atoms # return tuple(sorted(atoms))  # symmetric
        elif term_type == "angle":
            return atoms  # directional
        elif term_type == "dihedral":
            # dihedral symmetry: (1,2,3,4) equivalent to reversed (4,3,2,1)
            # reversed_atoms = tuple(reversed(atoms))
            # return min(atoms, reversed_atoms)  
            return atoms
        elif term_type == "repulsive":
            return atoms #tuple(sorted(atoms))  # symmetric
        else:
            raise ValueError(f"Unknown term type '{term_type}'")

    # Make shallow copies and normalize atom tuples
    df1_ = df1.copy()
    df2_ = df2.copy()
    df1_["atoms_norm"] = df1_["atoms"]#.apply(canonical_tuple)
    df2_["atoms_norm"] = df2_["atoms"]#.apply(canonical_tuple)

    # Combine and drop duplicates
    combined = pd.concat([df1_, df2_], ignore_index=True)
    combined = pd.DataFrame(np.unique(combined["atoms_norm"]))

    # Replace original atom tuples with canonicalized ones
    combined["atoms"] = combined["atoms_norm"]
    combined = combined.drop(columns=["atoms_norm"])

    return combined.reset_index(drop=True)
