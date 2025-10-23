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
    df1_["atoms_norm"] = df1_["atoms"].apply(canonical_tuple)
    df2_["atoms_norm"] = df2_["atoms"].apply(canonical_tuple)

    # Combine and drop duplicates
    combined = pd.concat([df1_, df2_], ignore_index=True)
    combined = combined.drop_duplicates(subset=["atoms_norm"], keep="first")

    # Replace original atom tuples with canonicalized ones
    combined["atoms"] = combined["atoms_norm"]
    combined = combined.drop(columns=["atoms_norm"])

    return combined.sort_values('atoms').reset_index(drop=True)

def remove_bonds_from_repulsive(df_source: pd.DataFrame, df_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows from df_source where the 'atoms' tuple is present
    in df_reference['atoms'].

    Both DataFrames must have a column named 'atoms' containing tuples.

    Parameters
    ----------
    df_source : pd.DataFrame
        DataFrame from which rows will be removed.
    df_reference : pd.DataFrame
        DataFrame providing the list of 'atoms' tuples to remove.

    Returns
    -------
    pd.DataFrame
        A new DataFrame identical to df_source, except rows with matching
        'atoms' tuples are removed.
    """
    if 'atoms' not in df_source.columns or 'atoms' not in df_reference.columns:
        raise ValueError("Both DataFrames must have an 'atoms' column.")

    # Convert to sets for O(1) lookup time per row
    reference_atoms = set(df_reference['atoms'])

    # Keep only rows whose 'atoms' are not in reference_atoms
    mask = ~df_source['atoms'].isin(reference_atoms)
    return df_source[mask].copy().reset_index(drop=True)
