import numpy as np
import pandas as pd
from src.utils.geometry_calc import bondlength, angle, dihedral_angle
from src.datatype.structure_data import ForceField, StructuralInformation

def create_tsff(ff1: ForceField, info1: StructuralInformation, ff2: ForceField, info2: StructuralInformation) -> ForceField:

    # TODO add parameter transfer for averaging and add that in printout too
    tsff = ForceField(ff1.nat, 'tsff', readff=False)
    tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
    tsff.repulsive = remove_bonds_from_repulsive(combine_ff_atoms(ff1.repulsive, ff2.repulsive), combine_ff_atoms(ff1.bonds, ff2.bonds))

    mix_parameters(tsff.bonds, ff1.bonds, ff2.bonds)
    mix_parameters(tsff.angles, ff1.angles, ff2.angles)
    mix_parameters(tsff.dihedrals, ff1.dihedrals, ff2.dihedrals)
    mix_parameters(tsff.repulsive, ff1.repulsive, ff2.repulsive)
    
    mix_reference_values(tsff, ff1, ff2, info1, info2)
    print(f'[INFO] TS FF generation finished.')
    tsff.write()
    return tsff

def combine_ff_atoms(df1: pd.DataFrame, df2: pd.DataFrame, term_type: str = "bond") -> pd.Series:
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


def mix_parameters(tsff_df: pd.DataFrame, ff1_df: pd.DataFrame, ff2_df: pd.DataFrame):
    """
    Fill tsff_df['parameter'] with the average of corresponding parameters
    found in ff1_df and ff2_df where 'atoms' entries match.
    """
    new_params = []

    for idx, row in tsff_df.iterrows():
        c1_series = ff1_df.loc[ff1_df['atoms'].apply(lambda x: x == row['atoms']), 'parameter']
        c2_series = ff2_df.loc[ff2_df['atoms'].apply(lambda x: x == row['atoms']), 'parameter']

        c1 = c1_series.squeeze() if not c1_series.empty else c2_series.squeeze() # TODO ggf hier das es dann der wert einzeln ist anstatt mit 0 geaveraged
        c2 = c2_series.squeeze() if not c2_series.empty else c1_series.squeeze()

        new_val = _average_c(c1, c2)
        new_params.append((idx, new_val))

    for idx, val in new_params:
        tsff_df.at[idx, 'parameter'] = val



def _average_c(c1: float, c2: float, c1_factor: float = 0.5, c2_factor: float = 0.5) -> float:
    if round(c1_factor + c2_factor, 2) != 1.00:
        raise ValueError(f'c1 factor {c1_factor} + c2 factor {c2_factor} needs to be 1')
    return c1 * c1_factor + c2 * c2_factor


def mix_reference_values(tsff: ForceField, ff1: ForceField, ff2: ForceField, info1: StructuralInformation, info2: StructuralInformation):

    _mix_reference(tsff.bonds, ff1.bonds, ff2.bonds, info1, info2, 'bonds')
    _mix_reference(tsff.angles, ff1.angles, ff2.angles, info1, info2, 'angles')
    _mix_reference(tsff.dihedrals, ff1.dihedrals, ff2.dihedrals, info1, info2, 'dihedrals')
    _mix_reference(tsff.repulsive, ff1.repulsive, ff2.repulsive, info1, info2, 'repulsive')

def _mix_reference(tsff_df: pd.DataFrame, ff1_df: pd.DataFrame, ff2_df: pd.DataFrame, info1: StructuralInformation, info2: StructuralInformation, calctype: str, fact1: float = 0.5, fact2: float = 0.5):
    new_params = []
    for idx, row in tsff_df.iterrows():
        val1_series = ff1_df.loc[ff1_df['atoms'].apply(lambda x: x == row['atoms']), 'reference_value']
        val2_series = ff2_df.loc[ff2_df['atoms'].apply(lambda x: x == row['atoms']), 'reference_value']

        # TODO when changed to python v higher match calctype:
        if calctype == 'bonds':
            a, b = row['atoms']
            val1 = val1_series.squeeze() if not val1_series.empty else info1.vander_matrix[a, b]
            val2 = val2_series.squeeze() if not val2_series.empty else info2.vander_matrix[a, b]
            new_val = _average_single_bond(val1, val2, fact1, fact2)
        elif calctype == 'angles':
            a, b, c = row['atoms']
            val1 = val1_series.squeeze() if not val1_series.empty else angle(info1.fortran_xyz, a, b, c)
            val2 = val2_series.squeeze() if not val2_series.empty else angle(info2.fortran_xyz, a, b, c)
            new_val = _average_single_angle(val1, val2, fact1, fact2)
        elif calctype == 'dihedrals':
            a, b, c, d = row['atoms']
            val1 = val1_series.squeeze() if not val1_series.empty else dihedral_angle(info1.fortran_xyz, a, b, c, d)
            val2 = val2_series.squeeze() if not val2_series.empty else dihedral_angle(info2.fortran_xyz, a, b, c, d)
            new_val = _average_single_dihedral(val1, val2, fact1, fact2)
        elif calctype == 'repulsive':
            a, b = row['atoms']
            val1 = val1_series.squeeze() if not val1_series.empty else val2_series.squeeze()
            val2 = val2_series.squeeze() if not val2_series.empty else val1_series.squeeze()
            new_val = _average_single_repulsive(val1, val2, fact1, fact2)
        else:
            raise Exception(f'Typ {calctype} not known.')

        new_params.append((idx, new_val))

    for idx, val in new_params:
        tsff_df.at[idx, 'reference_value'] = val


def _average_single_bond(val1, val2, fact1: float = 0.5, fact2: float = 0.5):
    return val1 * fact1 + val2 * fact2
def _average_single_angle(val1, val2, fact1: float = 0.5, fact2: float = 0.5):
    return val1 * fact1 + val2 * fact2

def _average_single_dihedral(val1, val2, fact1: float = 0.5, fact2: float = 0.5):
    pi = np.pi
    
    temp1 = abs(val1 - val2)
    temp2 = abs(val1 + 2.0 * pi - val2)
    temp3 = abs(val1 - (val2 + 2.0 * pi))

    if temp1 < temp2 and temp1 < temp3:
        cosphi0 = np.cos(val1) * fact1 + np.cos(val2) * fact2
        sinphi0 = np.sin(val1) * fact1 + np.sin(val2) * fact2
    elif temp2 < temp1 and temp2 < temp3:
        cosphi0 = np.cos(val1 + 2.0 * pi) * fact1 + np.cos(val2) * fact2
        sinphi0 = np.sin(val1 + 2.0 * pi) * fact1 + np.sin(val2) * fact2
    else:  # temp3 is smallest
        cosphi0 = np.cos(val1) * fact1 + np.cos(val2 + 2.0 * pi) * fact2
        sinphi0 = np.sin(val1) * fact1 + np.sin(val2 + 2.0 * pi) * fact2

    phi_avg = np.arctan2(sinphi0, cosphi0)

    if phi_avg > pi:
        phi_avg -= 2.0 * pi
    elif phi_avg <= -pi:
        phi_avg += 2.0 * pi

    return phi_avg

def _average_single_repulsive(val1, val2, fact1: float = 0.5, fact2: float = 0.5):
    return val1 * fact1 + val2 * fact2