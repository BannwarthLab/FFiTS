import numpy as np
import pandas as pd
from ffits.utils.geometry_calc import bondlength, angle, dihedral_angle
from ffits.datatype.structure_data import ForceField, StructuralInformation
from ffits.datatype.calculation_data import CalculationData
from ffits.utils.wbo_analysis import compare_wbo_differences

def create_tsff(ff1: ForceField, info1: StructuralInformation, ff2: ForceField, info2: StructuralInformation, fact1: float, fact2: float, calcdata: CalculationData, weigh_bonds_with_hessian: bool = True) -> ForceField:

    # TODO add parameter transfer for averaging and add that in printout too
    tsff = ForceField(ff1.nat, calcdata.ts_path.ff_filename, readff=False)
    tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
    tsff.repulsive = remove_bonds_from_repulsive(combine_ff_atoms(ff1.repulsive, ff2.repulsive), combine_ff_atoms(ff1.bonds, ff2.bonds))

    if weigh_bonds_with_hessian:
        param_dict = bond_mix_list(info1, info2, sharpness=0.2)

    mix_parameters(tsff.bonds, ff1.bonds, ff2.bonds, fact1, fact2, param_dict)
    mix_parameters(tsff.angles, ff1.angles, ff2.angles, fact1, fact2)
    mix_parameters(tsff.dihedrals, ff1.dihedrals, ff2.dihedrals, fact1, fact2)
    mix_parameters(tsff.repulsive, ff1.repulsive, ff2.repulsive, fact1, fact2)
    
    mix_reference_values(tsff, ff1, ff2, info1, info2, fact1, fact2, param_dict)

    # print out the mixing factors to file
    with open(calcdata.ts_path.ff_filename + ".mixing_factors.txt", "w") as f:
        f.write("Bond\tMixing Factor (Reactant)\n")
        for bond, factor in param_dict.items():
            f.write(f"{bond}\t{factor:.4f}\n")

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


def mix_parameters(tsff_df: pd.DataFrame, ff1_df: pd.DataFrame, ff2_df: pd.DataFrame, fact1: float, fact2: float, param_dict: dict = None):
    """
    Fill tsff_df['parameter'] with the average of corresponding parameters
    found in ff1_df and ff2_df where 'atoms' entries match.
    """
    new_params = []

    for idx, row in tsff_df.iterrows():
        c1_series = ff1_df.loc[ff1_df['atoms'].apply(lambda x: x == row['atoms']), 'parameter']
        c2_series = ff2_df.loc[ff2_df['atoms'].apply(lambda x: x == row['atoms']), 'parameter']

        c1 = c1_series.squeeze() if not c1_series.empty else c2_series.squeeze() 
        c2 = c2_series.squeeze() if not c2_series.empty else c1_series.squeeze()

        if (param_dict and row['atoms'] in param_dict) or (param_dict and tuple(reversed(row['atoms'])) in param_dict):
            fact1 = param_dict.get(row['atoms'], param_dict.get(tuple(reversed(row['atoms']))))
            fact2 = 1 - fact1

        new_val = _average_c(c1, c2, fact1, fact2)
        new_params.append((idx, new_val))

    for idx, val in new_params:
        tsff_df.at[idx, 'parameter'] = val


def bond_mix_list(info1: StructuralInformation, info2: StructuralInformation, sharpness: float = 0.2, threshold: float = 0.1):
    """
    Create a dictionary with bond tuples as keys and mixing factors for the reactant as values.
    The mixing factor is calculated based on the ratio of average Hessian values for the bond in the reactant and product. So if the Hessian block for the bond in the reactant is much larger than in the product, the factor will be closer to 1 (more weight on reactant parameters). If the Hessian block is much smaller, the factor will be closer to 0 (more weight on product parameters). 

    Parameters
    ----------
    info1 : StructuralInformation
        Structural information for the reactant.
    info2 : StructuralInformation
        Structural information for the product.
    sharpness : float
        Controls how sharply the mixing factor changes with the Hessian ratio (default: 0.2). The lower the sharpness, the more the factor will approach 0 or 1 for small deviations in the ratio. The higher the sharpness, the more the factor will be closer to 0.5 for a wider range of ratios.
    threshold : float
        Minimum WBO change to consider a bond for mixing (default: 0.1).
    Returns
    -------
    dict
        Dictionary with bond tuples (i, j) as keys and mixing factors for the reactant as values (between 0 and 1).
    """
    param_dict = {}

    wbo_diff = compare_wbo_differences(info1, info2, threshold=threshold)
    
    for bond, _, _, _ in wbo_diff['changing_bonds']:
        i, j = bond
        
        i_start, i_end = 3 * i, 3 * i + 3
        j_start, j_end = 3 * j, 3 * j + 3
        
        h1_block = info1.hessian[i_start:i_end, j_start:j_end]
        h2_block = info2.hessian[i_start:i_end, j_start:j_end]
        
        h1_avg = np.mean(np.abs(h1_block))
        h2_avg = np.mean(np.abs(h2_block))

        ratio = h1_avg / h2_avg if h2_avg != 0 else 0.0001

        param_reac = (ratio**sharpness) / ((ratio**sharpness) + 1)

        # create a dict with bond as key and param_reac and param_prod as values
        param_dict[(i, j)] = param_reac
    print(param_dict)
    return param_dict


def _average_c(c1: float, c2: float, c1_factor: float, c2_factor: float) -> float:
    if round(c1_factor + c2_factor, 2) != 1.00:
        raise ValueError(f'c1 factor {c1_factor} + c2 factor {c2_factor} needs to be 1')
    return round(c1 * c1_factor + c2 * c2_factor, 8)


def mix_reference_values(tsff: ForceField, ff1: ForceField, ff2: ForceField, info1: StructuralInformation, info2: StructuralInformation, fact1: float, fact2: float, dict_param: dict = None):

    _mix_reference(tsff.bonds, ff1.bonds, ff2.bonds, info1, info2, 'bonds', fact1, fact2, dict_param)
    _mix_reference(tsff.angles, ff1.angles, ff2.angles, info1, info2, 'angles', fact1, fact2, dict_param)
    _mix_reference(tsff.dihedrals, ff1.dihedrals, ff2.dihedrals, info1, info2, 'dihedrals', fact1, fact2, dict_param)
    _mix_reference(tsff.repulsive, ff1.repulsive, ff2.repulsive, info1, info2, 'repulsive', fact1, fact2, dict_param)

def _mix_reference(tsff_df: pd.DataFrame, ff1_df: pd.DataFrame, ff2_df: pd.DataFrame, info1: StructuralInformation, info2: StructuralInformation, calctype: str, fact1: float, fact2: float, dict_param: dict = None):
    new_params = []
    for idx, row in tsff_df.iterrows():
        val1_series = ff1_df.loc[ff1_df['atoms'].apply(lambda x: x == row['atoms']), 'reference_value']
        val2_series = ff2_df.loc[ff2_df['atoms'].apply(lambda x: x == row['atoms']), 'reference_value']

        # TODO when changed to python v higher match calctype:
        if calctype == 'bonds':
            a, b = row['atoms']
            val1 = val1_series.squeeze() if not val1_series.empty else info1.vander_matrix[a, b]
            val2 = val2_series.squeeze() if not val2_series.empty else info2.vander_matrix[a, b]
            if val1 > info1.vander_matrix[a, b]: 
                val1 = info1.vander_matrix[a, b]
            if val2 > info2.vander_matrix[a, b]: 
                val2 = info2.vander_matrix[a, b]
            if dict_param and row['atoms'] in dict_param:
                fact1 = dict_param[row['atoms']]
                fact2 = 1 - fact1
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
            # TODO eigentlich ist diese operation unnötig, könnte auch einfach vander distances suchen
        else:
            raise Exception(f'Typ {calctype} not known.')

        new_params.append((idx, new_val))

    for idx, val in new_params:
        tsff_df.at[idx, 'reference_value'] = val


def _average_single_bond(val1, val2, fact1: float, fact2: float):
    return round(val1 * fact1 + val2 * fact2, 8)
def _average_single_angle(val1, val2, fact1: float, fact2: float):
    return round(val1 * fact1 + val2 * fact2, 8)

def _average_single_dihedral(val1, val2, fact1: float, fact2: float):
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

    # phi_avg = 0.5* (val1 + val2 + 2.0 * pi)  # Alternative: simple average

    if phi_avg > pi:
        phi_avg -= 2.0 * pi
    elif phi_avg <= -pi:
        phi_avg += 2.0 * pi

    return round(phi_avg, 8)

def _average_single_repulsive(val1, val2, fact1: float, fact2: float):
    return round(val1 * fact1 + val2 * fact2, 8)