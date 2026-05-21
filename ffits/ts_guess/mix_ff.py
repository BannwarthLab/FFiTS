import logging
import numpy as np
import pandas as pd
from ffits.utils.geometry import bondlength, angle, dihedral_angle
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import StructuralInformation
from ffits.datatype.calculation_data import CalculationData
from ffits.utils.wbo_analysis import compare_wbo_differences

logger = logging.getLogger(__name__)


def create_tsff(
    ff1: ForceField,
    info1: StructuralInformation,
    ff2: ForceField,
    info2: StructuralInformation,
    calcdata: CalculationData,
    fact1: float = 0.5,
    fact2: float = 0.5,
    weigh_bonds_with_hessian: bool = True,
) -> dict[ForceField, dict]:
    """Creates TS force field (TS FF) from two given structures.

    Args:
        ff1 (ForceField): Force field of the reactant structure.
        info1 (StructuralInformation): Structural information of the reactant structure.
        ff2 (ForceField): Force field of the product structure.
        info2 (StructuralInformation): Structural information of the product structure.
        calcdata (CalculationData): Calculation data for the TS FF creation.
        fact1 (float, optional): Mixing factor for the reactant structure. Defaults to 0.5.
        fact2 (float, optional): Mixing factor for the product structure. Defaults to 0.5.
        weigh_bonds_with_hessian (bool, optional): Whether to weigh bonds with the Hessian. Defaults to True.

    Returns:
        dict[ForceField, dict]: ForceField object representing the TS FF and a dictionary with the mixing factors for each term if weigh_bonds_with_hessian is True.
    """
    if weigh_bonds_with_hessian and fact1 != 0.5:
        logger.warning(
            "Bonds from FFs are not averaged with chosen factors, since weigh_bonds_with_hessian is set to True. Weighting will be determined through Hessian analysis. "
        )

    # TODO add parameter transfer for averaginc dg and add that in printout too
    tsff = ForceField(ff1.nat, calcdata.ts_path.ff_filename, readff=False)
    tsff.bonds = combine_ff_atoms(ff1.bonds, ff2.bonds)
    tsff.angles = combine_ff_atoms(ff1.angles, ff2.angles)
    tsff.dihedrals = combine_ff_atoms(ff1.dihedrals, ff2.dihedrals)
    tsff.repulsive = remove_bonds_from_repulsive(
        combine_ff_atoms(ff1.repulsive, ff2.repulsive),
        combine_ff_atoms(ff1.bonds, ff2.bonds),
    )

    if weigh_bonds_with_hessian:
        logger.info("Calculating mixing factors based on Hessian analysis.")
        sharpness = 0.8
        params_mix = hessian_weighting_mix_list(
            info1, info2, tsff.bonds, tsff.angles, tsff.dihedrals, sharpness=sharpness
        )
        # get average reac parameter for bond terms (so tupels with only two entries)
        if params_mix:
            average_bond_param_reac = np.mean(
                [factor for atoms, factor in params_mix.items() if len(atoms) == 2]
            )
        else:
            average_bond_param_reac = 0.5

        if average_bond_param_reac < 0.5:
            logger.info(
                f"The optimization with the TSFF will start from the product geometry, since the average mixing factor for bond terms is {average_bond_param_reac:.4f} (< 0.5)."
            )
            tsff.start_from_reactant = False
        elif average_bond_param_reac >= 0.5:
            logger.info(
                f"The optimization with the TSFF will start from the reactant geometry, since the average mixing factor for bond terms is {average_bond_param_reac:.4f} (>= 0.5)."
            )
        else:
            logger.info(
                f"The optimization with the TSFF will start from the reactant geometry per default, since the average mixing factor for bond terms is {average_bond_param_reac:.4f}."
            )

    mix_parameters(tsff.bonds, ff1.bonds, ff2.bonds, fact1, fact2, params_mix)
    mix_parameters(tsff.angles, ff1.angles, ff2.angles, fact1, fact2, params_mix)
    mix_parameters(
        tsff.dihedrals, ff1.dihedrals, ff2.dihedrals, fact1, fact2, params_mix
    )
    mix_parameters(tsff.repulsive, ff1.repulsive, ff2.repulsive, fact1, fact2)

    mix_reference_values(
        tsff, ff1, ff2, info1, info2, fact1, fact2, dict_param_mix=params_mix
    )

    # print out the mixing factors to file
    if logger.level <= logging.DEBUG:
        logger.debug(
            f"Mixing factors written to {calcdata.ts_path.ff_filename[:-4]}.mixing_factors.txt"
        )
        with open(f"{calcdata.ts_path.ff_filename[:-4]}.mixing_factors.txt", "w") as f:
            f.write("Bond/Angle/Dihedral\tMixing Factor (Reactant)\n")
            for term, factor in params_mix.items():
                f.write(f"{term}\t{factor:.4f}\n")

    logger.info(f"TS FF generation finished.")
    tsff.write()
    return {"tsff": tsff, "params_mix": params_mix}


def combine_ff_atoms(
    df1: pd.DataFrame, df2: pd.DataFrame, term_type: str = "bond"
) -> pd.Series:
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
        """This function COULD normalize atom ordering, but decided not do that."""
        atoms = tuple(atoms)
        if term_type == "bond":
            return atoms  # return tuple(sorted(atoms))  # symmetric
        elif term_type == "angle":
            return atoms  # directional
        elif term_type == "dihedral":
            # dihedral symmetry: (1,2,3,4) equivalent to reversed (4,3,2,1)
            # reversed_atoms = tuple(reversed(atoms))
            # return min(atoms, reversed_atoms)
            return atoms
        elif term_type == "repulsive":
            return atoms  # tuple(sorted(atoms))  # symmetric
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

    return combined.sort_values("atoms").reset_index(drop=True)


def remove_bonds_from_repulsive(
    df_source: pd.DataFrame, df_reference: pd.DataFrame
) -> pd.DataFrame:
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
    if "atoms" not in df_source.columns or "atoms" not in df_reference.columns:
        raise ValueError("Both DataFrames must have an 'atoms' column.")

    # Convert to sets for O(1) lookup time per row
    reference_atoms = set(df_reference["atoms"])

    # Keep only rows whose 'atoms' are not in reference_atoms
    mask = ~df_source["atoms"].isin(reference_atoms)
    return df_source[mask].copy().reset_index(drop=True)


def mix_parameters(
    tsff_df: pd.DataFrame,
    ff1_df: pd.DataFrame,
    ff2_df: pd.DataFrame,
    fact1: float,
    fact2: float,
    param_dict: dict = None,
):
    """
    Fill tsff_df['parameter'] with the average of corresponding parameters
    found in ff1_df and ff2_df where 'atoms' entries match.
    """
    new_params = []

    for idx, row in tsff_df.iterrows():
        c1_series = ff1_df.loc[
            ff1_df["atoms"].apply(lambda x: x == row["atoms"]), "parameter"
        ]
        c2_series = ff2_df.loc[
            ff2_df["atoms"].apply(lambda x: x == row["atoms"]), "parameter"
        ]

        c1 = c1_series.squeeze() if not c1_series.empty else c2_series.squeeze()
        c2 = c2_series.squeeze() if not c2_series.empty else c1_series.squeeze()

        if (param_dict and row["atoms"] in param_dict) or (
            param_dict and tuple(reversed(row["atoms"])) in param_dict
        ):
            fact1 = param_dict.get(
                row["atoms"], param_dict.get(tuple(reversed(row["atoms"])))
            )
            fact2 = 1 - fact1

        new_val = _average_c(c1, c2, fact1, fact2)
        new_params.append((idx, new_val))

    for idx, val in new_params:
        tsff_df.at[idx, "parameter"] = val


def hessian_mix_list(
    info1: StructuralInformation,
    info2: StructuralInformation,
    atoms: tuple,
    sharpness: float = 0.2,
    changing_bonds: list = None,
) -> float:
    """
    Calculate mixing factor for a set of atoms based on hessian analysis.
    Only calculates if at least one bond in the atom group is in the changing_bonds list.

    Parameters
    ----------
    info1 : StructuralInformation
        Structural information for the reactant.
    info2 : StructuralInformation
        Structural information for the product.
    atoms : tuple
        Tuple of 2, 3, or 4 atom indices.
    sharpness : float
        Controls how sharply the mixing factor changes with the Hessian ratio (default: 0.2).
    changing_bonds : list
        List of bond tuples from WBO analysis that are changing. Only include this term if
        at least one of its bonds is in this list.

    Returns
    -------
    float or None
        Mixing factor for the reactant (between 0 and 1), or None if no changing bonds are present in this term.
    """
    if changing_bonds is None:
        changing_bonds = []

    # Generate all possible bonds for this set of atoms
    atom_pairs = []
    if len(atoms) == 2:
        atom_pairs = [atoms]

    elif len(atoms) == 3:
        i, j, k = atoms
        atom_pairs = [
            (min(i, j), max(i, j)),
            (min(i, k), max(i, k)),
            (min(j, k), max(j, k)),
        ]
    elif len(atoms) == 4:
        i, j, k, l = atoms
        atom_pairs = [
            (min(i, j), max(i, j)),
            (min(i, k), max(i, k)),
            (min(i, l), max(i, l)),
            (min(j, k), max(j, k)),
            (min(j, l), max(j, l)),
            (min(k, l), max(k, l)),
        ]
    else:
        raise ValueError(f"atoms must have 2, 3, or 4 elements, got {len(atoms)}")

    # Check if any bonds are in changing_bonds
    has_changing_bond = any(bond in changing_bonds for bond in atom_pairs)
    if not has_changing_bond:
        return None

    # Sum Hessian blocks
    h1_avg_sum = 0.0
    h2_avg_sum = 0.0

    for a1, a2 in atom_pairs:
        h1_block = info1.hessian[3 * a1 : 3 * a1 + 3, 3 * a2 : 3 * a2 + 3]
        h2_block = info2.hessian[3 * a1 : 3 * a1 + 3, 3 * a2 : 3 * a2 + 3]
        h1_avg_sum += np.mean(np.abs(h1_block))
        h2_avg_sum += np.mean(np.abs(h2_block))

    ratio = h1_avg_sum / h2_avg_sum if h2_avg_sum != 0 else 1
    param_reac = (ratio ** (1 - sharpness)) / (
        (ratio ** (1 - sharpness)) + 1
    )  # 1 - sharpness damit das Wort sharpness Sinn ergibt

    logger.debug(
        f"Atoms: {atoms}, H1 avg sum: {h1_avg_sum:.4f}, H2 avg sum: {h2_avg_sum:.4f}, Ratio: {ratio:.4f}, Param reac: {param_reac:.4f}"
    )

    return param_reac


def hessian_weighting_mix_list(
    info1: StructuralInformation,
    info2: StructuralInformation,
    bonds_df: pd.DataFrame,
    angles_df: pd.DataFrame,
    dihedrals_df: pd.DataFrame,
    sharpness: float = 0.2,
    threshold: float = 0.1,
) -> dict:
    """
    Create a dictionary with bond, angle, and dihedral tuples as keys and mixing factors for the reactant as values.
    Combines results from bonds, angles, and dihedrals, only including terms that contain at least one bond from the changing bonds.

    Args
        info1 (StructuralInformation): Structural information for the reactant.
        info2 (StructuralInformation): Structural information for the product.
        bonds_df (pd.DataFrame): DataFrame with bonds already combined from both force fields.
        angles_df (pd.DataFrame): DataFrame with angles already combined from both force fields.
        dihedrals_df (pd.DataFrame): DataFrame with dihedrals already combined from both force fields.
        sharpness (float): Controls how sharply the mixing factor changes with the Hessian ratio (default: 0.2).
        threshold (float): Minimum WBO change to consider a bond for mixing (default: 0.1).

    Returns
        (dict): Dictionary with bond tuples (i, j), angle tuples (i, j, k), and dihedral tuples (i, j, k, l) as keys and mixing factors for the reactant as values.
    """
    param_dict = {}
    wbo_diff = compare_wbo_differences(info1, info2, threshold=threshold)
    changing_bonds = [bond for bond, _, _, _ in wbo_diff["changing_bonds"]]

    if not changing_bonds:
        logger.info(
            f"No changing bonds found with WBO difference above {threshold}. All mixing factors will be 0.5."
        )
        return param_dict  # Return empty dict, which will lead to default 0.5 mixing in mix_parameters

    # Process bonds
    for idx, row in bonds_df.iterrows():
        atoms = row["atoms"]
        factor = hessian_mix_list(info1, info2, atoms, sharpness, changing_bonds)
        if factor is not None:
            param_dict[atoms] = factor

    # # Process angles
    # for idx, row in angles_df.iterrows():
    #     atoms = row['atoms']
    #     factor = hessian_mix_list(info1, info2, atoms, 0.8, changing_bonds)
    #     if factor is not None:
    #         param_dict[atoms] = factor

    # # Process dihedrals
    # for idx, row in dihedrals_df.iterrows():
    #     atoms = row['atoms']
    #     factor = hessian_mix_list(info1, info2, atoms, 0.8, changing_bonds)
    #     if factor is not None:
    #         param_dict[atoms] = factor

    return param_dict


def _average_c(c1: float, c2: float, c1_factor: float, c2_factor: float) -> float:
    if round(c1_factor + c2_factor, 2) != 1.00:
        raise ValueError(f"c1 factor {c1_factor} + c2 factor {c2_factor} needs to be 1")
    return round(c1 * c1_factor + c2 * c2_factor, 8)


def mix_reference_values(
    tsff: ForceField,
    ff1: ForceField,
    ff2: ForceField,
    info1: StructuralInformation,
    info2: StructuralInformation,
    fact1: float,
    fact2: float,
    dict_param_mix: dict = None,
):

    _mix_reference(
        tsff.bonds,
        ff1.bonds,
        ff2.bonds,
        info1,
        info2,
        "bonds",
        fact1,
        fact2,
        dict_param_mix,
    )
    _mix_reference(
        tsff.angles,
        ff1.angles,
        ff2.angles,
        info1,
        info2,
        "angles",
        fact1,
        fact2,
        dict_param_mix,
    )
    _mix_reference(
        tsff.dihedrals,
        ff1.dihedrals,
        ff2.dihedrals,
        info1,
        info2,
        "dihedrals",
        fact1,
        fact2,
        dict_param_mix,
    )
    _mix_reference(
        tsff.repulsive,
        ff1.repulsive,
        ff2.repulsive,
        info1,
        info2,
        "repulsive",
        fact1,
        fact2,
    )


def _mix_reference(
    tsff_df: pd.DataFrame,
    ff1_df: pd.DataFrame,
    ff2_df: pd.DataFrame,
    info1: StructuralInformation,
    info2: StructuralInformation,
    calctype: str,
    fact1: float,
    fact2: float,
    dict_param: dict = None,
):
    new_params = []
    for idx, row in tsff_df.iterrows():
        val1_series = ff1_df.loc[
            ff1_df["atoms"].apply(lambda x: x == row["atoms"]), "reference_value"
        ]
        val2_series = ff2_df.loc[
            ff2_df["atoms"].apply(lambda x: x == row["atoms"]), "reference_value"
        ]

        # TODO when changed to python v higher match calctype:
        if calctype == "bonds":
            a, b = row["atoms"]
            val1 = (
                val1_series.squeeze()
                if not val1_series.empty
                else info1.vander_matrix[a, b]
            )
            val2 = (
                val2_series.squeeze()
                if not val2_series.empty
                else info2.vander_matrix[a, b]
            )
            if val1 > info1.vander_matrix[a, b]:
                val1 = info1.vander_matrix[a, b]
            if val2 > info2.vander_matrix[a, b]:
                val2 = info2.vander_matrix[a, b]
            if dict_param and row["atoms"] in dict_param:
                fact1 = dict_param[row["atoms"]]
                fact2 = 1 - fact1
            new_val = _average_single_bond(val1, val2, fact1, fact2)
        elif calctype == "angles":
            a, b, c = row["atoms"]
            val1 = (
                val1_series.squeeze()
                if not val1_series.empty
                else angle(info1.fortran_xyz, a, b, c)
            )
            val2 = (
                val2_series.squeeze()
                if not val2_series.empty
                else angle(info2.fortran_xyz, a, b, c)
            )
            if dict_param and row["atoms"] in dict_param:
                fact1 = dict_param[row["atoms"]]
                fact2 = 1 - fact1
            new_val = _average_single_angle(val1, val2, fact1, fact2)
        elif calctype == "dihedrals":
            a, b, c, d = row["atoms"]
            val1 = (
                val1_series.squeeze()
                if not val1_series.empty
                else dihedral_angle(info1.fortran_xyz, a, b, c, d)
            )
            val2 = (
                val2_series.squeeze()
                if not val2_series.empty
                else dihedral_angle(info2.fortran_xyz, a, b, c, d)
            )
            if dict_param and row["atoms"] in dict_param:
                fact1 = dict_param[row["atoms"]]
                fact2 = 1 - fact1
            new_val = _average_single_dihedral(val1, val2, fact1, fact2)
        elif calctype == "repulsive":
            a, b = row["atoms"]
            val1 = (
                val1_series.squeeze()
                if not val1_series.empty
                else val2_series.squeeze()
            )
            val2 = (
                val2_series.squeeze()
                if not val2_series.empty
                else val1_series.squeeze()
            )
            new_val = _average_single_repulsive(val1, val2, fact1, fact2)
            # TODO eigentlich ist diese operation unnötig, könnte auch einfach vander distances suchen
        else:
            raise Exception(f"Typ {calctype} not known.")

        new_params.append((idx, new_val))

    for idx, val in new_params:
        tsff_df.at[idx, "reference_value"] = val


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
