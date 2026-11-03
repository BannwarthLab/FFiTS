"""Mixes reactant and product force fields into a TS force field."""

import logging
import numpy as np
import pandas as pd
from ffits.data.vanderwaals_radii import VANDER_VALUES, ATOMIC_RADII_TRUHLAR
from ffits.utils.geometry import angle, angstrom2bohr, dihedral_angle
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import StructuralInformation
from ffits.datatype.calculation_data import CalculationData

logger = logging.getLogger(__name__)

TERMS = ("bonds", "angles", "dihedrals", "repulsive")


def create_tsff(
    ff1: ForceField,
    info1: StructuralInformation,
    ff2: ForceField,
    info2: StructuralInformation,
    calcdata: CalculationData | None = None,
) -> ForceField:
    """Creates TS force field (TS FF) from two given structures.

    Mixing factors come from ``calcdata.ts_calc`` (factor_reactant/
    factor_product). With ``ts_calc.hydrogen_covalent_radius``, forming or
    breaking covalent H-X bonds use the covalent radius of hydrogen on the side
    where they are missing (see :func:`_mix_bond_references`). With
    ``ts_calc.scale_one_sided_bonds``, the parameters of bonds present in only
    one FF are scaled down when that FF has a weight below 0.5 (see
    :func:`mix_parameters`).

    Args:
        ff1 (ForceField): Force field of the reactant structure.
        info1 (StructuralInformation): Structural information of the reactant structure.
        ff2 (ForceField): Force field of the product structure.
        info2 (StructuralInformation): Structural information of the product structure.
        calcdata (CalculationData, optional): Calculation data for the TS
            FF creation. Defaults to a fresh ``CalculationData.from_default()``.

    Returns:
        ForceField: The TS force field.
    """
    if calcdata is None:
        calcdata = CalculationData.from_default()
    ts_calc = calcdata.ts_calc
    fact1 = ts_calc.factor_reactant
    fact2 = ts_calc.factor_product

    tsff = combine_ff_topology(ff1, ff2, calcdata.ts_path.ff_filename)

    for term in TERMS:
        mix_parameters(
            getattr(tsff, term),
            getattr(ff1, term),
            getattr(ff2, term),
            fact1,
            fact2,
            scale_one_sided=ts_calc.scale_one_sided_bonds and term == "bonds",
        )

    mix_reference_values(
        tsff,
        ff1,
        ff2,
        info1,
        info2,
        fact1,
        fact2,
        hydrogen_covalent_radius=ts_calc.hydrogen_covalent_radius,
    )

    logger.info("TS FF generation finished.")
    tsff.write()
    return tsff


def combine_ff_topology(ff1: ForceField, ff2: ForceField, filename: str) -> ForceField:
    """Force field holding every term of ff1 and ff2 once, before any mixing.

    Atom pairs that are bonded in either force field are dropped from the
    repulsive terms.

    Args:
        ff1 (ForceField): Force field of the reactant structure.
        ff2 (ForceField): Force field of the product structure.
        filename (str): File the combined force field is written to.

    Returns:
        ForceField: Combined force field (parameters and reference values
        still those of ff1, or ff2 for terms only present there).
    """
    tsff = ForceField(ff1.nat, filename, readff=False)
    for term in TERMS:
        setattr(tsff, term, combine_ff_atoms(getattr(ff1, term), getattr(ff2, term)))
    tsff.repulsive = remove_bonds_from_repulsive(tsff.repulsive, tsff.bonds)
    return tsff


def combine_ff_atoms(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Combine two force field term DataFrames, keeping each atom tuple once.

    Atom tuples are compared as given (no reordering), so (i, j) and (j, i)
    are different terms. For duplicates the row of ``df1`` is kept.

    Args:
        df1 (pd.DataFrame): Force field term DataFrame with an 'atoms' column.
        df2 (pd.DataFrame): Force field term DataFrame with an 'atoms' column.

    Returns:
        pd.DataFrame: Combined DataFrame sorted by atom tuple.
    """
    combined = pd.concat([df1, df2], ignore_index=True)
    combined["atoms"] = combined["atoms"].map(tuple)
    combined = combined.drop_duplicates(subset=["atoms"], keep="first")
    return combined.sort_values("atoms").reset_index(drop=True)


def remove_bonds_from_repulsive(
    df_source: pd.DataFrame, df_reference: pd.DataFrame
) -> pd.DataFrame:
    """Remove rows from df_source whose 'atoms' tuple is present in df_reference['atoms'].

    Args:
        df_source (pd.DataFrame): DataFrame from which rows will be removed.
        df_reference (pd.DataFrame): DataFrame providing the 'atoms' tuples to remove.

    Returns:
        pd.DataFrame: df_source with matching rows removed.
    """
    if "atoms" not in df_source.columns or "atoms" not in df_reference.columns:
        raise ValueError("Both DataFrames must have an 'atoms' column.")
    reference_atoms = set(df_reference["atoms"])
    mask = ~df_source["atoms"].isin(reference_atoms)
    return df_source[mask].reset_index(drop=True)


def mix_parameters(
    tsff_df: pd.DataFrame,
    ff1_df: pd.DataFrame,
    ff2_df: pd.DataFrame,
    fact1: float,
    fact2: float,
    scale_one_sided: bool = False,
):
    """Fill tsff_df['parameter'] with the fact1/fact2-weighted average of ff1_df/ff2_df, in place.

    Terms missing in one FF take the value of the other FF. With
    ``scale_one_sided``, such a term is additionally multiplied by
    min(1, 2 * w), w being the weight of the FF it is present in: it fades out
    towards the other structure and is unchanged for w >= 0.5.
    """
    _check_factors(fact1, fact2)
    if tsff_df.empty:
        return
    atoms = tsff_df["atoms"].tolist()
    c1 = _lookup(atoms, ff1_df, "parameter")
    c2 = _lookup(atoms, ff2_df, "parameter")
    only2, only1 = np.isnan(c1), np.isnan(c2)
    c1, c2 = np.where(only2, c2, c1), np.where(only1, c1, c2)
    mixed = c1 * fact1 + c2 * fact2
    if scale_one_sided:
        mixed = np.where(only1, mixed * min(1.0, 2 * fact1), mixed)
        mixed = np.where(only2, mixed * min(1.0, 2 * fact2), mixed)
    tsff_df["parameter"] = np.round(mixed, 8)


def mix_reference_values(
    tsff: ForceField,
    ff1: ForceField,
    ff2: ForceField,
    info1: StructuralInformation,
    info2: StructuralInformation,
    fact1: float,
    fact2: float,
    hydrogen_covalent_radius: bool = False,
):
    """Mix reference values (bond lengths, angles, dihedrals, repulsive distances) for every term of ``tsff``, in place.

    ``hydrogen_covalent_radius`` sets the missing-side length
    of forming/breaking covalent H-X bonds (see :func:`_mix_bond_references`).
    """
    for term in TERMS:
        mixer_kwargs = (
            {"hydrogen_covalent_radius": hydrogen_covalent_radius}
            if term == "bonds"
            else {}
        )
        _mix_reference(
            getattr(tsff, term),
            getattr(ff1, term),
            getattr(ff2, term),
            info1,
            info2,
            term,
            fact1,
            fact2,
            **mixer_kwargs,
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
    **mixer_kwargs,
):
    """Fill tsff_df['reference_value'] for one term type ('bonds'/'angles'/'dihedrals'/'repulsive').

    Missing reference values on either side fall back to the van der Waals
    distance (bonds), the geometric angle/dihedral computed directly from the
    corresponding structure (angles/dihedrals) or the value of the other FF
    (repulsive). ``mixer_kwargs`` are passed on to the mixer of ``calctype``.
    """
    _check_factors(fact1, fact2)
    if tsff_df.empty:
        return
    if calctype not in _REFERENCE_MIXERS:
        raise ValueError(f"Term type {calctype} not known.")
    atoms = tsff_df["atoms"].tolist()
    val1 = _lookup(atoms, ff1_df, "reference_value")
    val2 = _lookup(atoms, ff2_df, "reference_value")
    new_val = _REFERENCE_MIXERS[calctype](
        val1, val2, atoms, info1, info2, fact1, fact2, **mixer_kwargs
    )
    tsff_df["reference_value"] = np.round(new_val, 8)


def _mix_bond_references(
    val1,
    val2,
    atoms,
    info1,
    info2,
    f1,
    f2,
    hydrogen_covalent_radius: bool = False,
) -> np.ndarray:
    """Bond lengths, each capped at the van der Waals distance of its structure.

    A bond missing on one side (forming or breaking) takes the van der Waals
    distance there. With ``hydrogen_covalent_radius``, a forming/breaking bond
    between hydrogen and a heavier element X that is covalent on the side where
    it exists (an edge of ``complete_graph``) uses the covalent instead of the
    van der Waals radius of hydrogen on the missing side, but never less than
    the bonded length. Hydrogen bonds H...A added to the bonds keep the van der
    Waals distance.
    """
    idx = np.array(atoms, dtype=int)
    i, j = idx[:, 0], idx[:, 1]
    vander1 = info1.vander_matrix[i, j]
    vander2 = info2.vander_matrix[i, j]
    breaking, forming = np.isnan(val2), np.isnan(val1)
    val1 = np.fmin(val1, vander1)  # fmin also replaces missing (NaN) values
    val2 = np.fmin(val2, vander2)
    if hydrogen_covalent_radius:
        is_h = np.char.capitalize(np.asarray(info1.atom_types, dtype=str))[idx] == "H"
        h_heavy = is_h[:, 0] != is_h[:, 1]  # exactly one hydrogen
        bonded1 = np.array([info1.complete_graph.has_edge(a, b) for a, b in zip(i, j)])
        bonded2 = np.array([info2.complete_graph.has_edge(a, b) for a, b in zip(i, j)])
        breaking &= h_heavy & bonded1
        forming &= h_heavy & bonded2
        shift = angstrom2bohr(
            VANDER_VALUES[0] - ATOMIC_RADII_TRUHLAR[0]
        )  # H: vdW - cov
        val2 = np.where(breaking, np.fmax(vander2 - shift, val1), val2)
        val1 = np.where(forming, np.fmax(vander1 - shift, val2), val1)
    return val1 * f1 + val2 * f2


def _mix_angle_references(val1, val2, atoms, info1, info2, f1, f2) -> np.ndarray:
    """Angles, missing values computed from the geometry of their structure."""
    val1 = _fill_missing(val1, atoms, lambda a: angle(info1.fortran_xyz, *a))
    val2 = _fill_missing(val2, atoms, lambda a: angle(info2.fortran_xyz, *a))
    return val1 * f1 + val2 * f2


def _mix_dihedral_references(val1, val2, atoms, info1, info2, f1, f2) -> np.ndarray:
    """Dihedrals, missing values computed from the geometry of their structure."""
    val1 = _fill_missing(val1, atoms, lambda a: dihedral_angle(info1.fortran_xyz, *a))
    val2 = _fill_missing(val2, atoms, lambda a: dihedral_angle(info2.fortran_xyz, *a))
    return _average_dihedrals(val1, val2, f1, f2)


def _mix_repulsive_references(val1, val2, atoms, info1, info2, f1, f2) -> np.ndarray:
    """Repulsive distances, missing values taken from the other FF."""
    # this lookup is not really needed, the van der Waals distances could be used directly, but if it ever changes i keep it this general
    val1, val2 = (
        np.where(np.isnan(val1), val2, val1),
        np.where(np.isnan(val2), val1, val2),
    )
    return val1 * f1 + val2 * f2


_REFERENCE_MIXERS = {
    "bonds": _mix_bond_references,
    "angles": _mix_angle_references,
    "dihedrals": _mix_dihedral_references,
    "repulsive": _mix_repulsive_references,
}

#### HELPERS (most of them create arrays out of given data)


def _check_factors(fact1: float, fact2: float):
    """Raise if the two mixing factors do not sum to 1."""
    if round(fact1 + fact2, 2) != 1.00:
        raise ValueError(f"c1 factor {fact1} + c2 factor {fact2} needs to be 1")


def _lookup(atoms: list, ff_df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Creates a list of the values in ``column`` from ``ff_df``, sorted as in ``atoms`` by looking up the corresponding atom - column pair to make it easier to combine them.
    """
    values = dict(
        zip(ff_df["atoms"], ff_df[column])
    )  # creating a dict is done to avoid iterating through the whole list in a for loop every search step
    return np.array([values.get(a, np.nan) for a in atoms], dtype=float)


def _fill_missing(values: np.ndarray, atoms: list, compute) -> np.ndarray:
    """Replace NaN entries of ``values`` by ``compute(atoms[i])``."""
    values = values.copy()
    for i in np.flatnonzero(np.isnan(values)):
        values[i] = compute(atoms[i])
    return values


def _average_dihedrals(
    val1: np.ndarray, val2: np.ndarray, fact1: float, fact2: float
) -> np.ndarray:
    """Weighted circular average of two arrays of dihedral angles, result in (-pi, pi]."""
    cosphi0 = np.cos(val1) * fact1 + np.cos(val2) * fact2
    sinphi0 = np.sin(val1) * fact1 + np.sin(val2) * fact2
    phi_avg = np.arctan2(sinphi0, cosphi0)
    return np.where(phi_avg <= -np.pi, phi_avg + 2.0 * np.pi, phi_avg)
