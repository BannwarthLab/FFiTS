"""Derives initial force field terms (bonds, angles, dihedrals, repulsion, H-bonds) from structural data."""

import numpy as np
import pandas as pd
from ffits.utils.bond_threshold import get_bo_threshold_matrix
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import StructuralInformation
from ffits.utils.geometry import angle, bondlength, dihedral_angle
import logging

logger = logging.getLogger(__name__)


def canonical_dihedral(i, j, l, m):
    """
    Return a canonical ordering for a dihedral (i, j, l, m),
    so that (i,j,l,m) and (m,l,j,i) collapse to the same tuple.
    """
    forward = (i, j, l, m)
    reverse = (m, l, j, i)
    return min(forward, reverse)


def _normalize_improper(dihedral: tuple) -> tuple:
    """
    Normalize improper dihedral to canonical form: (central_atom, *sorted_terminals).
    This ensures all permutations of terminal atoms map to the same representation.

    Args:
        dihedral: tuple of (central_atom, terminal1, terminal2, terminal3)

    Returns:
        canonical form with terminals sorted
    """
    central = dihedral[0]
    terminals = tuple(sorted(dihedral[1:]))
    return (central, *terminals)


def _sort_candidates_canonical(atoms: set, priorities: dict) -> list:
    """
    Sort atoms by canonical ordering: highest priority first, then lowest index for ties.

    Args:
        atoms: set of atom indices
        priorities: dict mapping atom index to priority value

    Returns:
        list of atoms sorted by (-priority, index)
    """
    return sorted(
        atoms, key=lambda a: (-priorities.get(a, 0), a)
    )  # first then second criterion


def _create_improper_dihedrals(
    central_atom: int, candidates: list, result: list
) -> None:
    """
    Create improper dihedrals from candidates and append to result.
    Processes candidates in groups of 3 from the end until fewer than 3 remain.

    Args:
        central_atom: the central atom index for the improper dihedral
        candidates: sorted list of terminal atoms (will be modified)
        result: list to append created improper dihedrals to
    """
    central_atom = int(central_atom)
    while len(candidates) >= 3:
        # if len = 1, means that only the atom from the proper dihedral remained
        # if len = 2, there are not enough atoms to form an improper dihedral

        # these if statements are here for more complicated cases with many bonds like metal centers
        # generally: I start from the back, so the candidate atoms with lower priority are used first
        if len(candidates) >= 6 or len(candidates) == 3:
            result.append(((central_atom, *[int(c) for c in candidates[-3:]]), False))
        elif len(candidates) == 4:
            result.append(((central_atom, *[int(c) for c in candidates[-3:]]), False))
        else:
            result.append(((central_atom, *[int(c) for c in candidates[-3:]]), False))
            result.append(((central_atom, *[int(c) for c in candidates[:3]]), False))
        del candidates[-3:]


def _remove_duplicate_improper_dihedrals(dihedrals: list) -> list:
    """
    Remove duplicate improper dihedrals by normalizing to canonical form.
    All permutations of terminal atoms are treated as duplicates.

    Args:
        dihedrals: list of tuples (dihedral, is_proper) where dihedral is a tuple of atom indices and is_proper is a boolean

    Returns:
        list without duplicate improper dihedrals; proper dihedrals are not affected
    """
    seen = set()
    result = []

    for dihedral, is_proper in dihedrals:
        if is_proper:
            result.append((dihedral, is_proper))
        else:
            canonical_form = _normalize_improper(dihedral)
            if canonical_form not in seen:
                result.append((canonical_form, is_proper))
                seen.add(canonical_form)

    return result


def filter_dihedrals(dihedrals: list, priorities: dict, A: np.ndarray) -> list:
    """
    Filter dihedrals to keep one proper dihedral per central atom pair.
    Create improper dihedrals for remaining atoms around each central atom.
    Each improper dihedral only uses terminal atoms bonded to the central atom.

    Args:
        dihedrals: list of dihedral tuples (i, j, l, m) where j and l are central atoms
        priorities: dict mapping atom index to priority value
        A: adjacency matrix for connectivity checking

    Returns:
        list of tuples [(dihedral, is_proper), ...] where is_proper indicates whether
        the dihedral is a proper (True) or improper (False) dihedral
    """
    from collections import defaultdict

    # Group dihedrals by central atom pair
    central_pair_groups = defaultdict(list)
    for dihedral in dihedrals:
        i, j, l, m = dihedral
        central_pair = tuple(sorted([j, l]))
        central_pair_groups[central_pair].append(dihedral)
    result = []

    for central_pair, dihedral_group in central_pair_groups.items():
        # proper dihedral with highest combined terminal atom priority
        proper_dihedral = max(
            dihedral_group,
            key=lambda d: priorities.get(d[0], 0) + priorities.get(d[3], 0),
        )
        result.append((proper_dihedral, True))
        i_sel, j_sel, l_sel, m_sel = proper_dihedral

        # Collect all terminal atoms bonded to each central atom by picking it from the adjacency matrix A
        atoms_bonded_to_l = set(np.where(A[l_sel])[0])
        atoms_bonded_to_j = set(np.where(A[j_sel])[0])

        candidates_j = _sort_candidates_canonical(atoms_bonded_to_j, priorities)
        candidates_l = _sort_candidates_canonical(atoms_bonded_to_l, priorities)

        # remove the second central atom and put at the beginning of the list
        candidates_j.remove(l_sel)
        candidates_j.insert(0, l_sel)
        candidates_l.remove(j_sel)
        candidates_l.insert(0, j_sel)

        _create_improper_dihedrals(j_sel, candidates_j, result)
        _create_improper_dihedrals(l_sel, candidates_l, result)

    return _remove_duplicate_improper_dihedrals(result)


def _remove_duplicate_proper_dihedrals(dihedrals: list) -> list:
    """
    Remove duplicate dihedrals keeping improper (False) over proper (True) when duplicates exist.
    Also sorts the dihedrals by atom indices.

    Args:
        dihedrals: list of tuples (dihedral, is_proper) where dihedral is a tuple of atom indices

    Returns:
        list of tuples [(dihedral, is_proper), ...] with duplicates removed and sorted
    """
    sorted_pairs = sorted(dihedrals, key=lambda x: x[0])
    seen = {}
    deduplicated = []
    for atoms, is_proper_val in sorted_pairs:
        if atoms not in seen:
            seen[atoms] = is_proper_val
            deduplicated.append((atoms, is_proper_val))
        else:
            # Keep improper (False) over proper (True)
            if (
                not is_proper_val and seen[atoms]
            ):  # Current is improper, existing is proper
                # Replace the existing one
                deduplicated = [(a, p) for a, p in deduplicated if a != atoms]
                deduplicated.append((atoms, is_proper_val))
                seen[atoms] = is_proper_val

    return deduplicated


def _is_at_linear_angle(
    atoms: tuple, is_proper: bool, info: StructuralInformation, tolerance_deg: float
) -> bool:
    """True if a dihedral is built on a (near-)linear angle, where it is undefined.

    - Proper ``(i, j, k, l)``: linear if the angle i-j-k or j-k-l is straight.
    - Improper ``(central, t1, t2, t3)``: linear if any two substituents lie on
      opposite sides of the central atom.

    Args:
        atoms: the four 0-based atom indices of the dihedral.
        is_proper: whether ``atoms`` is a proper (True) or improper (False) dihedral.
        info: structural information providing the geometry.
        tolerance_deg: an angle within this many degrees of 0 or 180 counts as linear.
    """
    tolerance_rad = np.radians(tolerance_deg)

    def is_straight(a, vertex, b) -> bool:
        bond_angle = angle(info.fortran_xyz, a, vertex, b)
        distance_from_straight = min(bond_angle, np.pi - bond_angle)
        return distance_from_straight < tolerance_rad

    if is_proper:
        i, j, k, l = atoms
        return is_straight(i, j, k) or is_straight(j, k, l)
    else:
        central, t1, t2, t3 = atoms
        return (
            is_straight(t1, central, t2)
            or is_straight(t1, central, t3)
            or is_straight(t2, central, t3)
        )


def fill_ff(
    ff: ForceField,
    info: StructuralInformation,
    priorities: dict = None,
    repulsive_start: float = 0.01,
    bo_threshold: float = 0.0,
    linear_angle_tolerance_deg: float = 5.0,
) -> None:
    """Fills the ForceField object with bonds, angles, dihedrals, and repulsive terms based on the structural information. With priorities, one proper dihedral is kept per central bond and the rest become impropers. Dihedrals on a (near-)linear bond angle are dropped.

    Args:
        ff (ForceField): ForceField object to be filled with parameters.
        info (StructuralInformation): Structural information for the molecule.
        priorities (dict, optional): Dictionary mapping atom indices to their priorities. Defaults to None.
        repulsive_start (float, optional): Initial value for the FF parameter of the repulsive terms. Defaults to 0.01.
        bo_threshold (float, optional): Threshold for defining bonds based on bond order, used for all atom pairs with at least one atom from the third period or higher. Pairs of two atoms from the first or second period always use a threshold of 0.3. Defaults to 0.0.
        linear_angle_tolerance_deg (float, optional): A dihedral is dropped if the bond angle at either of its inner two atoms is within this many degrees of 0 or 180. Defaults to 5.0.

    Raises:
        ZeroDivisionError: When the bond order or bond length is zero for a bond or angle, which would lead to division by zero in parameter calculations.

    Returns:
        None: ForceField object is modified in place
    """
    n = ff.nat
    wbo = info.bo_matrix

    # --- Step 1: adjacency matrix ---
    thresholds = get_bo_threshold_matrix(info.atom_types, bo_threshold)
    A = (wbo > thresholds).astype(int)
    np.fill_diagonal(A, 0)

    # ============================================================
    # Subfunctions
    # ============================================================

    # ---- Atom generation ----
    def get_bond_atoms(A):
        """List all bonded atom pairs from the adjacency matrix."""
        i, j = np.where(np.triu(A, 1))
        bonds = np.stack([i, j], axis=1)
        bonds = bonds[np.lexsort((bonds[:, 1], bonds[:, 0]))]
        return [tuple(int(x) for x in pair) for pair in bonds]

    def get_angle_atoms(A):
        """List all (i, j, k) angle triplets from the adjacency matrix."""
        angles = []
        for j in range(n):
            neighbors = np.where(A[j])[0]
            for i in neighbors:
                for k in neighbors:
                    if i < k:
                        angles.append((int(i), int(j), int(k)))
        if not angles:
            return []
        angles = np.array(angles)
        angles = angles[np.lexsort((angles[:, 2], angles[:, 1], angles[:, 0]))]
        return [tuple(int(x) for x in triplet) for triplet in angles]

    def get_dihedral_atoms(A):
        """List all canonical (i, j, l, m) dihedral quadruplets from the adjacency matrix."""
        dihedrals = []
        for j in range(n):
            for l in np.where(A[j])[0]:
                for i in np.where(A[j])[0]:
                    if i == l:
                        continue
                    for m in np.where(A[l])[0]:
                        if m in (i, j, l):
                            continue
                        dihedrals.append(canonical_dihedral(i, j, l, m))
        dihedrals = sorted(set(dihedrals))
        return [tuple(int(x) for x in d) for d in dihedrals]

    def get_repulsive_atoms(A):
        """List all non-bonded atom pairs from the adjacency matrix."""
        all_i, all_j = np.triu_indices(n, 1)
        mask = A[all_i, all_j] == 0
        pairs = np.stack([all_i[mask], all_j[mask]], axis=1)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        return [tuple(int(x) for x in pair) for pair in pairs]

    # ---- Reference calculations ----
    def ref_bond(atoms):
        """Reference bond length for a bonded atom pair."""
        i, j = atoms
        return round(bondlength(info.fortran_xyz, i, j), 8)

    def ref_angle(atoms):
        """Reference angle for an atom triplet."""
        i, j, k = atoms
        return round(angle(info.fortran_xyz, i, j, k), 8)

    def ref_dihedral(atoms):
        """Reference dihedral angle for an atom quadruplet."""
        i, j, k, l = atoms
        return round(dihedral_angle(info.fortran_xyz, i, j, k, l), 8)

    def ref_repulsive(atoms):
        """Reference van der Waals distance for a non-bonded atom pair."""
        i, j = atoms
        return round(
            info.vander_matrix[i, j], 8
        )  # / (2 ** (1 / 6)) For some reason we dont need this and i dont know why

    # ---- Parameter calculations ----
    def param_bond(atoms):
        """Bond force constant, derived from bond order over bond length."""
        i, j = atoms
        bl = bondlength(info.fortran_xyz, i, j)
        bo = info.bo_matrix[i, j]
        if bo * bl == 0:
            raise ZeroDivisionError(f"Division by zero for bond {atoms}.")
        return round(bo / bl, 8)

    def param_angle(atoms):
        """Angle force constant, derived from the two adjacent bond orders and lengths."""
        i, j, k = atoms
        bl1 = bondlength(info.fortran_xyz, i, j)
        bl2 = bondlength(info.fortran_xyz, j, k)
        prod = info.bo_matrix[i, j] * info.bo_matrix[j, k]
        if bl1 * bl2 * prod == 0:
            raise ZeroDivisionError(f"Division by zero for angle {atoms}.")
        return round((prod / (bl1 * bl2)) ** 0.5, 8)

    def param_dihedral(atoms):
        """Dihedral force constant, derived from the three chain bond orders and lengths (0.5 fallback for improper dihedral bonds)."""
        temp_bo = info.bo_matrix.copy()
        i, j, k, l = atoms
        bl1 = bondlength(info.fortran_xyz, i, j)
        bl2 = bondlength(info.fortran_xyz, j, k)
        bl3 = bondlength(info.fortran_xyz, k, l)
        # for improper dihedrals
        if temp_bo[i, j] == 0:
            temp_bo[i, j] = 0.5
        if temp_bo[j, k] == 0:
            temp_bo[j, k] = 0.5
        if temp_bo[k, l] == 0:
            temp_bo[k, l] = 0.5
        prod = temp_bo[i, j] * temp_bo[j, k] * temp_bo[k, l]
        # if bl1 * bl2 * bl3 * prod == 0:
        #     raise ZeroDivisionError(f"Division by zero for dihedral {atoms}.")
        return round((prod / (bl1 * bl2 * bl3)) ** (1 / 3), 8)

    def param_repulsive(_atoms):
        """Starting repulsive parameter (same value for every non-bonded pair)."""
        return repulsive_start

    # ============================================================
    # Assemble ForceField DataFrames
    # =============================================================

    ff.bonds = pd.DataFrame({"type": "bonds", "atoms": get_bond_atoms(A)})
    ff.bonds["reference_value"] = ff.bonds["atoms"].apply(ref_bond)
    ff.bonds["parameter"] = ff.bonds["atoms"].apply(param_bond)

    ff.angles = pd.DataFrame({"type": "angles", "atoms": get_angle_atoms(A)})
    ff.angles["reference_value"] = ff.angles["atoms"].apply(ref_angle)
    ff.angles["parameter"] = ff.angles["atoms"].apply(param_angle)

    ff.dihedrals = pd.DataFrame({"type": "dihedrals", "atoms": get_dihedral_atoms(A)})

    if priorities is not None:
        filtered_result = filter_dihedrals(
            ff.dihedrals["atoms"].tolist(), priorities, A
        )
        filtered_result = _remove_duplicate_proper_dihedrals(filtered_result)
        atoms_list = [d[0] for d in filtered_result]
        is_proper = [d[1] for d in filtered_result]
        logger.debug(f"The results are: {filtered_result}")
        ff.dihedrals = pd.DataFrame(
            {"type": "dihedrals", "atoms": atoms_list, "proper_dihedral": is_proper}
        )
    else:
        ff.dihedrals["proper_dihedral"] = True

    ff.dihedrals["reference_value"] = ff.dihedrals["atoms"].apply(ref_dihedral)
    ff.dihedrals["parameter"] = ff.dihedrals["atoms"].apply(param_dihedral)

    # Drop dihedrals built on a (near-)linear bond angle (undefined torsion, diverging Hessian).
    is_linear = ff.dihedrals.apply(
        lambda row: _is_at_linear_angle(
            row["atoms"], row["proper_dihedral"], info, linear_angle_tolerance_deg
        ),
        axis=1,
    )
    if is_linear.any():
        logger.warning(
            f"Dropping {int(is_linear.sum())} dihedral(s) built on a bond angle within "
            f"{linear_angle_tolerance_deg} degrees of 0/180 (undefined torsion): "
            f"{ff.dihedrals.loc[is_linear, 'atoms'].tolist()}"
        )
    ff.dihedrals = ff.dihedrals[~is_linear].reset_index(drop=True)

    ff.repulsive = pd.DataFrame({"type": "repulsive", "atoms": get_repulsive_atoms(A)})
    ff.repulsive["reference_value"] = ff.repulsive["atoms"].apply(ref_repulsive)
    ff.repulsive["parameter"] = ff.repulsive["atoms"].apply(param_repulsive)
    hydrogen_bonds = _find_hydrogen_bonding(ff=ff, info=info)

    # Add hydrogen bonds to ff.bonds and remove from ff.repulsive
    if hydrogen_bonds:
        hbond_bonds = list(hydrogen_bonds.keys())
        hbond_angles = [
            (donor, h, data["bonded_atom"])
            for (donor, h), data in hydrogen_bonds.items()
        ]

        # Add hydrogen bonds to bonds dataframe
        hbond_df = pd.DataFrame({"type": "bonds", "atoms": hbond_bonds})
        hbond_df["reference_value"] = hbond_df["atoms"].apply(ref_bond)
        hbond_df["parameter"] = 0.5
        ff.bonds = pd.concat([ff.bonds, hbond_df], ignore_index=True)

        # Sort bonds
        atoms_array = np.array(ff.bonds["atoms"].tolist())
        ff.bonds = ff.bonds.iloc[
            np.lexsort((atoms_array[:, 1], atoms_array[:, 0]))
        ].reset_index(drop=True)

        # Add hydrogen bond angles to angles dataframe
        hangle_df = pd.DataFrame({"type": "angles", "atoms": hbond_angles})
        hangle_df["reference_value"] = hangle_df["atoms"].apply(ref_angle)
        hangle_df["parameter"] = 0.25
        ff.angles = pd.concat([ff.angles, hangle_df], ignore_index=True)

        # Sort angles
        angles_array = np.array(ff.angles["atoms"].tolist())
        ff.angles = ff.angles.iloc[
            np.lexsort((angles_array[:, 2], angles_array[:, 1], angles_array[:, 0]))
        ].reset_index(drop=True)

        # Remove hydrogen bonds from repulsive
        ff.repulsive = ff.repulsive[
            ~ff.repulsive["atoms"].isin(hbond_bonds)
        ].reset_index(drop=True)


def _find_hydrogen_bonding(ff: ForceField, info: StructuralInformation) -> dict:
    """Identifies hydrogen bonding among atoms for repulsive interactions by distance, angle and atom type.

    Args:
        ff (ForceField): ForceField object containing repulsive terms
        info (StructuralInformation): Structural information with atom types, coordinates, and connectivity

    Returns:
        dict: Dictionary of hydrogen bonds and their properties, keyed by (donor_atom, h_atom) with values containing bond length, angle, and bonded atom information for example {(21, 68): {'bl': 3.6, 'angle': 2.9, 'bonded_atom': 66}, (67, 34): {'bl': 3.8, 'angle': 3.0, 'bonded_atom': 5}}
    """
    import math

    hbond_donors = {"O", "N", "F"}
    angle_threshold_deg = 30.0
    angle_threshold_rad = math.radians(angle_threshold_deg)
    factor = 1.0

    hydrogen_bonds = {}
    logger.debug(
        f"Starting hydrogen bond detection among {len(ff.repulsive)} repulsive pairs."
    )

    for _, row in ff.repulsive.iterrows():
        atoms = row["atoms"]
        atom_i, atom_j = atoms

        elem_i = info.atom_types[atom_i]
        elem_j = info.atom_types[atom_j]

        is_donor_h = (elem_i in hbond_donors and elem_j == "H") or (
            elem_j in hbond_donors and elem_i == "H"
        )

        if not is_donor_h:
            continue

        if elem_j == "H":
            h_atom = atom_j
            donor_atom = atom_i
        else:
            h_atom = atom_i
            donor_atom = atom_j

        bl = bondlength(info.fortran_xyz, donor_atom, h_atom)
        vdw_dist = info.vander_matrix[donor_atom, h_atom]

        if bl >= factor * vdw_dist:
            continue

        bonded_to_h = []
        for neighbor_idx in range(info.nat):
            if neighbor_idx != donor_atom and neighbor_idx != h_atom:
                if info.bo_matrix[h_atom, neighbor_idx] > 0:
                    bonded_to_h.append(neighbor_idx)
        logger.debug(f"Atom {h_atom} is bonded to atoms: {bonded_to_h}")

        for bonded_atom in bonded_to_h:
            ang = angle(info.fortran_xyz, donor_atom, h_atom, bonded_atom)
            logger.debug(
                f"Calculating angle for pair ({donor_atom}, {h_atom}, {bonded_atom}): {math.degrees(ang):.2f}°"
            )

            if abs(ang - math.pi) < angle_threshold_rad:
                hydrogen_bonds[(donor_atom, h_atom)] = {
                    "bl": bl,
                    "angle": ang,
                    "bonded_atom": bonded_atom,
                }
                logger.info(
                    f"Hydrogen bond found: {info.atom_types[donor_atom]}"
                    f"({donor_atom})-H({h_atom})"
                    f"...{info.atom_types[bonded_atom]}({bonded_atom}), "
                    f"angle: {math.degrees(ang):.2f}°, "
                    f"distance: {bl:.4f}"
                )
    logger.debug(f"Total hydrogen bonds detected: {hydrogen_bonds}")
    return hydrogen_bonds
