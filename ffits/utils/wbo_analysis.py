"""
WBO (Wiberg Bond Order) analysis utilities for comparing structures.

Provides functions to identify bonds that are changing between reactant and product structures
based on their WBO differences. Uses 0-based indexing for atoms.
"""

from typing import Dict, List, Tuple
import numpy as np
from ffits.datatype.structure_data import StructuralInformation, Structure


def _sort_bond_keys(wbo_dict: Dict) -> Dict:
    """Return the WBO dictionary with sorted bond-tuple keys (smaller index first).

    Args:
        wbo_dict (dict): Dictionary with 0-based bond tuples as keys.

    Returns:
        dict: Dictionary with sorted 0-based bond tuples as keys.
    """
    return {tuple(sorted(bond)): value for bond, value in wbo_dict.items()}


def compare_wbo_differences(
    reac_info: StructuralInformation,
    prod_info: StructuralInformation,
    threshold: float = 0.5,
) -> Dict:
    """Compare WBO differences between reactant and product structures to identify changing bonds.

    Computes the absolute WBO difference for every bond present in either
    structure and flags those exceeding ``threshold``. Uses 0-based atom
    indexing.

    Args:
        reac_info (StructuralInformation): Reactant structure with WBO data in info.wbo.
        prod_info (StructuralInformation): Product structure with WBO data in info.wbo.
        threshold (float, optional): Minimum WBO difference to flag a bond as "changing". Defaults to 0.5.

    Returns:
        dict: 'changing_bonds' (list of (bond, change, reactant_wbo,
        product_wbo)), 'disappearing_bonds' (list of (bond, reactant_wbo)),
        'forming_bonds' (list of (bond, product_wbo)), 'all_wbo_changes'
        (dict per bond), and 'threshold'.

    Raises:
        ValueError: If either structure has no WBO data, or atom counts differ.
    """

    # Validate inputs
    if not reac_info.wbo:
        raise ValueError("Reactant structure has no WBO data")
    if not prod_info.wbo:
        raise ValueError("Product structure has no WBO data")
    if reac_info.nat != prod_info.nat:
        raise ValueError(
            f"Structures have different atom counts: "
            f"reactant={reac_info.nat}, product={prod_info.nat}"
        )

    reactant_wbo = _sort_bond_keys(reac_info.wbo)
    product_wbo = _sort_bond_keys(prod_info.wbo)

    # Initialize result containers
    changing_bonds = []
    all_wbo_changes = {}

    # Find all unique bonds across both structures
    all_bonds = set(reactant_wbo.keys()) | set(product_wbo.keys())

    # Compare WBO for each bond
    for bond in all_bonds:
        r_wbo = reactant_wbo.get(bond, 0.0)
        p_wbo = product_wbo.get(bond, 0.0)
        wbo_change = abs(p_wbo - r_wbo)

        all_wbo_changes[bond] = {
            "change": wbo_change,
            "reactant_wbo": r_wbo,
            "product_wbo": p_wbo,
        }

        if wbo_change >= threshold:
            changing_bonds.append((bond, wbo_change, r_wbo, p_wbo))

    # Identify disappearing and forming bonds
    disappearing_bonds = [
        (bond, reactant_wbo[bond]) for bond in reactant_wbo if bond not in product_wbo
    ]

    forming_bonds = [
        (bond, product_wbo[bond]) for bond in product_wbo if bond not in reactant_wbo
    ]

    # Sort by magnitude of change
    changing_bonds.sort(key=lambda x: x[1], reverse=True)
    disappearing_bonds.sort(key=lambda x: x[1], reverse=True)
    forming_bonds.sort(key=lambda x: x[1], reverse=True)

    return {
        "changing_bonds": changing_bonds,
        "disappearing_bonds": disappearing_bonds,
        "forming_bonds": forming_bonds,
        "all_wbo_changes": all_wbo_changes,
        "threshold": threshold,
    }


def get_wbo_matrix_difference(reactant: Structure, product: Structure) -> np.ndarray:
    """Calculate the element-wise WBO matrix difference (product - reactant), 0-based indexing.

    Args:
        reactant (Structure): Reactant structure.
        product (Structure): Product structure.

    Returns:
        np.ndarray: Symmetric matrix where element [i, j] is product_wbo[i,j] - reactant_wbo[i,j].
    """

    if reactant.info.nat != product.info.nat:
        raise ValueError("Structures must have same number of atoms")

    nat = reactant.info.nat
    diff_matrix = np.zeros((nat, nat))

    reactant_wbo = _sort_bond_keys(reactant.info.wbo)
    product_wbo = _sort_bond_keys(product.info.wbo)

    # Get difference for all bonds
    for bond in set(reactant_wbo.keys()) | set(product_wbo.keys()):
        i, j = bond
        r_wbo = reactant_wbo.get(bond, 0.0)
        p_wbo = product_wbo.get(bond, 0.0)
        diff_matrix[i, j] = p_wbo - r_wbo
        diff_matrix[j, i] = p_wbo - r_wbo  # Symmetric matrix

    return diff_matrix


def identify_affected_atoms(
    comparison_result: Dict, threshold: float = 0.5
) -> List[Tuple[int, List]]:
    """Identify atoms involved in bonds with WBO changes above threshold (0-based indexing).

    Args:
        comparison_result (dict): Result from :func:`compare_wbo_differences`.
        threshold (float, optional): Minimum WBO change to consider. Defaults to 0.5.

    Returns:
        list[tuple[int, list]]: (atom_index, affected_bonds) pairs, sorted
        by number of affected bonds (descending).
    """

    atom_bonds = {}

    # Collect bonds with sufficient change
    for bond, change, _, _ in comparison_result["changing_bonds"]:
        if change >= threshold:
            atom1, atom2 = bond  # Already 0-based
            if atom1 not in atom_bonds:
                atom_bonds[atom1] = []
            if atom2 not in atom_bonds:
                atom_bonds[atom2] = []
            atom_bonds[atom1].append(bond)
            atom_bonds[atom2].append(bond)

    # Sort by number of affected bonds
    affected_atoms = sorted(atom_bonds.items(), key=lambda x: len(x[1]), reverse=True)

    return affected_atoms
