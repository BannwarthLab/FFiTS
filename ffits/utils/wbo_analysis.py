"""
WBO (Wiberg Bond Order) analysis utilities for comparing structures.

Provides functions to identify bonds that are changing between reactant and product structures
based on their WBO differences. Uses 0-based indexing for atoms.
"""

from typing import Dict, List, Tuple
import numpy as np
from ffits.datatype.structure_data import StructuralInformation, Structure


def _normalize_wbo_to_0based(wbo_dict: Dict) -> Dict:
    """
    Convert WBO dictionary from 1-based to 0-based atom indexing.
    
    Parameters
    ----------
    wbo_dict : dict
        Dictionary with 1-based bond tuples as keys
    
    Returns
    -------
    dict
        Dictionary with 0-based bond tuples as keys
    """
    normalized = {}
    for bond, value in wbo_dict.items():
        new_bond = (bond[0] - 1, bond[1] - 1)
        # Ensure consistent ordering (smaller index first)
        new_bond = tuple(sorted(new_bond))
        normalized[new_bond] = value
    return normalized


def compare_wbo_differences(
    reac_info: StructuralInformation,
    prod_info: StructuralInformation,
    threshold: float = 0.5
) -> Dict:
    """
    Compare WBO differences between reactant and product structures to identify changing bonds.
    
    This function calculates the absolute difference in Wiberg Bond Order (WBO) for each bond
    between reactant and product structures, and identifies bonds where the difference exceeds
    a specified threshold. Uses 0-based atom indexing.
    
    Parameters
    ----------
    reac_info : StructuralInformation
        Reactant structure containing WBO data in info.wbo dictionary
    prod_info : StructuralInformation
        Product structure containing WBO data in info.wbo dictionary
    threshold : float, optional
        Minimum WBO difference to consider a bond as "changing" (default: 0.5)
    
    Returns
    -------
    dict
        Dictionary containing (all with 0-based atom indices):
        - 'changing_bonds': List of tuples (bond_pair, wbo_change, reactant_wbo, product_wbo)
        - 'disappearing_bonds': List of tuples (bond, reactant_wbo) for bonds in reactant but not product
        - 'forming_bonds': List of tuples (bond, product_wbo) for bonds in product but not reactant
        - 'all_wbo_changes': Dict mapping each bond to its WBO change
    
    Raises
    ------
    ValueError
        If structures have no WBO data or incompatible atom counts
    
    Examples
    --------
    >>> result = compare_wbo_differences(reactant_info, product_info, threshold=0.3)
    >>> for bond, change, r_wbo, p_wbo in result['changing_bonds']:
    ...     print(f"Bond {bond}: {r_wbo:.3f} -> {p_wbo:.3f} (Δ = {change:.3f})")
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
    
    # Convert to 0-based indexing
    reactant_wbo = _normalize_wbo_to_0based(reac_info.wbo)
    product_wbo = _normalize_wbo_to_0based(prod_info.wbo)
    
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
            'change': wbo_change,
            'reactant_wbo': r_wbo,
            'product_wbo': p_wbo
        }
        
        if wbo_change >= threshold:
            changing_bonds.append((bond, wbo_change, r_wbo, p_wbo))
    
    # Identify disappearing and forming bonds
    disappearing_bonds = [
        (bond, reactant_wbo[bond])
        for bond in reactant_wbo
        if bond not in product_wbo
    ]
    
    forming_bonds = [
        (bond, product_wbo[bond])
        for bond in product_wbo
        if bond not in reactant_wbo
    ]
    
    # Sort by magnitude of change
    changing_bonds.sort(key=lambda x: x[1], reverse=True)
    disappearing_bonds.sort(key=lambda x: x[1], reverse=True)
    forming_bonds.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'changing_bonds': changing_bonds,
        'disappearing_bonds': disappearing_bonds,
        'forming_bonds': forming_bonds,
        'all_wbo_changes': all_wbo_changes,
        'threshold': threshold
    }


def get_wbo_matrix_difference(
    reactant: Structure,
    product: Structure
) -> np.ndarray:
    """
    Calculate element-wise WBO matrix difference between product and reactant.
    Uses 0-based indexing.
    
    Parameters
    ----------
    reactant : Structure
        Reactant structure
    product : Structure
        Product structure
    
    Returns
    -------
    np.ndarray
        Difference matrix where element [i,j] contains product_wbo[i,j] - reactant_wbo[i,j]
        with 0-based indexing
    """
    
    if reactant.info.nat != product.info.nat:
        raise ValueError("Structures must have same number of atoms")
    
    nat = reactant.info.nat
    diff_matrix = np.zeros((nat, nat))
    
    # Convert to 0-based indexing
    reactant_wbo = _normalize_wbo_to_0based(reactant.info.wbo)
    product_wbo = _normalize_wbo_to_0based(product.info.wbo)
    
    # Get difference for all bonds
    for bond in set(reactant_wbo.keys()) | set(product_wbo.keys()):
        i, j = bond  # Already 0-based after normalization
        r_wbo = reactant_wbo.get(bond, 0.0)
        p_wbo = product_wbo.get(bond, 0.0)
        diff_matrix[i, j] = p_wbo - r_wbo
        diff_matrix[j, i] = p_wbo - r_wbo  # Symmetric matrix
    
    return diff_matrix


def identify_affected_atoms(
    comparison_result: Dict,
    threshold: float = 0.5
) -> List[Tuple[int, List]]:
    """
    Identify atoms involved in bonds with WBO changes above threshold.
    Uses 0-based atom indexing.
    
    Parameters
    ----------
    comparison_result : dict
        Result dictionary from compare_wbo_differences()
    threshold : float, optional
        Minimum WBO change to consider (default: 0.5)
    
    Returns
    -------
    list of tuples
        List of (atom_index, list_of_affected_bonds) tuples with 0-based indices,
        sorted by number of affected bonds
    """
    
    atom_bonds = {}
    
    # Collect bonds with sufficient change
    for bond, change, _, _ in comparison_result['changing_bonds']:
        if change >= threshold:
            atom1, atom2 = bond  # Already 0-based
            if atom1 not in atom_bonds:
                atom_bonds[atom1] = []
            if atom2 not in atom_bonds:
                atom_bonds[atom2] = []
            atom_bonds[atom1].append(bond)
            atom_bonds[atom2].append(bond)
    
    # Sort by number of affected bonds
    affected_atoms = sorted(
        atom_bonds.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    return affected_atoms
