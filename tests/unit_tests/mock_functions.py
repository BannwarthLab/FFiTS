"""
Mock functions and fixtures for unit testing.
Provides mock implementations of all datatypes, readers, external calls, and force field construction.
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Callable
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from ffits.datatype.calculation_data import (
    System, PathData, CalculationOptions, TSCalculationOptions, 
    Postprocessing, CalculationData
)
from ffits.datatype.structure_data import (
    Structure, StructurePath, ForceField, StructuralInformation,
    Name, convert_xyz_to_fortranstyle, angstrom2bohr, 
    atom_symbol_to_number, get_vander_matrix
)
from ffits.datatype.reaction_data import Reaction


# ============================================================================
# MOCK XYZ DATA AND MOLECULAR STRUCTURES
# ============================================================================

def create_mock_xyz_array(nat: int = 8, seed: int = 42) -> np.ndarray:
    """
    Create a mock xyz coordinate array with random valid atomic coordinates.
    
    Parameters
    ----------
    nat : int
        Number of atoms
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Shape (nat, 3) array of coordinates
    """
    np.random.seed(seed)
    xyz = np.random.randn(nat, 3) * 2 + 1  # Reasonable atomic coordinates
    return xyz.astype(float)


def create_mock_atom_types(nat: int = 8, elements: Optional[List[str]] = None) -> np.ndarray:
    """
    Create a mock atom types array.
    
    Parameters
    ----------
    nat : int
        Number of atoms
    elements : Optional[List[str]]
        Specific elements to use. If None, defaults to ['C', 'H', 'H', 'O', 'C', ...].
        
    Returns
    -------
    np.ndarray
        Array of atom type symbols
    """
    if elements is None:
        default_elements = ['C', 'H', 'O', 'N', 'S', 'F']
        elements = [default_elements[i % len(default_elements)] for i in range(nat)]
    return np.array(elements[:nat])


def create_mock_wbo_dict(nat: int = 8, seed: int = 42) -> Dict[Tuple[int, int], float]:
    """
    Create a mock WBO (Wiberg Bond Order) dictionary.
    
    Parameters
    ----------
    nat : int
        Number of atoms
    seed : int
        Random seed
        
    Returns
    -------
    Dict[Tuple[int, int], float]
        Dictionary mapping atom pairs to WBO values
    """
    np.random.seed(seed)
    wbo_dict = {}
    
    # Create bonds for nearby atoms
    for i in range(1, nat + 1):
        for j in range(i + 1, min(i + 3, nat + 1)):  # Connect to nearby atoms
            bond_order = np.random.uniform(0.5, 1.5)  # Typical single/double bond range
            wbo_dict[(i, j)] = bond_order
    
    return wbo_dict


def create_mock_hessian(nat: int = 8, seed: int = 42) -> np.ndarray:
    """
    Create a mock Hessian matrix (symmetric, positive semi-definite).
    
    Parameters
    ----------
    nat : int
        Number of atoms
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Shape (3*nat, 3*nat) Hessian matrix
    """
    np.random.seed(seed)
    dim = 3 * nat
    
    # Create a symmetric matrix
    A = np.random.randn(dim, dim)
    hessian = (A + A.T) / 2
    
    # Make it positive semi-definite by adding a large diagonal
    hessian += np.eye(dim) * (np.abs(np.min(np.linalg.eigvals(hessian))) + 1)
    
    return hessian.astype(float)


# ============================================================================
# MOCK READERS AND IO FUNCTIONS
# ============================================================================

def mock_readin_xyz(xyz_path: str) -> Tuple[int, str, np.ndarray, List[str]]:
    """
    Mock version of readin_xyz from io/reader.py
    
    Parameters
    ----------
    xyz_path : str
        Path to XYZ file (not actually read)
        
    Returns
    -------
    Tuple[int, str, np.ndarray, List[str]]
        (nat, comment, xyz_coords, atom_types)
    """
    nat = 3
    comment = "Mock XYZ file"
    xyz = create_mock_xyz_array(nat)
    atom_types = create_mock_atom_types(nat)
    return nat, comment, xyz, atom_types


def mock_read_wbo_file(wbo_path: str) -> Dict[Tuple[int, int], float]:
    """
    Mock version of read_wbo_file from io/reader.py
    
    Parameters
    ----------
    wbo_path : str
        Path to WBO file (not actually read)
        
    Returns
    -------
    Dict[Tuple[int, int], float]
        Mock WBO dictionary
    """
    return create_mock_wbo_dict(nat=3)


def mock_read_hessian(file_path: str) -> np.ndarray:
    """
    Mock version of read_hessian from io/reader.py
    
    Parameters
    ----------
    file_path : str
        Path to Hessian file (not actually read)
        
    Returns
    -------
    np.ndarray
        Mock Hessian matrix
    """
    return create_mock_hessian(nat=3)


# ============================================================================
# MOCK STRUCTURAL INFORMATION
# ============================================================================

def create_mock_structural_information(
    nat: int = 8,
    xyz: Optional[np.ndarray] = None,
    wbo_dict: Optional[Dict[Tuple[int, int], float]] = None,
    atom_types: Optional[np.ndarray] = None,
    hessian: Optional[np.ndarray] = None,
    seed: int = 42
) -> StructuralInformation:
    """
    Create a mock StructuralInformation object.
    
    Parameters
    ----------
    nat : int
        Number of atoms
    xyz : Optional[np.ndarray]
        Coordinate array. If None, generated randomly.
    wbo_dict : Optional[Dict]
        WBO dictionary. If None, generated randomly.
    atom_types : Optional[np.ndarray]
        Atom types. If None, generated randomly.
    hessian : Optional[np.ndarray]
        Hessian matrix. If None, generated randomly.
    seed : int
        Random seed
        
    Returns
    -------
    StructuralInformation
        Mock structural information object
    """
    if xyz is None:
        xyz = create_mock_xyz_array(nat, seed)
    if wbo_dict is None:
        wbo_dict = create_mock_wbo_dict(nat, seed)
    if atom_types is None:
        atom_types = create_mock_atom_types(nat)
    if hessian is None:
        hessian = create_mock_hessian(nat, seed)
    
    return StructuralInformation(
        nat=nat,
        xyz=xyz,
        wbo_dict=wbo_dict,
        atom_types=atom_types,
        hessian=hessian
    )


# ============================================================================
# MOCK FORCE FIELD CONSTRUCTION
# ============================================================================

def create_mock_forcefield_dataframe() -> pd.DataFrame:
    """
    Create a mock force field DataFrame with bonds, angles, dihedrals, and repulsive terms.
    
    Returns
    -------
    pd.DataFrame
        Mock force field data
    """
    data = {
        'type': ['bonds', 'bonds', 'angles', 'angles', 'dihedrals', 'dihedrals', 'repulsive'],
        'atoms': [(1, 2), (2, 3), (1, 2, 3), (2, 3, 4), (1, 2, 3, 4), (2, 3, 4, 5), (1, 2)],
        'parameter': [0.4, 0.35, 0.2, 0.22, 0.1, 0.15, 0.01],
        'reference_value': [1.5, 1.4, 120.0, 119.0, 180.0, 175.0, 3.0]
    }
    return pd.DataFrame(data)


def create_mock_forcefield(
    nat: int = 8,
    ff_filename: str = "mock_ff.csv",
    energy_calculator: Optional[Callable] = None,
    gradient_calculator: Optional[Callable] = None,
    hessian_calculator: Optional[Callable] = None,
    seed: int = 42
) -> ForceField:
    """
    Create a mock ForceField object.
    
    Parameters
    ----------
    nat : int
        Number of atoms
    ff_filename : str
        Force field filename
    energy_calculator : Optional[Callable]
        Energy calculation function. If None, uses a mock.
    gradient_calculator : Optional[Callable]
        Gradient calculation function. If None, uses a mock.
    hessian_calculator : Optional[Callable]
        Hessian calculation function. If None, uses a mock.
    seed : int
        Random seed
        
    Returns
    -------
    ForceField
        Mock ForceField object
    """
    np.random.seed(seed)
    
    # Create default mock functions if not provided
    if energy_calculator is None:
        energy_calculator = lambda xyz_disp, ff: np.random.randn() * 0.1
    
    if gradient_calculator is None:
        gradient_calculator = lambda xyz_disp, ff: np.random.randn(3 * nat) * 0.01
    
    if hessian_calculator is None:
        hessian_calculator = lambda xyz_disp, ff: create_mock_hessian(nat, seed)
    
    ff = ForceField(
        nat=nat,
        ff_filename=ff_filename,
        readff=False,
        energy_calculator=energy_calculator,
        gradient_calculator=gradient_calculator,
        hessian_calculator=hessian_calculator
    )
    
    # Populate with mock data
    df = create_mock_forcefield_dataframe()
    ff.bonds = df[df['type'] == 'bonds'].copy()
    ff.angles = df[df['type'] == 'angles'].copy()
    ff.dihedrals = df[df['type'] == 'dihedrals'].copy()
    ff.repulsive = df[df['type'] == 'repulsive'].copy()
    
    return ff


# ============================================================================
# MOCK STRUCTURE AND PATHS
# ============================================================================

def create_mock_structure_path(
    base_dir: str = "/tmp",
    name: str = "mock_structure"
) -> StructurePath:
    """
    Create a mock StructurePath object.
    
    Parameters
    ----------
    base_dir : str
        Base directory for files
    name : str
        Name of the structure
        
    Returns
    -------
    StructurePath
        Mock structure path object
    """
    return StructurePath(
        xyz_filename=f"{base_dir}/{name}.xyz",
        hess_filename=f"{base_dir}/{name}.hess",
        wbo_filename=f"{base_dir}/{name}.wbo",
        ff_filename=f"{base_dir}/{name}_ff.csv"
    )


def create_mock_structure(
    nat: int = 8,
    base_dir: str = "/tmp",
    name: str = "mock_structure",
    seed: int = 42
) -> Structure:
    """
    Create a complete mock Structure object.
    
    Parameters
    ----------
    nat : int
        Number of atoms
    base_dir : str
        Base directory for files
    name : str
        Name of the structure
    seed : int
        Random seed
        
    Returns
    -------
    Structure
        Complete mock structure
    """
    path = create_mock_structure_path(base_dir, name)
    ff = create_mock_forcefield(nat=nat, ff_filename=path.ff_filename, seed=seed)
    info = create_mock_structural_information(nat=nat, seed=seed)
    
    return Structure(
        path=path,
        ff=ff,
        info=info
    )


# ============================================================================
# MOCK REACTION DATA
# ============================================================================

def create_mock_reaction(
    nat: int = 8,
    seed: int = 42
) -> Reaction:
    """
    Create a mock Reaction object with two structures.
    
    Parameters
    ----------
    nat : int
        Number of atoms in each structure
    seed : int
        Random seed
        
    Returns
    -------
    Reaction
        Mock reaction object
    """
    structure1 = create_mock_structure(nat=nat, name="reactant", seed=seed)
    structure2 = create_mock_structure(nat=nat, name="product", seed=seed + 1)
    
    return Reaction(structure1, structure2)


# ============================================================================
# MOCK CALCULATION DATA
# ============================================================================

def create_mock_system(
    charge: int = 0,
    multiplicity: int = 1
) -> System:
    """
    Create a mock System object.
    
    Parameters
    ----------
    charge : int
        System charge
    multiplicity : int
        System multiplicity
        
    Returns
    -------
    System
        Mock system
    """
    return System(charge=charge, multiplicity=multiplicity)


def create_mock_path_data(
    xyz_filename: str = "structure.xyz",
    wbo_filename: str = "structure.wbo",
    hessian_filename: str = "structure.hess",
    ff_filename: str = "structure_ff.csv"
) -> PathData:
    """
    Create a mock PathData object.
    
    Parameters
    ----------
    xyz_filename : str
        XYZ file path
    wbo_filename : str
        WBO file path
    hessian_filename : str
        Hessian file path
    ff_filename : str
        Force field file path
        
    Returns
    -------
    PathData
        Mock path data
    """
    return PathData(
        xyz_filename=xyz_filename,
        wbo_filename=wbo_filename,
        hessian_filename=hessian_filename,
        ff_filename=ff_filename
    )


def create_mock_calculation_options(
    geometry_optimization: bool = True,
    wbo_calc: bool = True,
    hessian_calc: bool = True,
    ff_parameterization: bool = False
) -> CalculationOptions:
    """
    Create a mock CalculationOptions object.
    
    Parameters
    ----------
    geometry_optimization : bool
        Perform geometry optimization
    wbo_calc : bool
        Perform WBO calculation
    hessian_calc : bool
        Perform Hessian calculation
    ff_parameterization : bool
        Perform force field parameterization
        
    Returns
    -------
    CalculationOptions
        Mock calculation options
    """
    return CalculationOptions(
        geometry_optimization=geometry_optimization,
        wbo_calc=wbo_calc,
        hessian_calc=hessian_calc,
        ff_parameterization=ff_parameterization
    )


def create_mock_ts_calculation_options(
    factor_reactant: float = 0.5,
    factor_product: float = 0.5,
    optimizer: str = "molbar-optimizer"
) -> TSCalculationOptions:
    """
    Create a mock TSCalculationOptions object.
    
    Parameters
    ----------
    factor_reactant : float
        Reactant weighting factor
    factor_product : float
        Product weighting factor
    optimizer : str
        Optimizer method
        
    Returns
    -------
    TSCalculationOptions
        Mock TS calculation options
    """
    return TSCalculationOptions(
        factor_reactant=factor_reactant,
        factor_product=factor_product,
        optimizer=optimizer
    )


def create_mock_calculation_data(
    charge: int = 0,
    multiplicity: int = 1
) -> CalculationData:
    """
    Create a complete mock CalculationData object.
    
    Parameters
    ----------
    charge : int
        System charge
    multiplicity : int
        System multiplicity
        
    Returns
    -------
    CalculationData
        Complete mock calculation data
    """
    return CalculationData(
        system=create_mock_system(charge=charge, multiplicity=multiplicity),
        reactant_path=create_mock_path_data(
            xyz_filename="reactant.xyz",
            wbo_filename="reactant.wbo",
            hessian_filename="reactant.hess",
            ff_filename="reactant_ff.csv"
        ),
        product_path=create_mock_path_data(
            xyz_filename="product.xyz",
            wbo_filename="product.wbo",
            hessian_filename="product.hess",
            ff_filename="product_ff.csv"
        ),
        reactant_calc=create_mock_calculation_options(geometry_optimization=True),
        product_calc=create_mock_calculation_options(geometry_optimization=True),
        ts_calc=create_mock_ts_calculation_options(),
        ts_path=create_mock_path_data(
            xyz_filename="ts.xyz",
            wbo_filename="ts.wbo",
            hessian_filename="ts.hess",
            ff_filename="ts_ff.csv"
        ),
        postprocessing=Postprocessing(relaxation="None")
    )


# ============================================================================
# MOCK EXTERNAL CALLS
# ============================================================================

class MockXtb:
    """Mock xTB wrapper for testing."""
    
    def __init__(self, chrg: int = 0, mult: int = 1, xtb_path: str = "xtb"):
        self.chrg = chrg
        self.uhf = mult - 1
        self.xtb_path = xtb_path
    
    def geomopt(self, xyz_filename: str) -> Tuple[int, float, np.ndarray]:
        """
        Mock geometry optimization with xTB.
        
        Returns
        -------
        Tuple[int, float, np.ndarray]
            (nat, energy, xyz_coords)
        """
        nat = 3
        energy = np.random.randn() * 0.1 - 10.0  # Mock energy
        xyz = create_mock_xyz_array(nat)
        return nat, energy, xyz
    
    def wbo_calc(self, xyz_filename: str) -> Dict[Tuple[int, int], float]:
        """Mock WBO calculation."""
        return create_mock_wbo_dict(nat=3)
    
    def hessian_calc(self, xyz_filename: str) -> np.ndarray:
        """Mock Hessian calculation."""
        return create_mock_hessian(nat=3)


class MockCrest:
    """Mock CREST wrapper for testing."""
    
    def __init__(self, crest_path: str = "crest"):
        self.crest_path = crest_path
    
    def check_convergence(self, filepath: str, keyword: str = "geometry successfully optimized") -> bool:
        """Mock convergence check."""
        return True
    
    def get_rmsd(self, xyz1: str, xyz2: str) -> float:
        """Mock RMSD calculation."""
        return np.random.uniform(0.001, 0.1)
    
    def write_input2file(self, input_str: str, filename: str) -> None:
        """Mock input file writing."""
        pass
    
    def run_input(self, input_file: str, job_name: str) -> int:
        """Mock CREST job execution."""
        return 0


class MockGeometryOptimization:
    """Mock geometry optimization wrapper."""
    
    @staticmethod
    def with_xtb(input_structure: StructurePath, xtb_path: str = 'xtb') -> Tuple[int, float, np.ndarray]:
        """Mock xTB-based geometry optimization."""
        nat = 3
        energy = np.random.randn() * 0.1 - 10.0
        xyz = create_mock_xyz_array(nat)
        return nat, energy, xyz
    
    @staticmethod
    def with_ff_potential_crest(input_structure: StructurePath, ff: ForceField, output_xyz: str = '') -> int:
        """Mock FF-based CREST geometry optimization."""
        return 0


# ============================================================================
# MOCK OPTIMIZER FUNCTIONS
# ============================================================================

def mock_anc_optimizer(
    xyz_start: np.ndarray,
    ff: ForceField,
    atom_types: np.ndarray,
    g_tol: float = 1e-2,
    e_tol: float = 1e-4,
    x_tol: float = 1e-3,
    max_micro_steps: int = 1,
    trajectory_filename: str = 'trajectory.xyz',
    final_geometry_filename: str = 'optimized.xyz'
) -> Tuple[bool, float, np.ndarray, int, float, str]:
    """
    Mock ANC optimizer function.
    
    Returns
    -------
    Tuple[bool, float, np.ndarray, int, float, str]
        (converged, energy, final_geom, steps, time, message)
    """
    nat = len(atom_types)
    converged = True
    energy = np.random.randn() * 0.1 - 10.0
    final_geom = xyz_start + np.random.randn(*xyz_start.shape) * 0.01
    steps = 42
    time = 1.23
    message = "Mock convergence successful"
    
    return converged, energy, final_geom, steps, time, message


def mock_scipy_optimizer(
    xyz_start: np.ndarray,
    ff: ForceField,
    struc: StructuralInformation
) -> dict:
    """
    Mock scipy optimizer result.
    
    Returns
    -------
    dict
        Mock optimization result dictionary
    """
    return {
        'x': xyz_start + np.random.randn(*xyz_start.shape) * 0.01,
        'fun': np.random.randn() * 0.1 - 10.0,
        'nit': 42,
        'nfev': 84,
        'success': True,
        'message': 'Mock convergence successful'
    }


# ============================================================================
# MOCK UTILITY FUNCTIONS
# ============================================================================

def mock_failed_anc_opt(filename: str) -> bool:
    """Mock check for failed ANC optimization."""
    return False


def mock_write_last_valid_xyz(
    trajectory_filename: str = 'trajectory.xyz',
    final_geometry_filename: str = 'optimized.xyz'
) -> None:
    """Mock writing last valid XYZ geometry."""
    pass


# ============================================================================
# CONTEXT MANAGERS FOR MOCKING
# ============================================================================

class MockIOContext:
    """Context manager for mocking all IO operations."""
    
    def __init__(self):
        self.patches = []
    
    def __enter__(self):
        """Enter mock context."""
        from ffits.io import reader
        from ffits.external import xtb, crest
        from ffits.external import molbar_optimizer
        
        self.patches = [
            patch.object(reader, 'readin_xyz', side_effect=mock_readin_xyz),
            patch.object(reader, 'read_wbo_file', side_effect=mock_read_wbo_file),
            patch.object(reader, 'read_hessian', side_effect=mock_read_hessian),
            patch.object(xtb, 'Xtb', MockXtb),
            patch.object(crest, 'Crest', MockCrest),
            patch.object(molbar_optimizer, 'anc_optimizer', side_effect=mock_anc_optimizer),
            patch.object(molbar_optimizer, 'scipy_optimizer', side_effect=mock_scipy_optimizer),
        ]
        
        for p in self.patches:
            p.start()
    
    def __exit__(self, *args):
        """Exit mock context."""
        for p in self.patches:
            p.stop()


# ============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# ============================================================================

def create_mock_test_environment(
    nat: int = 8,
    charge: int = 0,
    multiplicity: int = 1,
    seed: int = 42
) -> dict:
    """
    Create a complete mock test environment with all components.
    
    Parameters
    ----------
    nat : int
        Number of atoms
    charge : int
        System charge
    multiplicity : int
        System multiplicity
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Dictionary containing all mock objects
    """
    np.random.seed(seed)
    
    return {
        'system': create_mock_system(charge=charge, multiplicity=multiplicity),
        'structure': create_mock_structure(nat=nat, seed=seed),
        'reaction': create_mock_reaction(nat=nat, seed=seed),
        'calculation_data': create_mock_calculation_data(charge=charge, multiplicity=multiplicity),
        'forcefield': create_mock_forcefield(nat=nat, seed=seed),
        'structural_info': create_mock_structural_information(nat=nat, seed=seed),
        'xyz': create_mock_xyz_array(nat, seed=seed),
        'atom_types': create_mock_atom_types(nat),
        'wbo_dict': create_mock_wbo_dict(nat, seed=seed),
        'hessian': create_mock_hessian(nat, seed=seed),
        'xtb': MockXtb(chrg=charge, mult=multiplicity),
        'crest': MockCrest(),
        'geom_opt': MockGeometryOptimization(),
    }
