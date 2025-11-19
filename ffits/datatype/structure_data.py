#!/bin/python

from dataclasses import dataclass
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
from collections.abc import Callable


VANDER_VALUES = np.array([
0.91, 0.92, # H, He
0.75, 1.28, 1.35, 1.32, 1.27, 1.22, 1.17, 1.13, # Li-Ne
1.04, 1.24, 1.49, 1.56, 1.55, 1.53, 1.49, 1.45, # Na-Ar
1.35, 1.34, # K, Ca
1.42, 1.42, 1.42, 1.42, 1.42, # Sc-Zn
1.42, 1.42, 1.42, 1.42, 1.42,
1.50, 1.57, 1.60, 1.61, 1.59, 1.57, # Ga-Kr
1.48, 1.46, # Rb, Sr
1.49, 1.49, 1.49, 1.49, 1.49, # Y-Cd
1.49, 1.49, 1.49, 1.49, 1.49,
1.52, 1.64, 1.71, 1.72, 1.72, 1.71, # In-Xe
2.00, 2.00,
2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, # La-Yb
2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
2.00, 2.00, 2.00, 2.00, 2.00, # Lu-Hg
2.00, 2.00, 2.00, 2.00, 2.00,
2.00, 2.00, 2.00, 2.00, 2.00, 2.00 # Tl-Rn
])

PERIODIC_TABLE = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86
}
def convert_xyz_to_fortranstyle(nat, xyz): 
    x = []
    y = []
    z = []
    for i in range(nat):
        x.append(xyz[i][0])
        y.append(xyz[i][1])
        z.append(xyz[i][2])
    return np.array([x,y,z],order='F')

def angstrom2bohr(val):
        if type(val) == np.array:
            return np.divide(val, 1/1.8897259)
        return val/(1/1.8897259)

def atom_symbol_to_number(symbol: str) -> int:
    """Convert an element symbol (e.g. 'C') to its atomic number (e.g. 6)."""
    try:
        return PERIODIC_TABLE[symbol.capitalize()]
    except KeyError:
        raise ValueError(f"Unknown atom symbol: {symbol}")
    
def get_vander_matrix(at: np.ndarray, vander_values=VANDER_VALUES, factor=1.8897259):
    """
    Build van der Waals interaction matrix.

    Parameters
    ----------
    nat : int
        Number of atoms.
    at : array-like of str
    vander_values : np.ndarray
        Reference van der Waals radii (length 86).
    factor : float
        Scaling factor.

    Returns
    -------
    vander_matrix : np.ndarray (nat x nat)
    """
    # Convert atomic numbers to 0-based indices
    radii = np.array([vander_values[atom_symbol_to_number(sym) - 1] * factor for sym in at])

    # Build the full symmetric matrix (outer sum)
    vander_matrix = radii[:, None] + radii[None, :]

    return vander_matrix

@dataclass
class Name:
    """
    Class, defining path names of data, generated during the calculation.
    """
    @staticmethod
    def modified_ff(input_str: str):
        return f'{input_str}'
    @staticmethod
    def fitted_ff(input_str: str):
        return f'{input_str}'
    @staticmethod
    def optimized_xyz(input_str: str):
        return f'opt_{input_str}'
    @staticmethod
    def aligned_xyz(input_str: str):
        return f'aligned_{input_str}'
    @staticmethod
    def original_xyz(input_str: str):
        return f'original_{input_str}'
    

@dataclass
class StructurePath:
    """
    Paths/Filenames and ID for a given structure.
    """
    xyz_filename: str 
    hess_filename: str 
    wbo_filename: str 
    ff_filename: str

class ForceField:
    """ 
    FF definition through FF parameters (np arrays starting with c_), reference values (bondlenghts, angles, etc) and corresponding atom numbers, which construct the bond / angle / dihedral angle / lj term. 
    """
    def __init__(self, nat: int, ff_filename: str, readff: bool = False, energy_calculator: Callable = None, gradient_calculator: Callable = None, hessian_calculator: Callable = None):
        self.nat = nat
        self.ff_filename = ff_filename
        self.columns = ['type', 'atoms', 'parameter', 'reference_value']
        self.bonds: pd.DataFrame     = pd.DataFrame(columns=self.columns)
        self.angles: pd.DataFrame    = pd.DataFrame(columns=self.columns)
        self.dihedrals: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.repulsive: pd.DataFrame = pd.DataFrame(columns=self.columns)
        if readff:
            self.readin(ff_filename)
            # if not self.correct_dimensions():
            #     raise Exception('Dimensions of reference values and given dimensions do not fit.')
        self.energy_calculator = energy_calculator
        self.gradient_calculator = gradient_calculator
        self.hessian_calculator = hessian_calculator

    def get_energy(self, xyz_displaced: np.ndarray): 
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.energy_calculator(angstrom2bohr(convert_xyz_to_fortranstyle(self.nat, xyz_displaced)), self)
        if len(xyz_displaced) == self.nat*3:
            return self.energy_calculator(angstrom2bohr(xyz_displaced.reshape(self.nat,3).T), self)
        return self.energy_calculator(xyz_displaced, self)
    
    def get_gradient(self, xyz_displaced: np.ndarray): 
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.gradient_calculator(angstrom2bohr(convert_xyz_to_fortranstyle(self.nat, xyz_displaced)), self)
        if len(xyz_displaced) == self.nat*3:
            return self.gradient_calculator(angstrom2bohr(xyz_displaced.reshape(self.nat,3).T), self)
        return self.gradient_calculator(xyz_displaced, self)
    
    def get_hessian(self, xyz_displaced: np.ndarray): 
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.hessian_calculator(angstrom2bohr(convert_xyz_to_fortranstyle(self.nat, xyz_displaced)), self)
        if len(xyz_displaced) == self.nat*3:
            return self.hessian_calculator(angstrom2bohr(xyz_displaced.reshape(self.nat,3).T), self)
        return self.hessian_calculator(xyz_displaced, self)

    def write(self):
        """Combine all parameter DataFrames and write to CSV."""
        def format_atoms(t):
            return "[" + " ".join(map(str, t)) + "]"

        df_combined = pd.concat([self.bonds, self.angles, self.dihedrals, self.repulsive], ignore_index=True)
        df_combined = df_combined.copy()
        df_combined['atoms'] = df_combined['atoms'].apply(format_atoms)
        df_combined.to_csv(self.ff_filename)
        print(f'[INFO] FF information written to {self.ff_filename}.')


    def readin(self, filename: str):
        """Read a force field CSV file and populate the corresponding DataFrames."""
        try:
            def parse_atoms(x):
                """Parse 'atoms' column into a tuple of integers."""
                if pd.isna(x):
                    return tuple()
                if isinstance(x, (list, tuple)):
                    # Already iterable — ensure tuple of ints
                    return tuple(int(i) for i in x)
                # Remove brackets and commas, split on whitespace
                x = re.sub(r'[\[\],]', ' ', str(x))
                tokens = x.split()
                return tuple(int(tok) for tok in tokens)

            df = pd.read_csv(
                filename,
                converters={
                    'atoms': parse_atoms,
                    'parameter': float,
                    'reference_value': float,
                },
                index_col=0
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"Force field file '{filename}' not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Force field file '{filename}' is empty or malformed.")
        except Exception as e:
            raise ValueError(f"Error while reading '{filename}': {e}")

        # --- Split into sub-dataframes ---
        self.bonds     = df[df['type'] == 'bonds'].copy()
        self.angles    = df[df['type'] == 'angles'].copy()
        self.dihedrals = df[df['type'] == 'dihedrals'].copy()
        self.repulsive = df[df['type'] == 'repulsive'].copy()



class StructuralInformation:
    """ 
    Information on the given structure.
        nat: number of atoms
        xyz: directly transferred into bohr
        wbo: Wilberg Bond Order as directory of bond pairs and corresponding WBO values
        complete_graph: networkx Graph object with atoms as nodes and bond order larger then 0 as edges.
        seperate_molecule_list: List of subgraphs not connectred by edges in complete_graph.
        molecule_count: number of seperate molecules in structure.
    """
    def __init__(self, nat: int, xyz: np.array, wbo_dict: dict, atom_types: np.array, hessian: np.ndarray = None):
        self.nat = nat 
        self.wbo = wbo_dict
        self.xyz = xyz
        self.atom_types = atom_types
        self.hessian: np.ndarray = hessian
        self.bo_matrix: np.ndarray = self.create_bomatrix_from_wbo()
        self.fortran_xyz: np.ndarray = self.angstrom2bohr(self.convert_xyz_to_fortranstyle(self.xyz))
        self.complete_graph: nx.Graph = self.create_graph_from_wbo()
        self.seperate_molecule_list = self.split_in_subgraphs()
        self.molecule_count = len(self.seperate_molecule_list)
        self.vander_matrix: np.ndarray = get_vander_matrix(self.atom_types)

    def convert_xyz_to_fortranstyle(self, xyz) -> np.array:
        '''returns column major version of xyz'''
        return np.asarray(xyz, dtype=float, order='F').T 
    
    def angstrom2bohr(self, val: float | np.ndarray):
        if type(val) == np.array:
            return np.divide(val, 1/1.8897259)
        return val/(1/1.8897259)

    def create_bomatrix_from_wbo(self) -> np.ndarray:
        bo_matrix = np.zeros((self.nat,self.nat))
        for atoms, val in self.wbo.items(): 
            i = atoms[0] 
            j = atoms[1]
            if bo_matrix[i-1,j-1] != 0:
                raise Exception('Double entry is present in wbo file.')
            bo_matrix[i-1,j-1] = val 
            bo_matrix[j-1,i-1] = val 
        return bo_matrix
    
    def scipy_optimizer_callback(self, xk: np.ndarray) -> None:
        """Callback function for scipy optimizer progress tracking."""
        cwd = os.getcwd()
        temp_wd = os.path.join(cwd, 'trj.xyz')

        coordinates = xk.reshape(-1, 3)
        elements = self.atom_types
        with open(os.path.join(temp_wd), 'a') as f:
            f.write(f"{len(coordinates)}\n")
            f.write("Debug optimization step\n")
            for e, c in zip(elements, coordinates):
                f.write(f"{e} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

    def create_graph_from_wbo(self) -> nx.Graph:
        """ 
        creates networkx Graph object with atoms as nodes and bond order values larger then 0 as edges.
        """
        G = nx.Graph()
        for node1, node2 in self.wbo:
            bo = self.wbo[(node1, node2)]
            if bo < 0.1:
                continue
            G.add_node(node1)
            G.add_node(node2)
            G.add_edge(node1, node2)
            nx.set_edge_attributes(G, {(node1, node2):{'bondorder': bo}})
        return G
    
    def split_in_subgraphs(self):
        """ 
        creates list of seperate networkx Graph objects, which correspond to subgraphs of self.complete_graph, meaning not connected by edges, thus seperate molecules in a structure.
        """
        G = self.complete_graph
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for graph in S:
            nodes = {node: {'id_in_subgraph': idx} for idx, node in enumerate(graph.nodes(), start=1)}
            nx.set_node_attributes(graph, nodes)
            # print(graph.nodes(data=True))
        return S



@dataclass
class Structure:
    '''
    Complete information about a given Structure.
    path: Path strings to all related files.
    ff: Force field data
    info: structural information like connectivities, atom count or geometrical structure.
    '''
    path: StructurePath 
    ff: ForceField
    info: StructuralInformation
