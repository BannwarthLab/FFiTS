#!/bin/python

from dataclasses import dataclass
from typing import List
import networkx as nx
import numpy as np
import os


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

def atom_symbol_to_number(symbol: str) -> int:
    """Convert an element symbol (e.g. 'C') to its atomic number (e.g. 6)."""
    try:
        return PERIODIC_TABLE[symbol.capitalize()]
    except KeyError:
        raise ValueError(f"Unknown atom symbol: {symbol}")
    
def get_vander_matrix(nat, at, vander_values=VANDER_VALUES, factor=1.0):
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
    atom_numbers = [PERIODIC_TABLE[s] for s in at]
    radii = vander_values[np.array(atom_numbers) - 1] * factor
    # Broadcasting sum of pairwise radii
    return radii[:, None] + radii[None, :]

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
    id: int
    xyz_filename: str 
    hess_filename: str 
    wbo_filename: str 
    ff_filename: str

    @staticmethod
    def from_json(id: int, str):
        """ 
        returns StructurePath filled with values from json file part 'str'.
        """
        return StructurePath(id=id,
                             xyz_filename = str["xyz"],
                             wbo_filename = str["wbo"],
                             hess_filename = str["hessian"],
                             ff_filename = str["forcefield"])   


class ForceField:
    """ 
    FF definition through FF parameters (np arrays starting with c_), reference values (bondlenghts, angles, etc) and corresponding atom numbers, which construct the bond / angle / dihedral angle / lj term. 
    """
    def __init__(self, nat: int, ff_filename: str, readff: bool = True):
        self.nat = nat
        self.ff_filename = ff_filename
        self.c_bond = {} 
        self.c_angle = {} 
        self.c_dihedral = {}  # matrix can get to large with N^4, so instead a dict will be used 
        self.c_lj = {} 
        self.bond_list = []
        self.angle_list = []
        self.dihedral_list = []
        self.lj_list = []
        self.bondlengths = []
        self.angles = []
        self.dihedrals = []
        self.sigmas = []
        if readff:
            self.readin_forcefield(ff_filename)
            # if not self.correct_dimensions():
            #     raise Exception('Dimensions of reference values and given dimensions do not fit.')

    def readin_forcefield(self, ff_filename: str):
        """ 
        reads force field definition from path 'ff_filename'. Works only for specific file structure:
            $bonds, 'number_of_bonds'
            atom1, atom2, ff_parameter_for_bond, bondlength
            ...
            $angles, 'number_of_angles'
            atom1, atom2, atom3, ff_parameter_for_angle, angle
            ...
            $dihedrals, 'number_of_dihedral_angles'
            atom1, atom2, atom3, atom4, ff_parameter_for_dihedral_angle, dihedral_angle
            ...
            $lj-terms, 'number_of_lj_terms'
            atom1, atom3, ff_parameter_for_lj_term, sigma
            ...
        """
        if not os.path.exists(ff_filename):
            raise Exception(FileNotFoundError(ff_filename))
        with open(ff_filename, 'r') as file:
            section = None
            for line in file:
                line = line.strip()
                if line.startswith("$"):
                    section = line.split(',')[0].strip(',')[1:]
                elif section == "bonds":
                    atom1, atom2, param, bondlength = map(float, line.split(','))
                    self.bond_list.append([int(atom1), int(atom2)])
                    self.c_bond[(int(atom1), int(atom2))] = param
                    self.bondlengths.append(bondlength)
                elif section == "angles":
                    atom1, atom2, atom3, param, angle = map(float, line.split(','))
                    self.angle_list.append([int(atom1), int(atom2), int(atom3)])
                    self.c_angle[(int(atom1), int(atom2), int(atom3))] = param
                    self.angles.append(angle)
                elif section == "dihedrals":
                    atom1, atom2, atom3, atom4, param, dihedral_angle = map(float, line.split(','))
                    self.dihedral_list.append([int(atom1), int(atom2), int(atom3), int(atom4)])
                    self.c_dihedral[(int(atom1), int(atom2), int(atom3), int(atom4))] = param
                    self.dihedrals.append(dihedral_angle)
                elif section == "lj-terms":
                    atom1, atom2, param, sigma = map(float, line.split(','))
                    self.lj_list.append([int(atom1), int(atom2)])
                    self.c_lj[(int(atom1), int(atom2))] = param
                    self.sigmas.append(sigma)
        if section == None:
            raise Exception('File seems to be empty or not contain the section markers.')


    def write_force_field(self, filename):
        """ 
        writes force field definition to path 'ff_filename'. Works only for specific file structure:
            $bonds, 'number_of_bonds'
            atom1, atom2, ff_parameter_for_bond, bondlength
            ...
            $angles, 'number_of_angles'
            atom1, atom2, atom3, ff_parameter_for_angle, angle
            ...
            $dihedrals, 'number_of_dihedral_angles'
            atom1, atom2, atom3, atom4, ff_parameter_for_dihedral_angle, dihedral_angle
            ...
            $lj-terms, 'number_of_lj_terms'
            atom1, atom3, ff_parameter_for_lj_term, sigma
            ...
        """
        with open(filename, 'w') as file:
            if len(self.bond_list) > 0: 
                file.write(f"$bonds, {len(self.bond_list)}\n")
                for (atom1, atom2), bondlength in zip(self.bond_list, self.bondlengths):
                    param = self.c_bond[atom1-1, atom2-1]
                    file.write(f"{atom1}, {atom2}, {param}, {bondlength}\n")
            else:
                file.write(f"$bonds, {len(self.bond_list)}\n")

            if len(self.angle_list) > 0:
                file.write(f"$angles, {len(self.angle_list)}\n")
                for (atom1, atom2, atom3), angle in zip(self.angle_list, self.angles):
                    param = self.c_angle[atom1-1, atom2-1, atom3-1]
                    file.write(f"{atom1}, {atom2}, {atom3}, {param}, {angle}\n")
            else:
                file.write(f"$angles, {len(self.angle_list)}\n")

            if len(self.dihedral_list) > 0:
                file.write(f"$dihedrals, {len(self.dihedral_list)}\n")
                for (atom1, atom2, atom3, atom4), dihedral_angle in zip(self.dihedral_list, self.dihedrals):
                    param = self.c_dihedral[atom1-1, atom2-1, atom3-1, atom4-1]
                    file.write(f"{atom1}, {atom2}, {atom3}, {atom4}, {param}, {dihedral_angle}\n")
            else:
                file.write(f"$dihedrals, {len(self.dihedral_list)}\n")

            if len(self.lj_list) > 0:
                file.write(f"$lj-terms, {len(self.lj_list)}\n")
                for (atom1, atom2), sigma in zip(self.lj_list, self.sigmas):
                    param = self.c_lj[atom1-1, atom2-1]
                    file.write(f"{atom1}, {atom2}, {param}, {sigma}\n")
            else:
                file.write(f"$lj-terms, {len(self.lj_list)}\n")



class StructuralInformation:
    """ 
    Information on the given structure.
        nat: number of atoms
        energy: energy calculated by unspecified method
        wbo: Wilberg Bond Order as directory of bond pairs and corresponding WBO values
        complete_graph: networkx Graph object with atoms as nodes and bond order larger then 0 as edges.
        seperate_molecule_list: List of subgraphs not connectred by edges in complete_graph.
        molecule_count: number of seperate molecules in structure.
    """
    def __init__(self, nat: int, xyz: np.array, wbo_dict: dict, atom_types: np.array, energy: float = np.NaN):
        self.nat = nat 
        self.energy = energy 
        self.wbo = wbo_dict
        self.xyz = xyz
        self.atom_types = atom_types
        self.bo_matrix: np.array = self.create_bomatrix_from_wbo()
        self.fortran_xyz: np.array = self.angstrom2bohr(self.convert_xyz_to_fortranstyle())
        self.complete_graph: nx.Graph = self.create_graph_from_wbo()
        self.seperate_molecule_list = self.split_in_subgraphs()
        self.molecule_count = len(self.seperate_molecule_list)
        self.vander_matrix: np.array = get_vander_matrix(self.nat, self.atom_types)

    def convert_xyz_to_fortranstyle(self) -> np.array:
        x = []
        y = []
        z = []
        for i in range(self.nat):
            x.append(self.xyz[i][0])
            y.append(self.xyz[i][1])
            z.append(self.xyz[i][2])
        return np.array([x,y,z],order='F')
    
    def angstrom2bohr(self, val):
        if type(val) == np.array:
            return np.divide(val, 1/1.8897259)
        return val/(1/1.8897259)

    def create_bomatrix_from_wbo(self) -> np.array:
        bo_matrix = np.zeros((self.nat,self.nat))
        for atoms, val in self.wbo.items(): 
            i = atoms[0] 
            j = atoms[1]
            if bo_matrix[i-1,j-1] != 0:
                raise Exception('Double entry is present in wbo file.')
            bo_matrix[i-1,j-1] = val 
            bo_matrix[j-1,i-1] = val 
        return bo_matrix

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
