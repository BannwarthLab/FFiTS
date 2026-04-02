#!/bin/python

import logging
from dataclasses import dataclass
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
from collections.abc import Callable
from ffits.data.vanderwaals_radii import get_vander_matrix
from ffits.utils.geometry import angstrom2bohr, convert_xyz_to_fortranstyle

logger = logging.getLogger(__name__)


@dataclass
class StructurePath:
    """
    Paths/Filenames for a given structure. This class mirrors PathData.
    """

    xyz_filename: str
    hessian_filename: str
    wbo_filename: str
    ff_filename: str


class ForceField:
    """
    FF definition through FF parameters (np arrays starting with c_), reference values (bondlenghts, angles, etc) and corresponding atom numbers, which construct the bond / angle / dihedral angle / lj term.
    """

    def __init__(
        self,
        nat: int,
        ff_filename: str,
        readff: bool = False,
        energy_calculator: Callable = None,
        gradient_calculator: Callable = None,
        hessian_calculator: Callable = None,
    ):
        self.nat = nat
        self.ff_filename = ff_filename
        self.columns = ["type", "atoms", "parameter", "reference_value"]
        self.bonds: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.angles: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.dihedrals: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.repulsive: pd.DataFrame = pd.DataFrame(columns=self.columns)
        if readff:
            self.readin(ff_filename)
        self.energy_calculator = energy_calculator
        self.gradient_calculator = gradient_calculator
        self.hessian_calculator = hessian_calculator
        self.start_from_reactant: bool = True  # only relevant for TS FF

    def get_energy(self, xyz_displaced: np.ndarray) -> float:
        """
        Calculates energy with defined self.energy_calculator. Handles transfer in correct xyz format for said calculator.

        Args:
            xyz_displaced (np.ndarray): Displaced xyz. Can be in shape (nat, 3) or (nat*3,).

        Returns:
            enerty (float): The calculated energy.
        """
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.energy_calculator(
                angstrom2bohr(convert_xyz_to_fortranstyle(xyz_displaced)),
                self,
            )
        if len(xyz_displaced) == self.nat * 3:
            return self.energy_calculator(
                angstrom2bohr(xyz_displaced.reshape(self.nat, 3).T), self
            )
        return self.energy_calculator(xyz_displaced, self)

    def get_gradient(self, xyz_displaced: np.ndarray) -> np.ndarray:
        """
        Calculates gradient with defined self.gradient_calculator. Handles transfer in correct xyz format for said calculator.

        Args:
            xyz_displaced (np.ndarray): Displaced xyz. Can be in shape (nat, 3) or (nat*3).

        Returns:
            gradient (np.ndarray, shape=(nat*3)): The calculated gradient.
        """
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.gradient_calculator(
                angstrom2bohr(convert_xyz_to_fortranstyle(xyz_displaced)),
                self,
            )
        if len(xyz_displaced) == self.nat * 3:
            return self.gradient_calculator(
                angstrom2bohr(xyz_displaced.reshape(self.nat, 3).T), self
            )
        return self.gradient_calculator(xyz_displaced, self)

    def get_hessian(self, xyz_displaced: np.ndarray) -> np.ndarray:
        """
        Calculates hessian with defined self.hessian_calculator. Handles transfer in correct xyz format for said calculator.

        Args:
            xyz_displaced (np.ndarray): Displaced xyz. Can be in shape (nat, 3) or (nat*3).

        Returns:
            hessian (np.ndarray, shape=(nat*3, nat*3)): The calculated hessian.
        """
        if np.shape(xyz_displaced) == (self.nat, 3):
            return self.hessian_calculator(
                angstrom2bohr(convert_xyz_to_fortranstyle(xyz_displaced)),
                self,
            )
        if len(xyz_displaced) == self.nat * 3:
            return self.hessian_calculator(
                angstrom2bohr(xyz_displaced.reshape(self.nat, 3).T), self
            )
        return self.hessian_calculator(xyz_displaced, self)

    def write(self):
        """
        Writes out all FF data in self.ff_filename in the csv format.
        """

        def format_atoms(t):
            return "[" + " ".join(map(str, t)) + "]"

        df_combined = pd.concat(
            [self.bonds, self.angles, self.dihedrals, self.repulsive], ignore_index=True
        )
        df_combined = df_combined.copy()
        df_combined["atoms"] = df_combined["atoms"].apply(format_atoms)
        df_combined.to_csv(self.ff_filename)
        logger.info(f"FF information written to {self.ff_filename}.")

    def readin(self, filename: str):
        """
        Reads a force field csv file and fills defines the ForceField object accordingly.

        Args:
            filename (str): path to the force field csv file. The file should have columns atoms, parameter, reference_value and type, where type can be bonds, angles, dihedrals or repulsive. The atoms column should contain a tuple of 0-based atom indices, which are involved in the corresponding term. For example, for a bond between atom 0 and 1, the atoms column should contain (0,1).
        """
        try:

            def parse_atoms(x):
                """Parse 'atoms' column into a tuple of integers."""
                if pd.isna(x):
                    return tuple()
                if isinstance(x, (list, tuple)):
                    # Already iterable — ensure tuple of ints
                    return tuple(int(i) for i in x)
                # Remove brackets and commas, split on whitespace
                x = re.sub(r"[\[\],]", " ", str(x))
                tokens = x.split()
                return tuple(int(tok) for tok in tokens)

            df = pd.read_csv(
                filename,
                converters={
                    "atoms": parse_atoms,
                    "parameter": float,
                    "reference_value": float,
                },
                index_col=0,
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"Force field file '{filename}' not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Force field file '{filename}' is empty or malformed.")
        except Exception as e:
            raise ValueError(f"Error while reading '{filename}': {e}")

        self.bonds = df[df["type"] == "bonds"].copy()
        self.angles = df[df["type"] == "angles"].copy()
        self.dihedrals = df[df["type"] == "dihedrals"].copy()
        self.repulsive = df[df["type"] == "repulsive"].copy()


class StructuralInformation:
    """
    Structural information on the given structure.
    """

    def __init__(
        self,
        nat: int,
        xyz: np.ndarray,
        wbo_dict: dict,
        atom_types: np.ndarray,
        hessian: np.ndarray = None,
        bo_threshold: float = 0.0,
    ):
        """
        Initialization of StructuralInformation object

        Args:
            nat (int): number of atoms
            xyz (np.ndarray): coordinates in Angström in shape (nat, 3)
            wbo_dict (dict): Dictionary (atom1, atom2: wbo_value) with 0-based atom indices
            atom_types (np.ndarray): array of element symbols for each atom
            hessian (np.ndarray, optional): Hessian of structure. Defaults to None.
            bo_threshold (float, optional): Threshold for bond order to consider a bond as existing. Defaults to 0.0.

        Attributes:
            bo_matrix (np.ndarray): bond order matrix derived from wbo_dict
            fortran_xyz (np.ndarray): coordinates in fortran style (shape (3, nat)) and bohr units
            complete_graph (nx.Graph): graph with atoms as nodes and bonds as edges, where bond order is an edge attribute
            seperate_molecule_list (list of nx.Graph): list of seperate graphs for each molecule in the structure
            molecule_count (int): number of seperate molecules in the structure
            vander_matrix (np.ndarray): van der Waals matrix of shape (nat, nat)
        """
        self.nat = nat
        self.wbo = wbo_dict
        self.xyz = xyz
        self.atom_types = atom_types
        self.hessian: np.ndarray = hessian
        self.bo_threshold = bo_threshold
        if self.wbo != {}:
            self.bo_matrix: np.ndarray = self._create_bomatrix_from_wbo()
            self.fortran_xyz: np.ndarray = angstrom2bohr(
                convert_xyz_to_fortranstyle(self.xyz)
            )
            self.complete_graph: nx.Graph = self._create_graph_from_wbo()
            self.seperate_molecule_list = self._split_in_subgraphs()
            self.molecule_count = len(self.seperate_molecule_list)
            logger.info(
                f"This structure has {self.molecule_count} seperate molecule(s) based on the WBO data and the defined bond order threshold of {self.bo_threshold}."
            )
        self.vander_matrix: np.ndarray = get_vander_matrix(self.atom_types)

    def _create_bomatrix_from_wbo(self) -> np.ndarray:
        """
        Creates bond order matrix from wbo dictionary. The wbo dictionary has keys as tuples of atom indices (0-based) and values as the corresponding WBOs.

        Raises:
            Exception: Double entry in wbo dictionary (i.e., if the same bond is defined more than once).

        Returns:
            np.ndarray: bond order matrix of shape (nat, nat) where element (i, j) is the WBO between atoms i and j, and 0 if no bond is defined.
        """
        bo_matrix = np.zeros((self.nat, self.nat))
        for atoms, val in self.wbo.items():
            i = atoms[0]
            j = atoms[1]
            if bo_matrix[i, j] != 0:
                raise Exception("Double entry is present in wbo file.")
            bo_matrix[i, j] = val
            bo_matrix[j, i] = val
        logger.debug(f"Bond order matrix created from WBO dictionary:\n{bo_matrix}")
        return bo_matrix

    def scipy_optimizer_callback(self, xk: np.ndarray) -> None:
        """Callback function for scipy optimizer progress tracking."""
        cwd = os.getcwd()
        temp_wd = os.path.join(cwd, "trj.xyz")

        coordinates = xk.reshape(-1, 3)
        elements = self.atom_types
        with open(os.path.join(temp_wd), "a") as f:
            f.write(f"{len(coordinates)}\n")
            f.write("Debug optimization step\n")
            for e, c in zip(elements, coordinates):
                f.write(f"{e} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

    def _create_graph_from_wbo(self) -> nx.Graph:
        """
        creates networkx Graph object with atoms as nodes and bond order values larger then 0 as edges.
        """
        G = nx.Graph()
        for node1, node2 in self.wbo:
            bo = self.wbo[(node1, node2)]
            if bo < self.bo_threshold:
                continue
            G.add_node(node1)
            G.add_node(node2)
            G.add_edge(node1, node2)
            nx.set_edge_attributes(G, {(node1, node2): {"bondorder": bo}})
        logger.debug(
            f"Graph created from WBO data with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )
        return G

    def _split_in_subgraphs(self):
        """
        Creates list of seperate networkx Graph objects, which correspond to subgraphs of self.complete_graph, meaning not connected by edges, thus seperate molecules in a structure.
        """
        G = self.complete_graph
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for graph in S:
            nodes = {
                node: {"id_in_subgraph": idx}
                for idx, node in enumerate(graph.nodes(), start=1)
            }
            nx.set_node_attributes(graph, nodes)
            # print(graph.nodes(data=True))
        logger.debug(
            f"Structure split into {len(S)} subgraph(s) based on connectivity."
        )
        return S


@dataclass
class Structure:
    """
    Complete information about a given Structure.

    Variables:
        path (StructurePath): Path strings to all related files.
        ff (ForceField): Force field data
        info (StructuralInformation): structural information like connectivities, atom count or geometrical structure.
    """

    path: StructurePath
    ff: ForceField
    info: StructuralInformation
