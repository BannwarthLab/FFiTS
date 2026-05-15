#!/bin/python

import logging
from dataclasses import dataclass
import networkx as nx
import numpy as np
import os
from collections.abc import Callable
from ffits.datatype.calculation_data import (
    CalculationData,
    CalculationOptions,
    PathData,
)
from ffits.forcefield.python_interface.ff_energy import (
    complete_gradient,
    complete_hessian,
    energy_ff,
)
from ffits.data.vanderwaals_radii import get_vander_matrix
from ffits.setup.structure_preparatation import randomize_coordinates
from ffits.utils.geometry import angstrom2bohr, convert_xyz_to_fortranstyle
from ffits.external.xtb import read_xtb_hessian, Xtb, read_wbo_file, readin_xyz
from ffits.io.file_writer import write_xyz_to_file
from ffits.datatype.forcefield_data import ForceField

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

    @staticmethod
    def from_config(
        cd: CalculationData,
        calcopt: CalculationOptions,
        pathdata: PathData,
        random_seed: int | None = None,
        bondorder_threshold: float = 0.0,
    ) -> "Structure":
        """_summary_

        Args:
            cd (CalculationData): _description_
            calcopt (CalculationOptions): _description_
            pathdata (PathData): _description_
            random_seed (int | None, optional): _description_. Defaults to None.

        Returns:
            Structure: _description_
        """
        xtbrunner = Xtb(
            chrg=cd.system.charge,
            mult=cd.system.multiplicity,
            xtb_path=cd.system.xtb_path,
            xtb_alpb_solvent=cd.system.xtb_alpb_solvent,
            xtb_input_name=cd.system.xtb_input_name,
        )

        if calcopt.geometry_optimization:
            new_xyz_filename, _, _, _, _ = xtbrunner.geomopt_with_topology_check(
                pathdata.xyz_filename, pathdata.xyz_filename, pathdata.wbo_filename
            )
            pathdata.xyz_filename = (
                new_xyz_filename  # I change that, so that old file is ignored
            )
            calcopt.wbo_calc = False

        nat, _, xyz, atom_types = readin_xyz(pathdata.xyz_filename)

        if random_seed is not None:
            xyz = randomize_coordinates(xyz, displacement=0.01, random_seed=random_seed)

        strucbuilder = StructureBuilder(
            xyz=xyz,
            xyz_filename=pathdata.xyz_filename,
            hessian_filename=pathdata.hessian_filename,
            wbo_filename=pathdata.wbo_filename,
            ff_filename=pathdata.ff_filename,
            bondorder_threshold=bondorder_threshold,
        )

        strucbuilder.nat(nat)
        strucbuilder.atom_types(atom_types)
        strucbuilder.path()

        if calcopt.hessian_calc:
            strucbuilder.hessian_from_xtb(xtbrunner)
        else:
            logger.info(
                f"Skipping Hessian calculation and reading in {pathdata.hessian_filename}."
            )
            strucbuilder.hessian_from_file()

        if calcopt.wbo_calc:
            strucbuilder.wbo_from_xtb(xtbrunner)
        else:
            logger.info(
                f"Skipping WBO calculation and reading in {pathdata.wbo_filename}."
            )
            strucbuilder.wbo_from_file()

        if calcopt.ff_parameterization:
            strucbuilder.ff_empty(
                energy_calculator=energy_ff,
                gradient_calculator=complete_gradient,
                hessian_calculator=complete_hessian,
            )
        else:
            logger.info(
                f"Skipping FF parameterization and reading in {pathdata.ff_filename}."
            )
            strucbuilder.ff_from_file(
                energy_calculator=energy_ff,
                gradient_calculator=complete_gradient,
                hessian_calculator=complete_hessian,
            )

        struc = strucbuilder.build()
        return struc


class StructureBuilder:
    def __init__(
        self,
        xyz: np.ndarray,
        xyz_filename: str = "struc.xyz",
        wbo_filename: str = "wbo",
        hessian_filename: str = "hess",
        ff_filename: str = "ff.csv",
        bondorder_threshold: float = 0.0,
    ):
        self._xyz = xyz  # (np.ndarray): coordinates in Angström in shape (nat, 3)
        self._xyz_filename = xyz_filename
        self._wbo_filename = wbo_filename
        self._hessian_filename = hessian_filename
        self._ff_filename = ff_filename
        self._bondorder_threshold = bondorder_threshold
        self._nat = None
        self._hessian = None
        self._wbo_dict = None
        self._atom_types = None
        self._ff: ForceField = None
        self._path: StructurePath = self.path()

    def _write_xyz_if_needed(self):
        if self._xyz_filename is not None and not os.path.exists(self._xyz_filename):
            logger.warning(
                f"XYZ file {self._xyz_filename} does not exist. It will be created with the provided xyz data."
            )
            write_xyz_to_file(self._xyz, self._xyz_filename)

    def nat(self, nat: int) -> "StructureBuilder":
        self._nat = nat
        return self

    def atom_types(self, atom_types: np.ndarray) -> "StructureBuilder":
        self._atom_types = atom_types
        return self

    # ---- wbo ----
    def wbo_from_xtb(self, xtbrunner: Xtb) -> "StructureBuilder":
        self._write_xyz_if_needed()
        self._wbo_dict = xtbrunner.wbocalc(self._xyz_filename, self._wbo_filename)
        return self

    def wbo_from_file(self) -> "StructureBuilder":
        self._wbo_dict = read_wbo_file(self._wbo_filename)
        return self

    def wbo_from_dict(self, wbo_dict: dict) -> "StructureBuilder":
        self._wbo_dict = wbo_dict
        return self

    # ---- hessian ----
    def hessian_from_xtb(self, xtbrunner: Xtb) -> "StructureBuilder":
        self._write_xyz_if_needed()
        self._hessian = xtbrunner.hesscalc(self._xyz_filename, self._hessian_filename)
        return self

    def hessian_from_file(self, format: str = "xtb") -> "StructureBuilder":
        if format == "xtb":
            self._hessian = read_xtb_hessian(self._hessian_filename)
        else:
            raise ValueError(f"Unsupported hessian file format: {format}")
        return self

    def hessian_from_array(self, hessian_array: np.ndarray) -> "StructureBuilder":
        self._hessian = hessian_array
        return self

    # ---- force field ----
    def ff_from_file(
        self,
        energy_calculator: Callable,
        gradient_calculator: Callable,
        hessian_calculator: Callable,
    ) -> "StructureBuilder":
        self._ff = ForceField(
            self._nat,
            self._ff_filename,
            readff=True,
            energy_calculator=energy_calculator,
            gradient_calculator=gradient_calculator,
            hessian_calculator=hessian_calculator,
        )
        return self

    def ff_empty(
        self,
        energy_calculator: Callable,
        gradient_calculator: Callable,
        hessian_calculator: Callable,
    ) -> "StructureBuilder":
        self._ff = ForceField(
            self._nat,
            self._ff_filename,
            readff=False,
            energy_calculator=energy_calculator,
            gradient_calculator=gradient_calculator,
            hessian_calculator=hessian_calculator,
        )
        return self

    # ---- path data ----
    def path(self) -> "StructureBuilder":
        self._path = StructurePath(
            xyz_filename=self._xyz_filename,
            hessian_filename=self._hessian_filename,
            wbo_filename=self._wbo_filename,
            ff_filename=self._ff_filename,
        )
        return self

    def build(self) -> Structure:
        if self._nat is None:
            raise ValueError("Number of atoms (nat) must be specified.")
        if self._hessian is None:
            raise ValueError("Hessian must be defined.")
        if self._wbo_dict is None:
            raise ValueError("WBO dictionary must be defined.")
        if self._atom_types is None:
            raise ValueError("Atom types must be defined.")
        if self._ff is None:
            raise ValueError("Force field must be defined.")

        return Structure(
            path=self._path,
            ff=self._ff,
            info=StructuralInformation(
                nat=self._nat,
                xyz=self._xyz,
                wbo_dict=self._wbo_dict,
                atom_types=self._atom_types,
                hessian=self._hessian,
                bo_threshold=self._bondorder_threshold,
            ),
        )
