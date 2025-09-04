#!/bin/python

from dataclasses import dataclass
from typing import List
import networkx as nx
import numpy as np
import json


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
    def __init__(self, nat, ff_filename, readff=True):
        self.nat = nat
        self.ff_filename = ff_filename
        self.c_bond = np.zeros((nat,nat))
        self.c_angle = np.zeros((nat,nat,nat))
        self.c_dihedral = np.zeros((nat,nat,nat,nat))
        self.c_lj = np.zeros((nat,nat))
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
        with open(ff_filename, 'r') as file:
            section = None
            for line in file:
                line = line.strip()
                if line.startswith("$"):
                    section = line.split(',')[0].strip(',')[1:]
                elif section == "bonds":
                    atom1, atom2, param, bondlength = map(float, line.split(','))
                    self.bond_list.append([int(atom1), int(atom2)])
                    self.c_bond[int(atom1)-1, int(atom2)-1] = param
                    self.bondlengths.append(bondlength)
                elif section == "angles":
                    atom1, atom2, atom3, param, angle = map(float, line.split(','))
                    self.angle_list.append([int(atom1), int(atom2), int(atom3)])
                    self.c_angle[int(atom1)-1, int(atom2)-1, int(atom3)-1] = param
                    self.angles.append(angle)
                elif section == "dihedrals":
                    atom1, atom2, atom3, atom4, param, dihedral_angle = map(float, line.split(','))
                    self.dihedral_list.append([int(atom1), int(atom2), int(atom3), int(atom4)])
                    self.c_dihedral[int(atom1)-1, int(atom2)-1, int(atom3)-1, int(atom4)-1] = param
                    self.dihedrals.append(dihedral_angle)
                elif section == "lj-terms":
                    atom1, atom2, param, sigma = map(float, line.split(','))
                    self.lj_list.append([int(atom1), int(atom2)])
                    self.c_lj[int(atom1)-1, int(atom2)-1] = param
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
    def __init__(self, nat, energy, xyz, wbo_list):
        self.nat = nat 
        self.energy = energy 
        self.wbo = wbo_list
        self.xyz = xyz
        self.fortran_xyz = self.convert_xyz_to_fortranstyle()
        self.complete_graph = self.create_graph_from_wbo()
        self.seperate_molecule_list = self.split_in_subgraphs()
        self.molecule_count = len(self.seperate_molecule_list)

    def convert_xyz_to_fortranstyle(self) -> np.array:
        x = []
        y = []
        z = []
        for i in range(self.nat):
            x.append(self.xyz[i][0])
            y.append(self.xyz[i][1])
            z.append(self.xyz[i][2])
        return np.array([x,y,z],order='F')

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
