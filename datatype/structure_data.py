#!/bin/python

from dataclasses import dataclass
from typing import List
import networkx as nx
import numpy as np
import json

@dataclass
class StructurePath:
    """
    Paths/Filenames and ID for a structure
    """
    id: int
    xyz_filename: str 
    hess_filename: str 
    wbo_filename: str 
    ff_filename: str

    @staticmethod
    def from_json(id: int, str):
        return StructurePath(id=id,
                             xyz_filename = str["xyz"],
                             wbo_filename = str["wbo"],
                             hess_filename = str["hessian"],
                             ff_filename = str["forcefield"])   


class ForceField:
    """ 
    The force field (FF) is defined through FF parameters (matrices starting with c_) as well 
    as the reference values (bondlengths, angles, dihedral angles and repulsion references) 
    and corresponging atom combinations (between which two atoms is the bond). 
    """
    def __init__(self, nat, ff_filename, readff=True):
        self.filename = filename
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


    def write_force_field(self, ff: ForceField, filename):
        print('WRITES TO', filename)
        with open(filename, 'w') as file:
            if len(ff.bond_list) > 0:
                file.write(f"$bonds, {len(ff.bond_list)}\n")
                for (atom1, atom2), bondlength in zip(ff.bond_list, ff.bondlengths):
                    param = ff.c_bond[atom1-1, atom2-1]
                    file.write(f"{atom1}, {atom2}, {param}, {bondlength}\n")

            if len(ff.angle_list) > 0:
                file.write(f"$angles, {len(ff.angle_list)}\n")
                for (atom1, atom2, atom3), angle in zip(ff.angle_list, ff.angles):
                    param = ff.c_angle[atom1-1, atom2-1, atom3-1]
                    file.write(f"{atom1}, {atom2}, {atom3}, {param}, {angle}\n")

            if len(ff.dihedral_list) > 0:
                file.write(f"$dihedrals, {len(ff.dihedral_list)}\n")
                for (atom1, atom2, atom3, atom4), dihedral_angle in zip(ff.dihedral_list, ff.dihedrals):
                    param = ff.c_dihedral[atom1-1, atom2-1, atom3-1, atom4-1]
                    file.write(f"{atom1}, {atom2}, {atom3}, {atom4}, {param}, {dihedral_angle}\n")

            if len(ff.lj_list) > 0:
                file.write(f"$lj-terms, {len(ff.lj_list)}\n")
                for (atom1, atom2), sigma in zip(ff.lj_list, ff.sigmas):
                    param = ff.c_lj[atom1-1, atom2-1]
                    file.write(f"{atom1}, {atom2}, {param}, {sigma}\n")



class StructuralInformation:
    def __init__(self, nat, energy, xyz, wbo_list):
        self.nat = nat 
        self.energy = energy 
        self.wbo = wbo_list
        self.xyz = xyz
        self.complete_graph = self.create_graph_from_wbo()
        self.seperate_molecule_list = self.split_in_subgraphs()
        self.molecule_count = len(self.seperate_molecule_list)

    def create_graph_from_wbo(self) -> nx.Graph:
        G = nx.Graph()
        for node1, node2 in self.wbo:
            bo = self.wbo[(node1, node2)]
            if bo < 0.2:
                continue
            G.add_node(node1)
            G.add_node(node2)
            G.add_edge(node1, node2)
            nx.set_edge_attributes(G, {(node1, node2):{'bondorder': bo}})
        return G
    
    def split_in_subgraphs(self):
        G = self.complete_graph
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for graph in S:
            nodes = {node: {'id_in_subgraph': idx} for idx, node in enumerate(graph.nodes(), start=1)}
            nx.set_node_attributes(graph, nodes)
            # print(graph.nodes(data=True))
        return S



@dataclass
class Structure:
    path: StructurePath 
    ff: ForceField
    info: StructuralInformation
