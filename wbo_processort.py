#!/bin/python

import networkx as nx
import subprocess
import os.path

class Structure:
    def __init__(self, struc_path, wbo_path):
        self.struc_path = struc_path
        self.wbo_path = wbo_path
        self.wbo_dict = self.parse_wbo_file()
        self.G = self.create_graph_from_wbo()
        self.sep_molecules = self.split_in_subgraphs()
        self.amount_of_molecules = len(self.sep_molecules)
        self.xyz = self.readin_xyz()
        self.nat = int(self.xyz[0].strip())
        print('Graph has', self.amount_of_molecules, 'Subgraphs')

    def parse_wbo_file(self):
        """Reads WBO data from a file and parses it into a dictionary of bonds and their WBO values."""
        wbo_dict = {}
        with open(self.wbo_path, 'r') as file:
            for line in file:
                if line.strip():  # skip empty lines
                    atom1, atom2, wbo = line.split()
                    bond = tuple(sorted((int(atom1), int(atom2))))
                    wbo_dict[bond] = float(wbo)
        return wbo_dict
    
    def create_graph_from_wbo(self) -> nx.Graph:
        G = nx.Graph()
        with open(self.wbo_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            split = line.split()
            if float(split[2]) < 0.2:
                continue
            node1 = int(split[0])
            node2 = int(split[1])
            G.add_node(node1)
            G.add_node(node2)
            G.add_edge(node1, node2)
            nx.set_edge_attributes(G, {(node1, node2):{'bondorder': float(split[2].strip())}})
        return G
    
    def split_in_subgraphs(self):
        S = [self.G.subgraph(c).copy() for c in nx.connected_components(self.G)]
        for graph in S:
            nodes = {node: {'id_in_subgraph': idx} for idx, node in enumerate(graph.nodes(), start=1)}
            nx.set_node_attributes(graph, nodes)
            # print(graph.nodes(data=True))

        return S

    def readin_xyz(self):
        with open(self.struc_path, 'r') as file:
            lines = file.readlines()
        return lines
    
    def relate_index(self, ind_G: int) -> int:
        return ind_G + 1

    def subgraph_to_xyz(self,graph, new_filename: str) -> None:
        xyz = self.readin_xyz()
        lines = [str(graph.number_of_nodes()) + '\n', '\n']
        for node in graph.nodes:
            lines.append(xyz[self.relate_index(node)])
        with open(new_filename, 'w') as file:
            for line in lines:
                file.write(line)

    def write_all_molecules_to_file(self, new_xyzname='coord'):
        id = 0
        print(self.struc_path, "is seperated into subgraphs.")
        for graph in self.sep_molecules: 
            id = id + 1
            self.subgraph_to_xyz(graph,new_xyzname + str(id) + '.xyz')
            print("Subgraph", graph, "printed to", new_xyzname + str(id) + '.xyz')


class Reaction:
    def __init__(self, reac: Structure, prod: Structure):
        self.reac = reac
        self.prod = prod
        self.unique_bonds = self.compare_wbos(reac.wbo_dict, prod.wbo_dict)

    def compare_wbos(self, wbo_dict1, wbo_dict2, threshold=0.8):
        """Compares two WBO data files and returns unique atom indices, including those with significant WBO differences."""
        all_bonds = set(wbo_dict1.keys()).union(set(wbo_dict2.keys()))
        
        unique_bonds = set()
        for bond in all_bonds:
            wbo1 = wbo_dict1.get(bond, 0)
            wbo2 = wbo_dict2.get(bond, 0)
            if bond not in wbo_dict1 or bond not in wbo_dict2 or abs(wbo1 - wbo2) > threshold:
                unique_bonds.add(bond)

        return list(unique_bonds)
        unique_atoms = set()
        for bond in unique_bonds:
            unique_atoms.update(bond)

        sorted_unique_atoms = sorted(unique_atoms)
        return sorted_unique_atoms




    # def create_line_of_xyz(atom_id: int, ):
    #     atom = "{:<12}".format(str(atom_int))
    #     node2 = "{:>12}".format(str(node2))
    #     bondorder = "{:0<17}".format(str(round(bondorder, 17)))
    #     return node1 + node2 + '  ' + bondorder + '\n'






# def main():
#     crestcaller = Crest()
#     xtbcaller = Xtb()
#     rmsds = []                                                                                                          
#     print_all_subgraphs('wbo', 'crestopt', 'opt')
#     print_all_subgraphs('wbo1', 'struc1', 'reac')
#     opts =  ['opt1.xyz', 'opt2.xyz', 'opt3.xyz']
#     reacs = ['reac1.xyz', 'reac2.xyz', 'reac3.xyz']
#     for opt in opts:
#         for reac in reacs:
#             if os.path.exists(opt) and os.path.exists(reac):
#                 if len(readin_xyz(opt)) == len(readin_xyz(reac)):
#                     rmsd = crestcaller.get_rmsd(opt, reac)
#                     print('rmsd between', opt, 'and', reac, 'is', rmsd)
#                     rmsds.append(rmsd)
#     if len(rmsds) == 0:
#         rmsd = crestcaller.get_rmsd('crestopt.xyz', 'struc1.xyz')
#         print('rmsd between', 'crestopt.xyz', 'and', 'struc1.xyz', 'is', rmsd)
#     #('wbo2', 'struc2', 'reac')


# if __name__ == "__main__":
#     main()
