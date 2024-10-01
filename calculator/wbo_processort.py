#!/bin/python

import networkx as nx
import datatype.structure_data

# -------
def get_unique_bonds(wbo_dict1, wbo_dict2, threshold=0.8):
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
