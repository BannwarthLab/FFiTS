#!/bin/python

import src.datatype.structure_data 

def print_box(name: str, width=80):
    print('┏' +    "━"*width       + "┓")
    print('┃' + name.center(width) + '┃')
    print('┗' +    "━"*width       + "┛")



def write_all_molecules_to_file(seperate_molecule_list, new_filename_prefix):
    def subgraph_to_xyz(xyz, graph, new_filename: str) -> None:
        def relate_index(ind_G: int) -> int:
            return ind_G + 1
        lines = [str(graph.number_of_nodes()) + '\n', '\n']
        for node in graph.nodes:
            lines.append(xyz[relate_index(node)])
        with open(new_filename, 'w') as file:
            for line in lines:
                file.write(line)
    id = 0
    for graph in seperate_molecule_list: 
        id = id + 1
        subgraph_to_xyz(graph, new_filename_prefix + str(id) + '.xyz')
        print("Subgraph", graph, "printed to", new_filename_prefix + str(id) + '.xyz')


# def write_force_field(ff: ForceField, filename):
#     print('WRITES TO', filename)
#     with open(filename, 'w') as file:
#         if len(ff.bond_list) > 0:
#             file.write(f"$bonds, {len(ff.bond_list)}\n")
#             for (atom1, atom2), bondlength in zip(ff.bond_list, ff.bondlengths):
#                 param = ff.c_bond[atom1-1, atom2-1]
#                 file.write(f"{atom1}, {atom2}, {param}, {bondlength}\n")

#         if len(ff.angle_list) > 0:
#             file.write(f"$angles, {len(ff.angle_list)}\n")
#             for (atom1, atom2, atom3), angle in zip(ff.angle_list, ff.angles):
#                 param = ff.c_angle[atom1-1, atom2-1, atom3-1]
#                 file.write(f"{atom1}, {atom2}, {atom3}, {param}, {angle}\n")

#         if len(ff.dihedral_list) > 0:
#             file.write(f"$dihedrals, {len(ff.dihedral_list)}\n")
#             for (atom1, atom2, atom3, atom4), dihedral_angle in zip(ff.dihedral_list, ff.dihedrals):
#                 param = ff.c_dihedral[atom1-1, atom2-1, atom3-1, atom4-1]
#                 file.write(f"{atom1}, {atom2}, {atom3}, {atom4}, {param}, {dihedral_angle}\n")

#         if len(ff.lj_list) > 0:
#             file.write(f"$lj-terms, {len(ff.lj_list)}\n")
#             for (atom1, atom2), sigma in zip(ff.lj_list, ff.sigmas):
#                 param = ff.c_lj[atom1-1, atom2-1]
#                 file.write(f"{atom1}, {atom2}, {param}, {sigma}\n")
