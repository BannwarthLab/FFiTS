#!/bin/python



def print_box(name: str, width=80):
    print('┏' +    "━"*width       + "┓")
    print('┃' + name.center(width) + '┃')
    print('┗' +    "━"*width       + "┛")


def generate_xtb_fix_input(atoms_to_fix: set) -> str:
    string = ''
    for val in list(atoms_to_fix):
        string += (str(val)+ ', ' )
    return (f"$fix\n"
            f"   atoms: {string[:-2]}\n"
            f"end\n")


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

def write_string2file(string, filename):
    with open(filename, "w") as text_file:
        text_file.write(string)
