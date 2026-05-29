#!/bin/python

from pathlib import Path

import numpy as np


def print_box(name: str, width=80):
    print("┏" + "━" * width + "┓")
    print("┃" + name.center(width) + "┃")
    print("┗" + "━" * width + "┛")


def generate_xtb_fix_input(atoms_to_fix: set) -> str:
    string = ""
    for val in list(atoms_to_fix):
        string += str(val) + ", "
    return f"$fix\n" f"   atoms: {string[:-2]}\n" f"end\n"


def write_all_molecules_to_file(seperate_molecule_list, new_filename_prefix):
    def subgraph_to_xyz(xyz, graph, new_filename: str) -> None:
        def relate_index(ind_G: int) -> int:
            return ind_G + 1

        lines = [str(graph.number_of_nodes()) + "\n", "\n"]
        for node in graph.nodes:
            lines.append(xyz[relate_index(node)])
        with open(new_filename, "w") as file:
            for line in lines:
                file.write(line)

    id = 0
    for graph in seperate_molecule_list:
        id = id + 1
        subgraph_to_xyz(graph, new_filename_prefix + str(id) + ".xyz")
        print("Subgraph", graph, "printed to", new_filename_prefix + str(id) + ".xyz")


def write_string2file(string, filename):
    with open(filename, "w") as text_file:
        text_file.write(string)


def write_xyz_to_file(xyz: np.ndarray, filename: str) -> None:
    nat = xyz.shape[0]
    lines = [str(nat) + "\n", "\n"]
    for i in range(nat):
        lines.append(f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f}\n")
    with open(filename, "w") as file:
        for line in lines:
            file.write(line)


def write_hessian_to_orcahessfile(
    nat: int, hessian: np.ndarray, xyz_with_masses: np.ndarray, filename: str
):
    lines = []
    natoms = nat

    # Header
    lines.append("$orca_hessian_file\n")

    # Atoms section
    lines.append("\n$atoms\n")
    lines.append(f"{natoms}\n")
    for atomline in xyz_with_masses:
        # Format: symbol mass x y z charge
        lines.append(f" {atomline}\n")

    # Hessian section
    lines.append("\n$hessian\n")
    dim = natoms * 3
    lines.append(f"{dim}\n")

    # Print hessian in blocks of 5 columns
    cols_per_block = 5
    for col_start in range(0, dim, cols_per_block):
        col_end = min(col_start + cols_per_block, dim)

        # Print column headers
        header = "".rjust(0)
        for j in range(col_start, col_end):
            header += f"{j:>18}"
        lines.append(header + "\n")

        # Print rows
        for i in range(dim):
            row = f"{i:>5}"
            for j in range(col_start, col_end):
                row += f"{hessian[i][j]:>18.10E}"
            lines.append(row + "\n")

    lines.append("\n$end\n")

    with open(filename, "w") as text_file:
        for line in lines:
            text_file.write(line)

def write_trajectory_to_xyz(trajectory: list, filename: Path) -> None:
    """Writes a trajectory which is saved as 

    Args:
        trajectory (list): List of directories for every frame. Each directory should contain the keys 'nat', 'atom_types' (np.ndarray), 'xyz', and optionally 'energy'.
        filename (Path): _description_
    """
    with open(filename, 'w') as f:
        for frame in trajectory:
            nat = frame['nat']
            atom_types = frame['atom_types']
            xyz = frame['xyz']
            
            f.write(f"{nat}\n")
            f.write(f"{frame.get('energy', 0):.8f}\n")
            for i, atom_type in enumerate(atom_types):
                f.write(f"{atom_type} {xyz[i][0]:.8f} {xyz[i][1]:.8f} {xyz[i][2]:.8f}\n")
            