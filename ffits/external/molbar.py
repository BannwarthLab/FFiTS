import numpy as np
import os
import logging
from ffits.datatype.structure_data import ForceField, StructuralInformation
from molbar.utils.optimizer import optimize_geometry

from scipy.optimize import minimize
from molbar.topology.priorities import _calculate_priorities
from molbar.topology.topology import _get_topology_barcode
from molbar.molecule.molecule import Molecule

logger = logging.getLogger(__name__)


def _define_bonds_for_molbar(
    mol: Molecule,
    struc1: StructuralInformation,
    struc2: StructuralInformation,
    bo_threshold: float = 0.5,
) -> None:
    """Fills mol object with connectivity information based on the combined bond order matrices of struc1 and struc2. Bonds are defined where the combined bond order exceeds the specified threshold.

    Args:
        mol (Molecule): MolBar Molecule object to be filled with connectivity information.
        struc1 (StructuralInformation): Structural information of reactant structure.
        struc2 (StructuralInformation): Structural information of product structure.
        bo_threshold (float, optional): Threshold for defining bonds. Defaults to 0.5.

    Raises:
        ValueError: If either struc1 or struc2 is missing bond order matrices.
    """
    # raise error when necessary data is missing in struc1 or struc2
    if struc1.bo_matrix is None or struc2.bo_matrix is None:
        raise ValueError(
            "Both struc1 and struc2 must have bond order matrices to define bonds for MolBar."
        )

    mixed_bo_matrix = struc1.bo_matrix + struc2.bo_matrix
    mol.cn_matrix = np.zeros(mixed_bo_matrix.shape, dtype=int)

    for i in range(mixed_bo_matrix.shape[0]):
        for j in range(i + 1, mixed_bo_matrix.shape[1]):
            if mixed_bo_matrix[i, j] > bo_threshold:
                mol.cn_matrix[i, j] = 1
                mol.cn_matrix[j, i] = 1
    mol.cn = np.sum(mol.cn_matrix, axis=1)


def get_combinded_priorities(
    struc1: StructuralInformation,
    struc2: StructuralInformation,
    bo_threshold: float = 0.5,
) -> dict:
    """Calculates combined priorities for each atom based on the connectivity of both structures. This is done by creating a MolBar Molecule object, defining bonds based on the combined bond order matrices of struc1 and struc2, and then calculating priorities using MolBar's internal functions.

    Args:
        struc1 (StructuralInformation): Structural information of reactant structure.
        struc2 (StructuralInformation): Structural information of product structure.
        bo_threshold (float, optional): Threshold for defining bonds. Defaults to 0.0.

    Returns:
        dict: Dictionary mapping atom indices to their combined priorities based on the connectivity of both structures.
    """
    # Assuming atom types are the same for both structures
    # coordinates are not used, struc1 is just set as a proxy
    mol = Molecule(coordinates=struc1.xyz, elements=struc2.atom_types)

    _define_bonds_for_molbar(mol, struc1, struc2, bo_threshold=bo_threshold)
    _get_topology_barcode(mol)
    _calculate_priorities(mol)

    return {i: int(mol.priorities[i]) for i in range(len(mol.priorities))}


def anc_optimizer(
    xyz_start: np.ndarray,
    ff: ForceField,
    atom_types: np.ndarray,
    e_tol: float = 1e-4,
    x_tol: float = 1e-3,
    max_micro_steps: int = 1,
    trajectory_filename: str = "trajectory.xyz",
    final_geometry_filename: str = "optimized.xyz",
):
    """Optimization using the ANC optimizer from MolBar.

    Args:
        xyz_start (np.ndarray): starting geometry for the optimization, shape (nat, 3)
        ff (ForceField): Force field with defined energy, gradient and hessian calculator
        atom_types (np.ndarray): Array of atom types, shape (nat,)
        e_tol (float, optional): Energy tolerance for convergence. Defaults to 1e-4.
        x_tol (float, optional): Displacement tolerance for convergence. Defaults to 1e-3.
        max_micro_steps (int, optional): Maximum number of micro-steps for each optimization iteration. Defaults to 1.
        trajectory_filename (str, optional): Filename for the trajectory file. Defaults to "trajectory.xyz".
        final_geometry_filename (str, optional): Filename for the final geometry file. Defaults to "optimized.xyz".

    Raises:
        Exception: Energy, gradient and hessian calculators need to be defined in the ForceField object when using the ANC optimization.

    Returns:
        _type_: _description_
    """
    if ff.energy_calculator == None:
        raise Exception(
            "Function for energy calculation needs to be defined in FF object when using the ANC optimization."
        )
    if ff.gradient_calculator == None:
        raise Exception(
            "Function for gradient calculation needs to be defined in FF object when using the ANC optimization."
        )
    if ff.hessian_calculator == None:
        raise Exception(
            "Function for hessian calculation needs to be defined in FF object when using the ANC optimization."
        )

    converged, energy, final_geom, steps, time, message = optimize_geometry(
        geometry=xyz_start,
        elements=atom_types,
        trajectory_file=trajectory_filename,
        final_geometry_file=final_geometry_filename,
        energy_func=ff.get_energy,
        gradient_func=ff.get_gradient,
        hessian_func=ff.get_hessian,
        masses=None,  # Optional: provide atomic masses
        trust_radius=0.1,
        e_tol=e_tol,
        x_tol=x_tol,
        max_steps=1000,
        max_micro_steps=max_micro_steps,
        verbose=True,
    )
    print(f"Converged: {converged}")
    print(f"Final energy: {energy:.6f}")
    print(f"Steps taken: {steps}")
    print(f"Duration: {time:.2f} seconds")

    return converged, energy, final_geom, steps, time, message


def failed_anc_opt(filename: str) -> bool: # TODO change this to just checking the message 
    """Checks, whether the ANC optimization from the anc_optimizer function was successfull or not by checking its output

    Args:
        filename (str): filename of the output file of the ANC optimization

    Returns:
        bool: True, if the optimization failed due to NaN gradient, False otherwise
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Final gradient norm: nan" in line.strip():
            logger.warning(
                f"ANC optimization failed due to NaN gradient. Check {filename} for details."
            )
            return True
    return False


def write_last_valid_xyz(
    trajectory_filename: str = "trajectory.xyz",
    final_geometry_filename: str = "optimized.xyz",
):
    """Writes out the last valid geometry from the trajectory file of the ANC optimization to a separate file. 

    Args:
        trajectory_filename (str, optional): filename of the trajectory file. Defaults to "trajectory.xyz".
        final_geometry_filename (str, optional): filename of the final geometry file, where the last valid geometry will be written. Defaults to "optimized.xyz".
    """
    with open(trajectory_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    nat = int(lines[0].strip())
    total_lines = len(lines)
    start_idx = max(total_lines - (nat + 2) * 2, 0)
    end_idx = min(start_idx + (nat + 2), total_lines)

    selected_lines = lines[start_idx:end_idx]

    os.remove(final_geometry_filename)
    with open(final_geometry_filename, "w", encoding="utf-8") as out:
        out.writelines(selected_lines)


def scipy_optimizer(
    xyz_start: np.ndarray, ff: ForceField, struc: StructuralInformation
):
    """Geometry optimization using scipy

    Args:
        xyz_start (np.ndarray): starting geometry, shape (nat, 3)
        ff (ForceField): Force field with defined energy, gradient and hessian calculator
        struc (StructuralInformation): Structural information

    Returns:
        tuple: Optimization result
    """
    if ff.energy_calculator == None:
        raise Exception(
            "Function for energy calculation needs to be defined in FF object when using the ANC optimization."
        )
    if ff.gradient_calculator == None:
        raise Exception(
            "Function for gradient calculation needs to be defined in FF object when using the ANC optimization."
        )
    if ff.hessian_calculator == None:
        raise Exception(
            "Function for hessian calculation needs to be defined in FF object when using the ANC optimization."
        )
    x0 = xyz_start.flatten()

    # Run scipy optimization
    result = minimize(
        ff.get_energy,
        x0,
        method="Newton-CG",
        jac=ff.get_gradient,
        hess=ff.get_hessian,
        callback=struc.scipy_optimizer_callback,
        options={"maxiter": 5000, "xtol": 1e-6},
    )

    return result
