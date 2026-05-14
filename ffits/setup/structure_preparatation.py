import logging

from ffits.io.reader import read_xtb_hessian, read_wbo_file, readin_xyz
from ffits.external.xtb import Xtb
from ffits.utils.temp_dir_manager import TempDirManager
from ffits.forcefield.python_interface.ff_energy import (
    energy_ff,
    complete_gradient,
    complete_hessian,
)
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def randomize_coordinates(
    xyz: np.ndarray, displacement: float = 0.01, random_seed: int = 42
) -> np.ndarray:
    """Randomizes the coordinates of a structure by adding small random displacements to each atom's position. This can be useful for testing the robustness of the TS guess generation process against small perturbations in the input geometry.

    Args:
        xyz (np.ndarray): Original coordinates of the structure (shape: [nat, 3]).
        displacement (float, optional): Maximum displacement for each atom. Defaults to 0.1.
        random_seed (int, optional): Seed for reproducibility of randomization. Defaults to 42.

    Returns:
        np.ndarray: Randomized coordinates of the structure (shape: [nat, 3]).
    """
    np.random.seed(random_seed)
    # randomize only the sign of the displacement, but the displacement itself is fixed to ensure consistency
    random_signs = np.random.choice([-1, 1], size=xyz.shape)  # Adjust scale as needed
    random_displacement = (
        np.random.rand(*xyz.shape) * np.random.rand() * displacement
    )  # Random displacement scaled by the specified factor
    randomized_xyz = xyz + random_signs * displacement
    # randomize_xyz = xyz + random_displacement# random_signs * displacement
    return randomized_xyz


# def get_preliminary_information(
#     cd: CalculationData,
#     calcopt: CalculationOptions,
#     pathdata: PathData,
#     id: int,
#     random_seed: int | None = None,
# ) -> Structure:
#     """Creates Structure object 

#     Args:
#         cd (CalculationData): _description_
#         calcopt (CalculationOptions): _description_
#         pathdata (PathData): _description_
#         id (int): _description_
#         random_seed (int | None, optional): _description_. Defaults to None.

#     Returns:
#         Structure: _description_
#     """
    
#     xtbrunner = Xtb(
#         chrg = cd.system.charge,
#         mult = cd.system.multiplicity,
#         xtb_path=cd.system.xtb_path,
#         xtb_alpb_solvent=cd.system.xtb_alpb_solvent,
#         xtb_input_name=cd.system.xtb_input_name,
#     )

#     if calcopt.geometry_optimization:
#         new_xyz_filename, _ = xtbrunner.geomopt_with_topology_check(
#             pathdata.xyz_filename, pathdata.xyz_filename, pathdata.wbo_filename
#         )
#         pathdata.xyz_filename = (
#             new_xyz_filename  # I change that, so that old file is ignored
#         )
#         calcopt.wbo_calc = False

#     nat, _, xyz, atom_types = readin_xyz(pathdata.xyz_filename)

#     if random_seed is not None:
#         xyz = randomize_coordinates(xyz, displacement=0.01, random_seed=random_seed)

#     strucbuilder = StructureBuilder(xyz=xyz, 
#                                     xyz_filename=pathdata.xyz_filename, 
#                                     hessian_filename=pathdata.hessian_filename,
#                                     wbo_filename=pathdata.wbo_filename, 
#                                     ff_filename=pathdata.ff_filename)
    
#     strucbuilder.nat(nat)
#     strucbuilder.atom_types(atom_types)
    
#     if calcopt.hessian_calc: 
#         strucbuilder.hessian_from_xtb(xtbrunner)
#     else:
#         logger.info(
#             f"Skipping Hessian calculation and reading in {pathdata.hessian_filename}."
#         )
#         strucbuilder.hessian_from_file()
    
#     if calcopt.wbo_calc:
#         strucbuilder.wbo_from_xtb(xtbrunner)
#     else:
#         logger.info(f"Skipping WBO calculation and reading in {pathdata.wbo_filename}.")
#         strucbuilder.wbo_from_file()

#     if calcopt.ff_parameterization:
#         strucbuilder.ff_empty(energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
#     else:
#         logger.info(f"Skipping FF parameterization and reading in {pathdata.ff_filename}.")
#         strucbuilder.ff_from_file(energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)

#     struc = strucbuilder.build()

#     return struc
