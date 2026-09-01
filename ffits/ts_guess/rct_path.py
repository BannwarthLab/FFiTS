"""Reaction path generation by interpolating the TS force field mixing factors."""
from ffits.external.xtb import Xtb
from ffits.ts_guess.guess import get_ts_guess_from_xyz
from ffits.datatype.calculation_data import CalculationData
from ffits.io.reader import read_xyz_2dict, readin_xyz
from ffits.io.file_writer import write_trajectory_to_xyz
import os
import logging
import contextlib

logger = logging.getLogger(__name__)


def _write_structure_to_xyz(structure: dict, filename: str) -> None:
    """Write a structure dict (with 'nat', 'atom_types', 'xyz', optional 'comment') to an xyz file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"{structure['nat']}\n")
        file.write(f"{structure.get('comment', '')}\n")
        for atom_type, coords in zip(structure["atom_types"], structure["xyz"]):
            file.write(f"{atom_type} {coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f}\n")


def get_one_image(cd: CalculationData, trajectory: list, step: int) -> list:
    """Generates one TS guess image along the path and appends it to the trajectory.

    For steps after the first, reuses the already-computed WBO/Hessian/FF
    data instead of recomputing it.

    Args:
        cd (CalculationData): Calculation data, with ``ts_calc.factor_reactant``/
            ``factor_product`` set for this step's mixing weights.
        trajectory (list): Trajectory frames accumulated so far.
        step (int): Index of the current step, used to name output files.

    Returns:
        tuple[list, bool]: The updated trajectory and whether the TS
        optimization for this step converged.
    """
    nat, _, xyz, atom_types = readin_xyz(cd.reactant_path.xyz_filename)
    if step != 0:
        cd.reactant_calc.wbo_calc = False
        cd.reactant_calc.hessian_calc = False
        cd.reactant_calc.ff_parameterization = False
        cd.product_calc.wbo_calc = False
        cd.product_calc.hessian_calc = False
        cd.product_calc.ff_parameterization = False
    tsff, converged, energy, final_geom = get_ts_guess_from_xyz(
        cd.reactant_path.xyz_filename, cd.product_path.xyz_filename, cd
    )
    _, _, xyz, atom_types = readin_xyz("optimized.xyz")
    os.rename("optimized.xyz", f"optimized_{step}.xyz")
    os.rename("tsff.csv", f"tsff_{step}.csv")
    trajectory.append({"nat": nat, "atom_types": atom_types, "xyz": xyz})
    return trajectory, converged


def create_path(
    calcdata: CalculationData = CalculationData().from_default(),
    steps: int = 10,
    minfact1: float = 0.1,
    maxfact1: float = 0.9,
) -> tuple[list, dict | None]:
    """Builds a reaction-path trajectory by sweeping the TS mixing factor between reactant and product.

    Args:
        calcdata (CalculationData, optional): Calculation data; its
            ``ts_calc`` mixing factors are overwritten for each step.
        steps (int, optional): Number of intermediate steps to generate. Defaults to 10.
        minfact1 (float, optional): Reactant-product mixing factor at the first step. Defaults to 0.1.
        maxfact1 (float, optional): Reactant-product mixing factor at the last step. Defaults to 0.9.

    Returns:
        tuple[list, dict | None]: The full trajectory (including reactant
        and product endpoints) and the highest-energy frame found, if any.
    """
    if calcdata.ts_calc.average_with_hess_weight:
        logger.warning(
            "Weighting the reactant and product FF with the hessian is currently not implemented for the reaction path mode. The factors will be varied from minfact1 to maxfact1 for the reactant to get the reaction path."
        )
        calcdata.ts_calc.average_with_hess_weight = False

    trajectory = []
    highest_energy_frame = None
    highest_energy_value = None
    xtb = Xtb(
        chrg=calcdata.system.charge,
        mult=calcdata.system.multiplicity,
        xtb_path=calcdata.system.xtb_path,
        xtb_input_name=calcdata.system.xtb_input_name,
        g_xtb=calcdata.system.use_gxtb,
        xtb_alpb_solvent=calcdata.system.xtb_alpb_solvent,
    )

    # Add starting structure
    struc1 = read_xyz_2dict(calcdata.reactant_path.xyz_filename)
    energy = xtb.singlepoint(f"{calcdata.reactant_path.xyz_filename}", f"out_r.txt")
    trajectory.append(struc1)
    trajectory[-1]["energy"] = energy
    highest_energy_frame = trajectory[-1].copy()
    highest_energy_frame["xyz"] = trajectory[-1]["xyz"].copy()
    highest_energy_value = energy

    for step in range(steps):
        fact2 = minfact1 + (maxfact1 - minfact1) * step / (steps - 1)
        fact1 = 1 - fact2
        print(
            f"Step {step}: factor_reactant = {fact1:.2f}, factor_product = {fact2:.2f}"
        )
        calcdata.ts_calc.factor_reactant = fact1
        calcdata.ts_calc.factor_product = fact2

        trajectory, converged = get_one_image(calcdata, trajectory, step)
        energy = xtb.singlepoint(f"optimized_{step}.xyz", f"{step}")
        if not converged:
            logger.warning(
                f"Optimization for step {step} did not converge. The last valid structure of the optimization trajectory is written to optimized_{step}.xyz."
            )
        # add energy to trajectory
        trajectory[-1]["energy"] = energy
        if highest_energy_value is None or energy > highest_energy_value:
            highest_energy_frame = trajectory[-1].copy()
            highest_energy_frame["xyz"] = trajectory[-1]["xyz"].copy()
            highest_energy_value = energy
        print(f"Energy at step {step}: {energy:.8f} Hartree")

    # Add ending structure
    struc2 = read_xyz_2dict(calcdata.product_path.xyz_filename)
    energy = xtb.singlepoint(f"{calcdata.product_path.xyz_filename}", f"out_p.txt")
    trajectory.append(struc2)
    trajectory[-1]["energy"] = energy
    if highest_energy_value is None or energy > highest_energy_value:
        highest_energy_frame = trajectory[-1].copy()
        highest_energy_frame["xyz"] = trajectory[-1]["xyz"].copy()
        highest_energy_value = energy

    write_trajectory_to_xyz(trajectory, "path_trj.xyz")
    if highest_energy_frame is not None:
        _write_structure_to_xyz(highest_energy_frame, "optimized.xyz")
    return trajectory, highest_energy_frame
