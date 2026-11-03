"""Reaction path generation by interpolating the TS force field mixing factors."""

from ffits.external.xtb import Xtb
from ffits.ts_guess.guess import get_ts_guess, prepare_structures
from ffits.datatype.calculation_data import CalculationData
from ffits.datatype.structure_data import Structure
from ffits.io.print.config import print_calculation_data
from ffits.io.print.details import (
    HARTREE2KCAL,
    print_rct_path_image,
    print_rct_path_start,
    print_rct_path_summary,
)
from ffits.io.reader import read_xyz_2dict, readin_xyz
from ffits.io.file_writer import write_trajectory_to_xyz
import os
import logging

logger = logging.getLogger(__name__)


def _write_structure_to_xyz(structure: dict, filename: str) -> None:
    """Write a structure dict (with 'nat', 'atom_types', 'xyz', optional 'comment') to an xyz file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"{structure['nat']}\n")
        file.write(f"{structure.get('comment', '')}\n")
        for atom_type, coords in zip(structure["atom_types"], structure["xyz"]):
            file.write(f"{atom_type} {coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f}\n")


def get_one_image(
    cd: CalculationData,
    struc1: Structure,
    struc2: Structure,
    trajectory: list,
    step: int,
) -> tuple[list, bool]:
    """Generates one TS guess image along the path and appends it to the trajectory.

    The TS optimization starts from the neighboring image: the previous image
    ``ts_guess_{step-1}.xyz``, or the reactant for the first step.

    Args:
        cd (CalculationData): Calculation data, with ``ts_calc.factor_reactant``/
            ``factor_product`` set for this step's mixing weights.
        struc1 (Structure): Prepared reactant structure (WBOs, Hessian, FF).
        struc2 (Structure): Prepared product structure (WBOs, Hessian, FF).
        trajectory (list): Trajectory frames accumulated so far.
        step (int): Index of the current step, used to name output files.

    Returns:
        tuple[list, bool]: The updated trajectory and whether the TS
        optimization for this step converged.
    """
    cd.ts_calc.start_structure = "reactant" if step == 0 else f"ts_guess_{step - 1}.xyz"
    _, converged, _, _ = get_ts_guess(struc1, struc2, calcdata=cd)
    nat, _, xyz, atom_types = readin_xyz("ts_guess.xyz")
    os.rename("ts_guess.xyz", f"ts_guess_{step}.xyz")
    os.rename(cd.ts_path.ff_filename, f"tsff_{step}.csv")
    trajectory.append({"nat": nat, "atom_types": atom_types, "xyz": xyz})
    return trajectory, converged


def create_path(
    calcdata: CalculationData = CalculationData().from_default(),
    steps: int = 10,
    minfact1: float = 0.1,
    maxfact1: float = 0.9,
) -> tuple[list, dict | None]:
    """Builds a reaction-path trajectory by sweeping the TS mixing factor between reactant and product.

    The input and the setup (xtb WBOs/Hessians, FF fitting) are done and
    printed once; every image only builds its TS FF and optimizes on it,
    starting from the previous image.

    Args:
        calcdata (CalculationData, optional): Calculation data; its
            ``ts_calc`` mixing factors and start structure are overwritten for
            each step, ``perform_two_optimizations`` is switched off and
            ``scale_one_sided_bonds`` is switched on.
        steps (int, optional): Number of intermediate steps to generate. Defaults to 10.
        minfact1 (float, optional): Reactant-product mixing factor at the first step. Defaults to 0.1.
        maxfact1 (float, optional): Reactant-product mixing factor at the last step. Defaults to 0.9.

    Returns:
        tuple[list, dict | None]: The full trajectory (including reactant
        and product endpoints) and the highest-energy frame found, if any.
    """
    if calcdata.ts_calc.start_structure not in ("", "reactant"):
        logger.warning(
            f"start_structure = {calcdata.ts_calc.start_structure} is ignored in reaction-path mode, "
            "each image starts from the previous image (the first one from the reactant)."
        )
    # every image is one optimization, started from the previous image
    calcdata.ts_calc.perform_two_optimizations = False
    # forming/breaking bonds fade out towards the structure they are missing in
    calcdata.ts_calc.scale_one_sided_bonds = True
    print_calculation_data(calcdata)
    struc1, struc2 = prepare_structures(calcdata)

    xtb = Xtb(
        chrg=calcdata.system.charge,
        mult=calcdata.system.multiplicity,
        xtb_path=calcdata.system.xtb_path,
        xtb_input_name=calcdata.system.xtb_input_name,
        g_xtb=calcdata.system.use_gxtb,
        xtb_alpb_solvent=calcdata.system.xtb_alpb_solvent,
    )
    print_rct_path_start(steps)

    # Add starting structure
    trajectory = [read_xyz_2dict(calcdata.reactant_path.xyz_filename)]
    trajectory[-1]["energy"] = xtb.singlepoint(
        f"{calcdata.reactant_path.xyz_filename}", "out_r.txt"
    )
    names = [calcdata.reactant_path.xyz_filename]
    product_factors = [0.0]

    for step in range(steps):
        if steps == 1:  # a single image (3 frames) sits halfway
            fact2 = (minfact1 + maxfact1) / 2
        else:
            fact2 = minfact1 + (maxfact1 - minfact1) * step / (steps - 1)
        fact1 = 1 - fact2
        calcdata.ts_calc.factor_reactant = fact1
        calcdata.ts_calc.factor_product = fact2
        print_rct_path_image(step, steps, fact1, fact2)

        trajectory, converged = get_one_image(calcdata, struc1, struc2, trajectory, step)
        energy = xtb.singlepoint(f"ts_guess_{step}.xyz", f"{step}")
        if not converged:
            logger.warning(
                f"Optimization for step {step} did not converge. The last valid structure of the optimization trajectory is written to ts_guess_{step}.xyz."
            )
        trajectory[-1]["energy"] = energy
        names.append(f"ts_guess_{step}.xyz")
        product_factors.append(fact2)
        print(f"  xtb energy of step {step + 1}/{steps}: {energy:.8f} Eh")

    # Add ending structure
    trajectory.append(read_xyz_2dict(calcdata.product_path.xyz_filename))
    trajectory[-1]["energy"] = xtb.singlepoint(
        f"{calcdata.product_path.xyz_filename}", "out_p.txt"
    )
    names.append(calcdata.product_path.xyz_filename)
    product_factors.append(1.0)

    highest = max(range(len(trajectory)), key=lambda i: trajectory[i]["energy"])
    highest_energy_frame = trajectory[highest].copy()
    highest_energy_frame["xyz"] = trajectory[highest]["xyz"].copy()

    # comment line of each frame: xtb energy relative to the reactant
    for frame in trajectory:
        frame["comment"] = (
            f"{(frame['energy'] - trajectory[0]['energy']) * HARTREE2KCAL:.4f} kcal/mol "
            "(xtb, relative to reactant)"
        )
    write_trajectory_to_xyz(trajectory, "path_trj.xyz")
    _write_structure_to_xyz(highest_energy_frame, "ts_guess.xyz")
    print_rct_path_summary(
        names, product_factors, [frame["energy"] for frame in trajectory], highest
    )
    return trajectory, highest_energy_frame
