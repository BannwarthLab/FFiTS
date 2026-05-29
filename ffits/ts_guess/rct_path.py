from ffits.external.xtb import Xtb
from ffits.ts_guess.guess import get_ts_guess_from_xyz
from ffits.datatype.calculation_data import CalculationData
from ffits.io.reader import read_xyz_2dict, readin_xyz
from ffits.io.file_writer import write_trajectory_to_xyz
import os
import logging
import contextlib

logger = logging.getLogger(__name__)


def get_one_image(
    cd: CalculationData, trajectory: list, step: int
) -> list:
    """Calculates an image of the path, adds it to the existing trajectory and returns the whole trajectory

    Args:
        cd (CalculationData): _description_
        trajectory (list): _description_
        step (int): _description_

    Returns:
        list: _description_
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
) -> list:
    if calcdata.ts_calc.average_with_hess_weight:
        logger.warning(
            "Weighting the reactant and product FF with the hessian is currently not implemented for the reaction path mode. The factors will be varied from minfact1 to maxfact1 for the reactant to get the reaction path."
        )
        calcdata.ts_calc.average_with_hess_weight = False

    trajectory = []
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
        print(f"Energy at step {step}: {energy:.8f} Hartree")

    # Add ending structure
    struc2 = read_xyz_2dict(calcdata.product_path.xyz_filename)
    energy = xtb.singlepoint(f"{calcdata.product_path.xyz_filename}", f"out_p.txt")
    trajectory.append(struc2)
    trajectory[-1]["energy"] = energy

    write_trajectory_to_xyz(trajectory, "path_trj.xyz")
    return trajectory,
