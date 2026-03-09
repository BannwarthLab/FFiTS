import numpy as np
import os
from ffits.datatype.structure_data import ForceField, StructuralInformation
from molbar.utils.optimizer import optimize_geometry 
from scipy.optimize import minimize


def anc_optimizer(xyz_start: np.ndarray, ff: ForceField, atom_types: np.ndarray, e_tol: float = 1e-4, x_tol: float = 1e-3, max_micro_steps: int = 1, trajectory_filename: str = 'trajectory.xyz', final_geometry_filename: str = 'optimized.xyz'):
    if ff.energy_calculator == None:
        raise Exception('Function for energy calculation needs to be defined in FF object when using the ANC optimization.')
    if ff.gradient_calculator == None:
        raise Exception('Function for gradient calculation needs to be defined in FF object when using the ANC optimization.')
    if ff.hessian_calculator == None:
        raise Exception('Function for hessian calculation needs to be defined in FF object when using the ANC optimization.')
    
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
        verbose=True
    )
    print(f"Converged: {converged}")
    print(f"Final energy: {energy:.6f}")
    print(f"Steps taken: {steps}")
    print(f"Duration: {time:.2f} seconds")

    return converged, energy, final_geom, steps, time, message

def failed_anc_opt(filename: str) -> bool:
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'Final gradient norm: nan' in line.strip():
            print(f'[WARNING] ANC optimization failed.')
            return True 
    return False
  
def write_last_valid_xyz(trajectory_filename: str = 'trajectory.xyz', final_geometry_filename: str = 'optimized.xyz'):
    with open(trajectory_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    nat = int(lines[0].strip())
    total_lines = len(lines)
    start_idx = max(total_lines - (nat + 2) * 2, 0)
    end_idx = min(start_idx + (nat + 2), total_lines)

    selected_lines = lines[start_idx:end_idx]

    os.remove(final_geometry_filename)
    with open(final_geometry_filename, 'w', encoding='utf-8') as out:
        out.writelines(selected_lines)




def scipy_optimizer(xyz_start: np.ndarray, ff: ForceField, struc: StructuralInformation):
    x0 = xyz_start.flatten()

    # Run scipy optimization
    result = minimize(
        ff.get_energy, 
        x0, 
        method='Newton-CG',
        jac=ff.get_gradient,
        hess=ff.get_hessian,
        callback=struc.scipy_optimizer_callback,
        options={
            'maxiter': 5000,
            'xtol': 1e-6
        }
    )

    return result


    
