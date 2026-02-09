
from ffits.datatype.calculation_data import CalculationData, CalculationOptions, PathData, System
from ffits.datatype.structure_data import Structure,  StructuralInformation, StructurePath, ForceField
from ffits.io.reader import read_xtb_hessian, read_wbo_file, readin_xyz
from ffits.external.xtb import Xtb
from ffits.forcefield.python_interface.ff_energy import energy_ff, complete_gradient, complete_hessian



def get_preliminary_information(calcopt: CalculationOptions, pathdata: PathData, id: int, systemdata: System) -> Structure: 
    '''input is cd.reactant_calc and cd.reactant_path'''
    chrg = systemdata.charge
    mult = systemdata.multiplicity
    path = StructurePath(pathdata.xyz_filename, pathdata.hessian_filename, pathdata.wbo_filename, pathdata.ff_filename)
    xtbrunner = Xtb(chrg, mult, xtb_path=systemdata.xtb_path) 

    # TODO add somewhere check that wbo and hess needs to be calculated if geomopt is performed
    if calcopt.geometry_optimization: 
        new_xyz_filename, wbo = xtbrunner.geomopt_with_topology_check(pathdata.xyz_filename, pathdata.xyz_filename, pathdata.wbo_filename)
        path.xyz_filename = new_xyz_filename # need to change that, so that old file is ignored
        calcopt.wbo_calc = False

    nat, _, xyz, atom_types = readin_xyz(path.xyz_filename)
    
    if calcopt.hessian_calc:
        hessian = xtbrunner.hesscalc(path.xyz_filename, path.hessian_filename)
    else:
        print(f"[INFO] Skipping Hessian calculation and reading in {path.hessian_filename}.")
        hessian = read_xtb_hessian(path.hessian_filename)
    
    if calcopt.wbo_calc:
        wbo = xtbrunner.wbocalc(path.xyz_filename, path.wbo_filename)
    else:
        print(f"[INFO] Skipping WBO calculation and reading in {path.wbo_filename}.")
        wbo = read_wbo_file(path.wbo_filename)

    info = StructuralInformation(nat, xyz, wbo, atom_types, hessian)

    if calcopt.ff_parameterization:
        ff = ForceField(nat, path.ff_filename, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
    else:
        print(f"[INFO] Skipping FF parameterization and reading in {path.ff_filename}.")
        ff = ForceField(nat, path.ff_filename, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
    
    return Structure(path, ff, info)

