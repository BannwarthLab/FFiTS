
from src.datatype.calculation_data import CalculationData, CalculationOptions, PathData
from src.datatype.structure_data import Structure,  StructuralInformation, StructurePath, ForceField
from src.io.reader import read_hessian, read_wbo_file, readin_xyz
from src.external.xtb import Xtb
from src.forcefield.python_interface.ff_energy import energy_ff, complete_gradient, complete_hessian



def get_preliminary_information(calcopt: CalculationOptions, pathdata: PathData, id: int, chrg: int, mult: int) -> Structure: 
    '''input is cd.reactant_calc and cd.reactant_path'''
    path = StructurePath(pathdata.xyz_filename, pathdata.hessian_filename, pathdata.wbo_filename, pathdata.ff_filename)
    xtbrunner = Xtb(chrg, mult)

    # TODO add somewhere check that wbo and hess needs to be calculated if geomopt is performed
    if calcopt.geometry_optimization: 
        new_xyz_filename, wbo = xtbrunner.geomopt_with_topology_check(pathdata.xyz_filename, pathdata.xyz_filename, pathdata.wbo_filename)
        path.xyz_filename = new_xyz_filename # need to change that, so that old file is ignored
        calcopt.wbo_calc = False

    nat, _, xyz, atom_types = readin_xyz(path.xyz_filename)
    
    if calcopt.hessian_calc:
        hessian = xtbrunner.hesscalc(path.xyz_filename, path.hess_filename)
    else:
        hessian = read_hessian(path.hess_filename)
    
    if calcopt.wbo_calc:
        wbo = xtbrunner.wbocalc(path.xyz_filename, path.wbo_filename)
    else:
        wbo = read_wbo_file(path.wbo_filename)

    info = StructuralInformation(nat, xyz, wbo, atom_types, hessian)

    ff = ForceField(nat, path.ff_filename, energy_calculator=energy_ff, gradient_calculator=complete_gradient, hessian_calculator=complete_hessian)
    
    return Structure(path, ff, info)
    #TODO add FF initilization and then return Structure()


def parameterize_ff():
    pass