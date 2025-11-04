
from src.datatype.calculation_data import CalculationData, CalculationOptions, PathData
from src.datatype.structure_data import Structure,  StructuralInformation, StructurePath, ForceField
from src.io.reader import read_hessian, read_wbo_file, readin_xyz
from src.external.xtb import Xtb



def get_preliminary_information(calcopt: CalculationOptions, pathdata: PathData, id: int) -> Structure: 
    '''input is cd.reactant_calc and cd.reactant_path'''
    path = StructurePath(pathdata.xyz_filename, pathdata.hessian_filename, pathdata.wbo_filename, pathdata.ff_filename)
    xtbrunner = Xtb(calcopt.system.charge, calcopt.system.multiplicity)


    if calcopt.geometry_optimization: 
        new_xyz_filename = xtbrunner.geomopt_with_topology_check(pathdata.xyz_filename, pathdata.xyz_filename)
        path.xyz_filename = new_xyz_filename # need to change that, so that old file is ignored

    nat, _, xyz, atom_types = readin_xyz(path.xyz_filename)
    
    if calcopt.hessian_calc:
        hessian = xtbrunner.hesscalc(path.xyz_filename, path.hess_filename)
    else:
        hessian = read_hessian(path.hess_filename)
    
    if calcopt.wbo_calc:
        wbo = xtbrunner.hesscalc(path.xyz_filename, path.wbo_filename)
    else:
        wbo = read_wbo_file(path.wbo_filename)

    info = StructuralInformation(nat, xyz, wbo, atom_types, hessian)

    ff = ForceField(nat, path.ff_filename)
    
    return Structure(path, ff, info)
    #TODO add FF initilization and then return Structure()


def parameterize_ff():
    pass