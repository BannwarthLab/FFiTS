

from src.datatype.calculation_data import CalculationData, CalculationOptions, PathData
from src.datatype.structure_data import Structure,  StructuralInformation, StructurePath, ForceField
from src.io.reader import read_hessian, read_wbo_file, readin_xyz

def preliminary 


def get_preliminary_information(calcopt: CalculationOptions, pathd: PathData) -> Structure: 
    '''input is cd.reactant_calc and cd.reactant_path'''
    path = StructurePath(pathd.xyz_filename, pathd.hessian_filename, pathd.wbo_filename, pathd.ff_filename)


    if calcopt.geometry_optimization: 
        pass # and set new name

    nat, comment, xyz, atom_types = readin_xyz(path.xyz_filename)
    
    if calcopt.hessian_calc:
        pass
    else:
        hessian = read_hessian(path.hess_filename)
    
    if calcopt.wbo_calc:
        pass
    else:
        wbo = read_wbo_file(path.wbo_filename)

    info = StructuralInformation()



