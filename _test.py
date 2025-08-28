# import sys
# sys.path.insert(0,"./_skbuild/linux-x86_64-3.8/cmake-build")
import src.force_field.fortran_forcefield as fortran_forcefield


print(fortran_forcefield.fortran_helper.cross([1,2,3],[3,4,5]))


