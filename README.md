# pyTSguess



## Introduction

This project is a Python wrapper for the [CREST code](https://git.rwth-aachen.de/bannwarthlab/crest.git) in branch FF_TSsearch. The goal of the whole program is to calculate a transition state (TS) guess from reactant and product structures of a given reaction. This guess can then be further optimized with common optimization methods, like the [ORCA optts keyword](https://sites.google.com/site/orcainputlibrary/geometry-optimizations) to get a true TS with one imaginary mode. 

This python wrapper serves several purposes (wip):
- automated generation of toml-style CREST input files for the necessary calculations
- an alignment procedure of multiple molecules in one xyz file to improve the TS guess calculation
- straight forward way to generate and change FF values, generated through the FF generation of the tsff method in CREST 
- error catching (e.g. non-converged TS guess calculations, where last structure of log is used)


## How does TS guess calculation in CREST work?
The generation of the TS guess in CREST can be seperated into three steps: The force field (FF) parameterization of reactant and product structures, the generation of the TS FF and the optimization using the TS FF.
The employed FF is reused from [MolBar](https://git.rwth-aachen.de/bannwarthlab/molbar.git) with added repulsive interactions. Firstly, two system-specific FFs (ssFF) are parameterized through fitting to a QM Hessian so that the ssFFs represent the electronic landscape at the minima. These two FFs are then averaged bond-, angle-, and dihedral angle-wise to generate the new TS FF. The TS FF can then be used as the potential for an optimization through [CRESTs ancopt routine](https://crest-lab.github.io/crest-docs/page/documentation/inputfiles.html) to get the TS guess. Additionally, this code adds a seperate alignment routine to improve the result of the TS guess calculation.


## CREST Installation
As the code is not in the master branch of crest, either ask me for the binary or clone [CREST](https://git.rwth-aachen.de/bannwarthlab/crest.git) and checkout to branch FF_TSsearch. You can find more information of compliling in the CREST repo, but this is my compliling process, starting in the folder crest and with loaded meson and cmake:

```
source /opt/intel/oneapi/setvars.sh
export FC=ifort CC=icc
meson _build
cd _build
ninja -j 10
```

## Modules

for more information on modules, there is documentation in the code. 


## psssht, tis a secret
add file "optvals" with following values to influence the ff fitting:
maxit, stepsize, threshold, constant_repulsion, rep_start
the standard is:
```
    maxit = 10000000                # maximal iterations of newton raphson for ff fitting
    stepsize = 0.05_wp              # stepsize to influence the update step
    threshold = 0.001_wp            # val_before - val_now lt threshold ends calculation
    constant_repulsion = .true.     # repulsion is not fitted but kept constant
    rep_start = 0.01                # the starting ff param for all repulsive interactions
```
a working file (with the name 'optvals') would be:
```
10000000, 0.05, 0.001, True, 0.01
```
ATTENTION! Every value needs to be defined for it to work. So you CAN NOT write sth like `1000, 0.05, 0.001, 0.01` in optvals.




## Compiling fortran code 
How to compile the fortran code (scikit-build needs to be pip installed)
```
rm -rf _skbuild/ pyTSguess.egg-info/ && python setup.py build_ext --inplace 
```

The current list of packages is in the virtual environment is: 
```
Package       Version
------------- -------
distro        1.9.0  
networkx      3.1    
numpy         1.24.4 
packaging     25.0   
pip           20.0.2 
pkg-resources 0.0.0  
scikit-build  0.18.1 
setuptools    44.0.0 
tomli         2.2.1  
wheel         0.45.1 
```