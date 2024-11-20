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

# Modules

for more information on modules, there is documentation in the code. 

maxit, stepsize, threshold, constant_repulsion, rep_start