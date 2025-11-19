# pyTSguess



## Introduction

The goal of this program is to generate a transition state (TS) guess from a reactant and a product structure. 


## Usage

After installation (see below), the code can be executed while in the virtual environment by:
```
ffits reactant.xyz product.xyz
```

Command line keywords are:
--input
--multiplicity
--charge

For using the input, see below

## Installation
To get this code running, please execute following lines:
```
git clone git@git.rwth-aachen.de:bannwarthlab/ffits.git
cd ffits 
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt 
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone git@git.rwth-aachen.de:bannwarthlab/molbar.git	
cd molbar
git fetch
git checkout dev
make install
```
The second part is necessary as the molbar optimizer is used, which is in the dev branch of molbar. 


## Configuration file

todo