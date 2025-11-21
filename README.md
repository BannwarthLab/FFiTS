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

Configuration is done via a TOML file (default: `config.toml`). The configuration file is optional; if not provided, default values will be used.

### Using a configuration file

Pass a configuration file using the `--input` flag:
```bash
ffits reactant.xyz product.xyz --input config.toml
```

### Configuration Parameters

#### **System Settings** (`[system]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `charge` | int | 0 | Total charge of the molecular system |
| `multiplicity` | int | 1 | Spin multiplicity (1=singlet, 2=doublet, 3=triplet, etc.). Must be between 1 and 3. |

**Example:**
```toml
[system]
charge = -1
multiplicity = 1
```

---

#### **Reactant Settings** (`[reactant]`)

##### Reactant Paths (`[reactant.path]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xyz_filename` | string | "struc1.xyz" | Path to reactant structure file (XYZ format) |
| `wbo_filename` | string | "wbo1" | Path to file with Wiberg Bond Orders for reactant |
| `hessian_filename` | string | "hess1" | Path to file with Hessian matrix for reactant |
| `ff_filename` | string | "ff1" | Path to file with force field parameters for reactant in csv file format |

##### Reactant Calculations (`[reactant.calculation]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geometry_optimization` | bool | false | Whether to perform geometry optimization with GFN2-xTB |
| `wbo_calc` | bool | true | Whether to calculate WBO at runtime. If false, read from `wbo_filename` |
| `hessian_calc` | bool | true | Whether to calculate Hessian at runtime. If false, read from `hessian_filename` |
| `ff_parameterization` | bool | false | Whether to parameterize force field. If false, read from `ff_filename` |
| `test_parameterization` | bool | false | If true, use parameterized FF to optimize structure and print RMSD |
| `ff_parameterization_maxiteration` | int | 1000 | Maximum iterations for FF parameterization |
| `ff_parameterization_stepsize` | float | 0.15 | Step size for FF parameterization optimization |
| `ff_parameterization_threshold` | float | 0.0005 | Convergence threshold for FF parameterization (lower = more prone to overfitting) |
| `ff_parameterization_constant_repulsion` | bool | true | If true, do not change repulsive term parameters during fitting |

**Example:**
```toml
[reactant.path]
xyz_filename = "reactant.xyz"
wbo_filename = "reactant.wbo"

[reactant.calculation]
wbo_calc = true
hessian_calc = false
hessian_filename = "reactant.hess"
ff_parameterization = true
ff_parameterization_threshold = 0.001
```

---

#### **Product Settings** (`[product]`)

##### Product Paths (`[product.path]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xyz_filename` | string | "struc2.xyz" | Path to product structure file (XYZ format) |
| `wbo_filename` | string | "wbo2" | Path to file with Wiberg Bond Orders for product |
| `hessian_filename` | string | "hess2" | Path to file with Hessian matrix for product |
| `ff_filename` | string | "ff2" | Path to file with force field parameters for product in csv file format |

##### Product Calculations (`[product.calculation]`)

Same as Reactant Calculations (see above). All the same parameters apply to the product structure.

**Example:**
```toml
[product.path]
xyz_filename = "product.xyz"

[product.calculation]
geometry_optimization = true
ff_parameterization = true
```

---

#### **Transition State Guess Settings** (`[ts_guess_calculation]`)

##### TS Paths (`[ts_guess_calculation.path]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ff_filename` | string | "tsff" | Output path for transition state force field parameters |

##### TS Calculations (`[ts_guess_calculation.calculation]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `factor_reactant` | float | 0.5 | Weight of reactant in TS interpolation (0.0-1.0) |
| `factor_product` | float | 0.5 | Weight of product in TS interpolation (0.0-1.0). Must sum to 1.0 with factor_reactant |
| `optimizer` | string | "molbar-optimizer" | Which optimizer to use. Options: `"molbar-optimizer"`, `"scipy-optimizer"` |
| `molbar_optimizer_e_tol` | float | 1e-8 | Energy tolerance for molbar optimizer |
| `molbar_optimizer_g_tol` | float | 1e-2 | Gradient tolerance for molbar optimizer |
| `molbar_optimizer_x_tol` | float | 1e-3 | Step size tolerance for molbar optimizer |
| `molbar_optimizer_max_micro_steps` | int | 1 | Maximum micro-steps in molbar optimizer |
| `energy_threshold_two_optimizations` | float | 0.15 | Energy threshold for repeating optimization from product if reactant path is too high |
| `perform_two_optimizations` | bool | false | If true, optimize from both reactant and product, then choose lower energy structure |

**Example:**
```toml
[ts_guess_calculation.path]
ff_filename = "ts_forcefield.ff"

[ts_guess_calculation.calculation]
factor_reactant = 0.6
factor_product = 0.4
optimizer = "molbar-optimizer"
molbar_optimizer_e_tol = 1e-7
perform_two_optimizations = true
```

---

#### **Postprocessing Settings** (`[postprocessing]`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relaxation` | string | "None" | Post-optimization relaxation method. Options: `"None"`, `"gfn2-xtb"`, `"pbeh-3c"`. Fixes reactive atoms and performs geometry optimization at chosen theory level. |

**Example:**
```toml
[postprocessing]
relaxation = "gfn2-xtb"
```

---

### Complete Example Configuration

```toml
[system]
charge = 0
multiplicity = 1

[reactant.path]
xyz_filename = "reactant.xyz"
wbo_filename = "reactant.wbo"
hessian_filename = "reactant.hess"
ff_filename = "reactant.ff"

[reactant.calculation]
geometry_optimization = false
wbo_calc = true
hessian_calc = true
ff_parameterization = false
test_parameterization = false
ff_parameterization_maxiteration = 1000
ff_parameterization_stepsize = 0.15
ff_parameterization_threshold = 0.0005
ff_parameterization_constant_repulsion = true

[product.path]
xyz_filename = "product.xyz"
wbo_filename = "product.wbo"
hessian_filename = "product.hess"
ff_filename = "product.ff"

[product.calculation]
geometry_optimization = false
wbo_calc = true
hessian_calc = true
ff_parameterization = false
test_parameterization = false
ff_parameterization_maxiteration = 1000
ff_parameterization_stepsize = 0.15
ff_parameterization_threshold = 0.0005
ff_parameterization_constant_repulsion = true

[ts_guess_calculation.path]
ff_filename = "ts_forcefield.ff"

[ts_guess_calculation.calculation]
factor_reactant = 0.5
factor_product = 0.5
optimizer = "molbar-optimizer"
molbar_optimizer_e_tol = 1e-8
molbar_optimizer_g_tol = 1e-2
molbar_optimizer_x_tol = 1e-3
molbar_optimizer_max_micro_steps = 1
energy_threshold_two_optimizations = 0.15
perform_two_optimizations = false

[postprocessing]
relaxation = "None"
```

### Notes

- **Constraints**: 
  - `multiplicity` must be 1, 2, or 3
  - `factor_reactant + factor_product` must equal 1.0
  - `optimizer` must be one of the specified options
  - `relaxation` must be one of the specified options

- **Command-line Override**: You can override specific settings using command-line arguments:
  ```bash
  ffits reactant.xyz product.xyz --input config.toml --charge -1 --multiplicity 2
  ```
