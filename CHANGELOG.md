# Changelog

## pre [0.9.2] - 2026-11-02

### Added
- **`hydrogen_covalent_radius`** option (default `True`). For forming or breaking covalent H–X bonds, the side where the bond is missing uses the covalent radius of H (0.32 Å) instead of its vdW radius as the reference distance, but never less than the bonded length. Hydrogen bonds keep the vdW distance.
- **Fortran build info** (compiler, flags, build date) is written to `build_info.txt` and printed in the ffits header.
- `scripts/benchmark_tsff.py` for benchmarking TS guesses against reference TSs.

### Changed
- **Hessian-weighted mixing removed.** The TS FF is now always the plain `factor_reactant`/`factor_product` average of reactant and product. The options `average_with_hess_weight`, `hess_weight_sharpness` and `hess_weight_angles_dihedrals` no longer exist.
- **`perform_two_optimizations` is now `True` by default.** The TS FF is optimized from both reactant and product, and the result with the lower xtb (g-xTB if `use_gxtb`) singlepoint energy is kept.
- **Faster TS FF construction and FF energy evaluation.** Parameter/reference mixing uses hash lookups instead of per-term DataFrame scans, and the FF energy and geometry functions are vectorized.

### Fixed
- Gradient and Hessian were passed to the molbar optimizer per bohr instead of per Å (factor 1.89 / 3.57 too small).
- Wrong second derivative of the angle term in the Fortran Hessian. Intel compilers are now forced to `-fp-model=precise`, since the default fast model broke the angle/dihedral Hessians.
- Hydrogen-bond pairs were stored unsorted.
- The molbar optimizer modified the input geometry in place; it now works on a copy.

## [0.9.1] - 2026-10-09

### Added
- **`--timing` command-line flag** for TS guess mode. It enables a new `TIMING` log level and writes a per-stage report (`ffits_timing.log`) covering xTB calculations, FF parameterization, TSFF construction and optimization, plus total time and completion status. 
- **Element-dependent bond-order thresholds** (`ffits/utils/bond_threshold.py`). Bonds between two first- or second-period atoms (H–Ne) now use a fixed threshold of 0.3. Pairs with at least one heavier atom keep using the configured `bo_treshold`. 
- **Linear-angle check for dihedrals.** Proper and improper dihedrals built on a (near-)linear bond angle (within `linear_angle_tolerance_deg`, default 5°) are dropped with a warning. 

### Changed
- **Reworked FF parameter fitting** in `parameterize_ff.py`. The per-parameter sequential updates are replaced by a damped Newton step on all parameters at once. 
- **New defaults:** `only_proper_dihedrals` is now `False` (was `True`), and `ff_parameter_repulsion` is now `0.001` (was `0.01`).

### Fixed
- Reaction-path generation failed when writing the last valid structure. `write_last_valid_xyz` is now called with the trajectory and final-geometry filenames.
- The central bond of a dihedral was not canonicalized, so `(j, l)` and `(l, j)` were treated as different bonds. It is now sorted.
- Fixed thresholds in tests.
