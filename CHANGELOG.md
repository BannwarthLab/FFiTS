# Changelog

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
