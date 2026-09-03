# Example fixtures

Each subdirectory here holds the data used by the unit and integration tests
under `tests/`. Two kinds of files live side by side, and it matters which
is which:

## Raw (hand-picked, never regenerated)

- `struc1.xyz`, `struc2.xyz` -- real reactant/product molecular geometries
  (e.g. `small_single_molecule` is reaction 34 from the Zimmerman benchmark
  set, see `h2_elimination/read.me`). These are the only inputs that were
  chosen by hand; everything else in a directory is computed from them.

## Derived (regenerate with `scripts/regenerate_examples.py`)

Produced by actually running the default ffits TS-guess pipeline
(`ffits struc1.xyz struc2.xyz`) on the raw structures above:

- `wbo1`, `wbo2` -- Wiberg bond orders
- `struc1.hess`, `struc2.hess` -- xtb Hessians
- `ff1.csv`, `ff2.csv` -- parameterized force fields for reactant/product
- `tsff.csv`, `tsff.mixing_factors.txt` -- the mixed transition-state force field
- `ts_guess.xyz`, `trj_ts_guess.xyz`, `ts_optimization.out` -- the TS-guess optimization output
- `pyout` -- a captured stdout log from the run, kept for human reference only (no test reads it)

Tests use these files as fixture *inputs* (starting geometries, force
fields to load, etc.), and deliberately check *properties* of the results
(RMSD against a known geometry, self-consistency between two runs,
finite-difference agreement, thresholds) rather than diffing exact values
out of these files. That's what keeps most tests from needing an update
every time a fixture is regenerated.

The exception is `tests/test_utils.py`'s `NAT`/`XYZ`/`ATOM_TYPES`/`WBO`
constants, which are read directly from `struc1.xyz`/`wbo1` at import time
(not hardcoded) for exactly this reason -- so they always match whatever is
currently on disk here.

## When to regenerate

Rerun `python scripts/regenerate_examples.py` after a change to the FF
model, the Hessian-fitting algorithm, or the TS-guess/mixing procedure --
i.e. whenever these derived files would no longer reflect what current
ffits code actually produces. Routine bug fixes that don't change the
underlying physics/algorithm shouldn't require it.

After regenerating: review with `git diff --stat tests/examples/` (and
spot-check a couple of the changed files) before committing, then rerun
`pytest tests/`.
