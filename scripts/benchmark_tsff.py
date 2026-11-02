#!/usr/bin/env python
"""Benchmark the ffits TS guess pipeline function by function.

Runs the three expensive stages of ``ffits`` on the files of a finished run
directory (no xTB calculation is done, Hessians and WBOs are read from file):

1. ``param``: FF parameterization of reactant and product
   (``fill_ff`` + ``fit_ff_to_hessian``), as in ``get_ts_guess_from_xyz``
2. ``tsff``:  TS force field construction (``create_tsff``)
3. ``opt``:   TS guess optimization on the TSFF (``optimize_with_forcefield``)

Skipped stages take their input from the run directory instead
(``ff1.csv``/``ff2.csv`` if ``param`` is skipped, ``tsff.csv`` if ``tsff`` is
skipped).

For each stage the wall time of every relevant ffits function is reported
(total, number of calls, time per call). Indented rows are called from inside
the row above, so their time is part of it. Rows named "(remaining)" give the
part of a function's time not covered by the rows below it (e.g. the time
spent inside the MolBar optimizer itself). With ``--profile`` a pyinstrument
call tree is printed in addition.

All output files (ff*.csv, tsff.csv, trajectories, profile) are written to
``--outdir`` (a temporary directory by default), so the run directory is not
touched.

Examples:
    python scripts/benchmark_tsff.py path/to/run_dir
    python scripts/benchmark_tsff.py path/to/run_dir --stages tsff --repeat 5
    python scripts/benchmark_tsff.py path/to/run_dir --stages param,tsff --compare
    python scripts/benchmark_tsff.py path/to/run_dir --profile --html profile.html
    python scripts/benchmark_tsff.py path/to/run_dir --json timings.json

Requires ``pyinstrument`` for ``--profile`` (``pip install pyinstrument``).
"""

import argparse
import functools
import json
import logging
import os
import platform
import statistics
import sys
import tempfile


def _set_threads_from_argv():
    """Limit BLAS/OpenMP threads before numpy is imported (see --threads)."""
    threads = "1"
    for i, arg in enumerate(sys.argv):
        if arg == "--threads" and i + 1 < len(sys.argv):
            threads = sys.argv[i + 1]
        elif arg.startswith("--threads="):
            threads = arg.split("=", 1)[1]
    if threads != "0":
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = threads


_set_threads_from_argv()
from collections import defaultdict
from time import perf_counter

import numpy as np

import ffits
import ffits.datatype.forcefield_data as forcefield_data
import ffits.external.molbar as molbar_interface
import ffits.forcefield.python_interface.ff_energy as ff_energy
import ffits.forcefield.python_interface.fortran_bindings as fortran_bindings
import ffits.forcefield.python_interface.optimization as optimization
import ffits.ts_guess.define_starting_parameters as starting_parameters
import ffits.ts_guess.mix_ff as mix_ff
import ffits.ts_guess.parameterize_ff as parameterize
from ffits.datatype.calculation_data import CalculationData
from ffits.datatype.forcefield_data import ForceField
from ffits.datatype.structure_data import Structure, StructureBuilder
from ffits.external.xtb import readin_xyz

STAGES = ("param", "tsff", "opt")

# Functions that are timed: (row label, module or class, attribute, depth).
# The function is wrapped where it is looked up at call time, so a call from
# anywhere in ffits is recorded. Depth only controls the indentation (and the
# "(remaining)" rows): a row is called from inside the closest row above it
# with a smaller depth.
TIMED = {
    "param": [
        ("parameterize_ff", parameterize, "parameterize_ff", 0),
        ("fill_ff", parameterize, "fill_ff", 1),
        ("filter_dihedrals", starting_parameters, "filter_dihedrals", 2),
        ("_find_hydrogen_bonding", starting_parameters, "_find_hydrogen_bonding", 2),
        ("fit_ff_to_hessian", parameterize, "fit_ff_to_hessian", 1),
        ("build_unit_hessians", parameterize, "build_unit_hessians", 2),
        ("complete_hessian", ff_energy, "complete_hessian", 2),
        ("bond_hessian", ff_energy, "bond_hessian", 3),
        ("angle_hessian", ff_energy, "angle_hessian", 3),
        ("dihedral_hessian", ff_energy, "dihedral_hessian", 3),
        ("repulsive_hessian", ff_energy, "repulsive_hessian", 3),
        ("objective_derivatives", parameterize, "objective_derivatives", 2),
        ("newton_step", parameterize, "newton_step", 2),
        ("calculate_hessian_rmsd", parameterize, "calculate_hessian_rmsd", 2),
        ("ForceField.write", ForceField, "write", 2),
    ],
    "tsff": [
        ("create_tsff", mix_ff, "create_tsff", 0),
        ("combine_ff_topology", mix_ff, "combine_ff_topology", 1),
        ("combine_ff_atoms", mix_ff, "combine_ff_atoms", 2),
        ("remove_bonds_from_repulsive", mix_ff, "remove_bonds_from_repulsive", 2),
        ("mix_parameters", mix_ff, "mix_parameters", 1),
        ("mix_reference_values", mix_ff, "mix_reference_values", 1),
        ("ForceField.write", ForceField, "write", 1),
    ],
    "opt": [
        ("optimize_with_forcefield", optimization, "optimize_with_forcefield", 0),
        ("molbar optimize_geometry", molbar_interface, "optimize_geometry", 1),
        ("energy_ff", ff_energy, "energy_ff", 2),
        ("complete_gradient", ff_energy, "complete_gradient", 2),
        ("bond_gradient", ff_energy, "bond_gradient", 3),
        ("angle_gradient", ff_energy, "angle_gradient", 3),
        ("dihedral_gradient", ff_energy, "dihedral_gradient", 3),
        ("repulsive_gradient", ff_energy, "repulsive_gradient", 3),
        ("complete_hessian", ff_energy, "complete_hessian", 2),
        ("bond_hessian", ff_energy, "bond_hessian", 3),
        ("angle_hessian", ff_energy, "angle_hessian", 3),
        ("dihedral_hessian", ff_energy, "dihedral_hessian", 3),
        ("repulsive_hessian", ff_energy, "repulsive_hessian", 3),
    ],
}

# Wrapped in every stage to count the individual calls into the Fortran library.
FORTRAN_CALLS = [
    name for name in dir(fortran_bindings) if name.startswith("get_single_")
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "rundir",
        help="Directory with struc1/2.xyz, struc1/2.hess, wbo1/2 (and ff1/2.csv, tsff.csv for skipped stages).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="TOML config (default: <rundir>/input.toml if present, else ffits defaults).",
    )
    parser.add_argument("--reactant", default="struc1.xyz")
    parser.add_argument("--product", default="struc2.xyz")
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Comma separated stages to run, out of {','.join(STAGES)} (default: all).",
    )
    parser.add_argument(
        "--threads",
        default="1",
        help="BLAS/OpenMP threads (default 1, as in a cpus-per-task=1 job; 0 = leave environment as is).",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of timed runs (default: 1)."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Do one extra (untimed) run under pyinstrument and print its call tree.",
    )
    parser.add_argument(
        "--html", default=None, help="Write the pyinstrument profile as HTML."
    )
    parser.add_argument(
        "--json", default=None, help="Write all timings as JSON (for later comparison)."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Where to write output files (default: temp dir).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the newly built ff1/ff2/tsff.csv against those in rundir.",
    )
    args = parser.parse_args()
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(args.stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(sorted(unknown))}")
    # resolve user paths before changing into outdir
    for attr in ("rundir", "config", "html", "json"):
        if getattr(args, attr):
            setattr(args, attr, os.path.abspath(getattr(args, attr)))
    return args


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
class Timer:
    """Records wall time and number of calls of wrapped functions.

    Calls are recorded under ``"<stage>:<label>"``, where ``stage`` is the
    currently running pipeline stage, so a function used in several stages
    (e.g. complete_hessian) is reported separately for each of them.
    """

    def __init__(self):
        self.times = defaultdict(float)
        self.calls = defaultdict(int)
        self.stage = "load"
        self._originals = []

    def wrap(self, label: str, owner, attr: str, prefix: str | None = None):
        """Replace ``owner.attr`` by a timed version (no-op if missing or already wrapped)."""
        func = getattr(owner, attr, None)
        if func is None or getattr(func, "_benchmark_wrapped", False):
            return

        @functools.wraps(func)
        def wrapper(*a, **kw):
            key = f"{prefix or self.stage}:{label}"
            start = perf_counter()
            try:
                return func(*a, **kw)
            finally:
                self.times[key] += perf_counter() - start
                self.calls[key] += 1

        wrapper._benchmark_wrapped = True
        self._originals.append((owner, attr, func))
        setattr(owner, attr, wrapper)

    def restore(self):
        for owner, attr, func in reversed(self._originals):
            setattr(owner, attr, func)
        self._originals.clear()


def install_timers(timer: Timer, stages: list):
    for stage in stages:
        for label, owner, attr, _ in TIMED[stage]:
            timer.wrap(label, owner, attr)
    for name in FORTRAN_CALLS:
        timer.wrap(name, fortran_bindings, name, prefix="fortran")


# ----------------------------------------------------------------------------
# Setup (mirrors ffits.ts_guess.guess.get_ts_guess_from_xyz without xTB)
# ----------------------------------------------------------------------------
def load_calcdata(args) -> CalculationData:
    """Load config and point all reactant/product paths to absolute files in rundir."""
    config = args.config or os.path.join(args.rundir, "input.toml")
    if os.path.isfile(config):
        cd = CalculationData.from_config(config, test=True)
    else:
        cd = CalculationData.from_default()

    cd.reactant_path.xyz_filename = args.reactant
    cd.product_path.xyz_filename = args.product
    for path in (cd.reactant_path, cd.product_path):
        for attr in ("xyz_filename", "wbo_filename", "hessian_filename", "ff_filename"):
            setattr(path, attr, os.path.join(args.rundir, getattr(path, attr)))
    return cd


def load_structure(pathdata, read_ff: bool, ff_filename: str) -> Structure:
    """Build a Structure purely from files (no xTB). The FF is read or left empty."""
    nat, _, xyz, atom_types = readin_xyz(pathdata.xyz_filename)
    builder = StructureBuilder(
        xyz=xyz,
        xyz_filename=pathdata.xyz_filename,
        hessian_filename=pathdata.hessian_filename,
        wbo_filename=pathdata.wbo_filename,
        ff_filename=pathdata.ff_filename if read_ff else ff_filename,
    )
    builder.nat(nat).atom_types(atom_types).path()
    builder.hessian_from_file().wbo_from_file()
    # looked up at call time, so the timed wrappers are used
    calculators = (
        lambda *a: ff_energy.energy_ff(*a),
        lambda *a: ff_energy.complete_gradient(*a),
        lambda *a: ff_energy.complete_hessian(*a),
    )
    if read_ff:
        builder.ff_from_file(*calculators)
    else:
        builder.ff_empty(*calculators)
    return builder.build()


def set_calculators(ff: ForceField):
    ff.energy_calculator = lambda *a: ff_energy.energy_ff(*a)
    ff.gradient_calculator = lambda *a: ff_energy.complete_gradient(*a)
    ff.hessian_calculator = lambda *a: ff_energy.complete_hessian(*a)


def run_param(cd: CalculationData, struc1: Structure, struc2: Structure):
    priorities = None
    if (
        not cd.reactant_calc.only_proper_dihedrals
        or not cd.product_calc.only_proper_dihedrals
    ):
        priorities = molbar_interface.get_combinded_priorities(
            struc1.info,
            struc2.info,
            bo_threshold=min(cd.reactant_calc.bo_treshold, cd.product_calc.bo_treshold),
        )
    parameterize.parameterize_ff(struc1, cd.reactant_calc, priorities)
    parameterize.parameterize_ff(struc2, cd.product_calc, priorities)


def run_tsff(cd: CalculationData, struc1: Structure, struc2: Structure) -> ForceField:
    tsff = mix_ff.create_tsff(
        ff1=struc1.ff, info1=struc1.info, ff2=struc2.ff, info2=struc2.info, calcdata=cd
    )
    set_calculators(tsff)
    return tsff


def run_opt(
    cd: CalculationData, struc1: Structure, struc2: Structure, tsff: ForceField
):
    initial = struc1.info if tsff.start_from_reactant else struc2.info
    converged, energy, _ = optimization.optimize_with_forcefield(
        initial,
        tsff,
        molbar_interface.anc_optimizer,
        cd.ts_calc,
        "trj_ts_guess.xyz",
        "ts_guess.xyz",
        opt_stdout_filename="ts_optimization.out",
    )
    return converged, energy


def run_pipeline(args, cd: CalculationData, timer: Timer, stage_times: dict) -> dict:
    """One full benchmark run; adds the wall time of each stage to stage_times."""
    run_param_stage = "param" in args.stages
    timer.stage = "load"
    start = perf_counter()
    struc1 = load_structure(cd.reactant_path, not run_param_stage, "ff1.csv")
    struc2 = load_structure(cd.product_path, not run_param_stage, "ff2.csv")
    stage_times["load"].append(perf_counter() - start)
    result = {"struc1": struc1, "struc2": struc2}

    if run_param_stage:
        timer.stage = "param"
        start = perf_counter()
        run_param(cd, struc1, struc2)
        stage_times["param"].append(perf_counter() - start)

    if "tsff" in args.stages:
        timer.stage = "tsff"
        start = perf_counter()
        tsff = run_tsff(cd, struc1, struc2)
        stage_times["tsff"].append(perf_counter() - start)
    else:
        tsff = ForceField(
            struc1.info.nat, os.path.join(args.rundir, "tsff.csv"), readff=True
        )
        set_calculators(tsff)
        tsff.start_from_reactant = _start_from_reactant(args.rundir)
    result["tsff"] = tsff

    if "opt" in args.stages:
        timer.stage = "opt"
        start = perf_counter()
        result["converged"], result["energy"] = run_opt(cd, struc1, struc2, tsff)
        stage_times["opt"].append(perf_counter() - start)
    return result


def _start_from_reactant(rundir: str) -> bool:
    """Guess the start geometry of an earlier run from its log (reactant if unknown)."""
    log = os.path.join(rundir, "pyout")
    if os.path.isfile(log):
        with open(log) as f:
            return "from product geometry" not in f.read()
    return True


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_header(args, cd, result):
    struc1, struc2 = result["struc1"], result["struc2"]
    print("=" * 78)
    print(" ffits benchmark")
    print("=" * 78)
    print(f"Run directory : {args.rundir}")
    print(f"Output dir    : {args.outdir}")
    print(f"Stages        : {', '.join(args.stages)}   (repeat {args.repeat})")
    print(f"ffits loaded  : {os.path.dirname(ffits.__file__)}")
    print(
        f"Python/numpy  : {platform.python_version()} / {np.__version__}, "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}"
    )
    print(f"Atoms         : {struc1.info.nat}")
    print(f"{'terms':<12}{'reactant':>10}{'product':>10}{'TSFF':>10}")
    for term in ("bonds", "angles", "dihedrals", "repulsive"):
        print(
            f"  {term:<10}{len(getattr(struc1.ff, term)):>10}"
            f"{len(getattr(struc2.ff, term)):>10}{len(getattr(result['tsff'], term)):>10}"
        )
    if "converged" in result:
        print(
            f"Optimization  : converged={result['converged']}, E={result['energy']:.6f}"
        )


def print_stage_table(stage: str, timer: Timer, n: int, stage_total: float):
    print(f"\n--- stage '{stage}': {stage_total:.3f} s per run " + "-" * 30)
    print(
        f"  {'function':<40}{'calls':>8}{'total [s]':>11}{'per call':>11}{'share':>7}"
    )
    rows = TIMED[stage]
    for i, (label, _, _, depth) in enumerate(rows):
        key = f"{stage}:{label}"
        if timer.calls.get(key, 0) == 0:
            continue
        total = timer.times[key] / n
        calls = timer.calls[key] / n
        _print_row("  " * depth + label, calls, total, stage_total)

        # time of this function not covered by its direct children
        children = []
        for child_label, _, _, child_depth in rows[i + 1 :]:
            if child_depth <= depth:
                break
            if child_depth == depth + 1:
                children.append(f"{stage}:{child_label}")
        covered = sum(timer.times.get(c, 0.0) for c in children) / n
        if children and any(timer.calls.get(c, 0) for c in children):
            _print_row(
                "  " * (depth + 1) + "(remaining)", None, total - covered, stage_total
            )


def _print_row(label, calls, total, stage_total):
    calls_str = f"{calls:>8.0f}" if calls is not None else " " * 8
    per_call = _fmt_time(total / calls) if calls else ""
    share = 100 * total / stage_total if stage_total else 0.0
    print(f"  {label:<40}{calls_str}{total:>11.3f}{per_call:>11}{share:>6.1f}%")


def _fmt_time(t: float) -> str:
    if t >= 1:
        return f"{t:.2f} s"
    if t >= 1e-3:
        return f"{1e3 * t:.2f} ms"
    return f"{1e6 * t:.1f} us"


def print_fortran_table(timer: Timer, n: int):
    keys = [
        f"fortran:{name}"
        for name in FORTRAN_CALLS
        if timer.calls.get(f"fortran:{name}")
    ]
    if not keys:
        return
    print(
        "\n--- calls into Fortran (all stages; time incl. Python wrapper) " + "-" * 14
    )
    print(f"  {'function':<40}{'calls':>8}{'total [s]':>11}{'per call':>11}")
    for key in keys:
        total = timer.times[key] / n
        calls = timer.calls[key] / n
        print(
            f"  {key.split(':', 1)[1]:<40}{calls:>8.0f}{total:>11.3f}{_fmt_time(total / calls):>11}"
        )


def compare_ff(new: ForceField, ref_file: str, nat: int, name: str):
    """Print the max deviation of parameters/reference values per term type."""
    if not os.path.isfile(ref_file):
        print(f"\n{name}: no reference {ref_file}")
        return
    ref = ForceField(nat, ref_file, readff=True)
    print(f"\n{name} compared against {ref_file}:")
    print(
        f"  {'term':<10} {'n_new':>7} {'n_ref':>7} {'max|dparam|':>12} {'max|dref|':>12}"
    )
    for term in ("bonds", "angles", "dihedrals", "repulsive"):
        df_new = getattr(new, term).set_index("atoms")
        df_ref = getattr(ref, term).set_index("atoms")
        common = df_new.index.intersection(df_ref.index)
        if len(common):
            dpar = np.abs(
                df_new.loc[common, "parameter"].astype(float)
                - df_ref.loc[common, "parameter"].astype(float)
            ).max()
            dref = np.abs(
                df_new.loc[common, "reference_value"].astype(float)
                - df_ref.loc[common, "reference_value"].astype(float)
            ).max()
        else:
            dpar = dref = 0.0
        print(
            f"  {term:<10} {len(df_new):>7} {len(df_ref):>7} {dpar:>12.2e} {dref:>12.2e}"
        )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)

    cd = load_calcdata(args)
    args.outdir = os.path.abspath(
        args.outdir or tempfile.mkdtemp(prefix="ffits_bench_")
    )
    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(args.outdir)
    cd.ts_path.ff_filename = "tsff.csv"  # relative, i.e. written to outdir

    timer = Timer()
    stage_times = defaultdict(list)
    install_timers(timer, args.stages)
    try:
        for _ in range(args.repeat):
            result = run_pipeline(args, cd, timer, stage_times)
    finally:
        timer.restore()

    # The profile is recorded in an extra, untimed run: the profiler slows down
    # code with many small Python calls by a factor 2-3.
    profiler = None
    if args.profile:
        from pyinstrument import Profiler

        profiler = Profiler(interval=0.001)
        with profiler:
            run_pipeline(args, cd, Timer(), defaultdict(list))

    n = args.repeat
    print_header(args, cd, result)
    print(f"\nLoading files: {statistics.mean(stage_times['load']):.3f} s per run")
    for stage in STAGES:
        if stage in stage_times:
            print_stage_table(stage, timer, n, statistics.mean(stage_times[stage]))
    print_fortran_table(timer, n)
    print(
        "\nIndented rows are part of the row above; '(remaining)' is the time of"
        " the row above\nthat is not spent in the listed sub-functions."
    )
    if n > 1:
        for stage in STAGES:
            if stage in stage_times:
                ts = stage_times[stage]
                print(
                    f"  {stage}: min {min(ts):.3f} s, max {max(ts):.3f} s over {n} runs"
                )

    if args.compare:
        nat = result["struc1"].info.nat
        if "param" in args.stages:
            compare_ff(
                result["struc1"].ff, os.path.join(args.rundir, "ff1.csv"), nat, "ff1"
            )
            compare_ff(
                result["struc2"].ff, os.path.join(args.rundir, "ff2.csv"), nat, "ff2"
            )
        if "tsff" in args.stages:
            compare_ff(
                result["tsff"], os.path.join(args.rundir, "tsff.csv"), nat, "tsff"
            )

    if args.json:
        data = {
            "rundir": args.rundir,
            "nat": result["struc1"].info.nat,
            "repeat": n,
            "stages": {s: stage_times[s] for s in stage_times},
            "functions": {
                k: {"total": timer.times[k] / n, "calls": timer.calls[k] / n}
                for k in timer.times
            },
        }
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nTimings written to {args.json}")

    if profiler is not None:
        # ffits may live in site-packages, which pyinstrument hides by default
        print(
            profiler.output_text(
                unicode=True,
                color=False,
                processor_options={
                    "show_regex": r".*/(ffits|molbar)/.*",
                    "hide_regex": r".*/(pandas|numpy|scipy)/.*",
                },
            )
        )
        if args.html:
            with open(args.html, "w") as f:
                f.write(profiler.output_html())
            print(f"HTML profile written to {args.html}")


if __name__ == "__main__":
    main()
