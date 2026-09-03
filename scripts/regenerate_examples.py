#!/usr/bin/env python3
"""Regenerate the derived example fixtures under tests/examples/.

Each example directory contains two *raw* inputs (struc1.xyz, struc2.xyz --
real molecular geometries, e.g. from a benchmark reaction set) and a set of
*derived* files that are produced by actually running the ffits pipeline on
those two geometries: wbo1/wbo2, struc1.hess/struc2.hess, ff1.csv/ff2.csv,
tsff.csv, tsff.mixing_factors.txt, optimized.xyz, trajectory.xyz,
ts_optimization.out, and a captured stdout log (pyout).

Tests use the derived files as fixture *inputs*, not as exact expected
output, so they don't normally need updating. But when the FF model, the
Hessian-fitting algorithm, or the TS-guess procedure changes meaningfully,
the derived files stop reflecting what current ffits code would actually
produce, and hand-editing them (or copies of their values elsewhere, like
tests/test_utils.py's NAT/XYZ/ATOM_TYPES/WBO constants, which are read
directly from struc1.xyz/wbo1) is how they go stale. This script is the
single, repeatable way to bring them back in sync with the current code.

Requirements:
    - ffits installed and on PATH (`make install-dev`, or `pip install -e .`
      after building the Fortran extension).
    - xtb and molbar available (xtb on PATH; molbar importable), since a
      real run computes bond orders and Hessians with them.

Usage:
    python scripts/regenerate_examples.py                  # all examples
    python scripts/regenerate_examples.py small_single_molecule
    python scripts/regenerate_examples.py --list

After running, inspect the changes with `git diff --stat tests/examples/`
(and spot-check a few numeric files) before committing -- a regeneration
that silently produces very different geometries or force fields is a sign
something regressed, not a reason to accept the new fixtures blindly. Then
rerun `pytest tests/` to make sure nothing that *does* assert on specific
fixture-derived values (there is very little of this by design -- see
tests/examples/README.md) needs a matching update.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPO_ROOT / "tests" / "examples"

RAW_FILES = ("struc1.xyz", "struc2.xyz")


def discover_examples() -> list[Path]:
    """Find example directories that contain both raw structure files."""
    if not EXAMPLES_ROOT.is_dir():
        return []
    return sorted(
        d
        for d in EXAMPLES_ROOT.iterdir()
        if d.is_dir() and all((d / f).is_file() for f in RAW_FILES)
    )


def check_prerequisites() -> None:
    """Exit with an error message if ffits, xtb, or molbar are not available."""
    missing = []
    if shutil.which("ffits") is None:
        missing.append(
            "'ffits' is not on PATH (build + install it first, e.g. `make install-dev`)"
        )
    if shutil.which("xtb") is None:
        missing.append("'xtb' is not on PATH")
    try:
        import molbar  # noqa: F401
    except ImportError:
        missing.append("'molbar' is not importable in this Python environment")

    if missing:
        print(
            "Cannot regenerate examples -- missing prerequisites:\n  - "
            + "\n  - ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)


def regenerate_one(example_dir: Path) -> None:
    """Rerun the default ffits TS-guess pipeline on one example's raw
    structures, in an isolated temp dir, then copy every file it produced
    back over the example directory."""
    print(f"\n=== Regenerating {example_dir.relative_to(REPO_ROOT)} ===")

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for raw_file in RAW_FILES:
            shutil.copy2(example_dir / raw_file, tmp_path / raw_file)

        command = ["ffits", "struc1.xyz", "struc2.xyz"]
        print(f"  running: {' '.join(command)}  (cwd={tmp_path})")
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True
        )
        (tmp_path / "pyout").write_text(result.stdout)

        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(
                f"ffits exited with code {result.returncode} for {example_dir.name}; "
                "not touching the existing fixtures."
            )

        produced = [
            p
            for p in tmp_path.iterdir()
            if p.is_file() and p.name not in RAW_FILES
        ]
        if not produced:
            raise RuntimeError(
                f"ffits ran successfully but produced no new files for "
                f"{example_dir.name} -- refusing to overwrite fixtures with nothing."
            )

        for produced_file in produced:
            dest = example_dir / produced_file.name
            shutil.copy2(produced_file, dest)
            print(f"  updated {dest.relative_to(REPO_ROOT)}")


def main() -> int:
    """Parse CLI args and regenerate the requested (or all) example fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        help="Names of example directories to regenerate (default: all of them).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discoverable example directories and exit.",
    )
    args = parser.parse_args()

    all_examples = discover_examples()
    if args.list:
        for d in all_examples:
            print(d.name)
        return 0

    if args.examples:
        by_name = {d.name: d for d in all_examples}
        unknown = [name for name in args.examples if name not in by_name]
        if unknown:
            print(f"Unknown example dir(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(by_name) or '(none found)'}", file=sys.stderr)
            return 1
        targets = [by_name[name] for name in args.examples]
    else:
        targets = all_examples

    if not targets:
        print(f"No example directories with {RAW_FILES} found under {EXAMPLES_ROOT}")
        return 1

    check_prerequisites()

    for example_dir in targets:
        regenerate_one(example_dir)

    print(
        "\nDone. Review the changes before committing:\n"
        "  git diff --stat tests/examples/\n"
        "  pytest tests/\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
