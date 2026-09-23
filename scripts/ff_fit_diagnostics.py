#!/usr/bin/env python
"""
ff_fit_diagnostics.py -- standalone FF-parametrization diagnostic report.

Runs an ffits force-field-to-Hessian fit for a given molecule (xyz + WBO +
xtb Hessian files) and renders a PDF (or PNG pages) with a few sanity
checks: how well the fitted FF Hessian reproduces the reference Hessian,
whether the fit made the RMSD better, and whether parameters collapsed to
(near) zero or blew up -- the two problems this was written to catch.

It uses only the "front door" of ffits:

    ffits.datatype.structure_data.StructureBuilder
    ffits.datatype.forcefield_data.ForceField        (.bonds/.angles/
        .dihedrals/.repulsive DataFrames' "type"/"atoms"/"parameter"
        columns, and .get_hessian())
    ffits.forcefield.python_interface.ff_energy.{complete_hessian,
        complete_gradient, energy_ff}
    ffits.external.xtb.readin_xyz
    ffits.ts_guess.define_starting_parameters.fill_ff
    ffits.ts_guess.parameterize_ff.fit_ff_to_hessian

It deliberately does NOT import any of the helper functions that implement
the current Newton-Raphson step itself (build_unit_hessians,
objective_derivatives, _fitted_term_groups, newton_step, ...). Those are
treated as private/version-specific. Everything this script needs from a
fit -- before/after parameters and before/after FF Hessians -- is read
back out through the ForceField object itself, so the script should keep
working even if the internals of fit_ff_to_hessian are rewritten, as long
as fill_ff() and fit_ff_to_hessian() keep roughly their current job and
signature. If a future/older ffits version has renamed or reshaped either
of those two calls, this script will fail with a clear error naming the
call that broke, rather than silently producing a wrong report -- that
boundary is the price of being portable across versions, so treat a
failure there as "the adapter needs a small update", not as your fit being
broken.

No files this script needs are found in your ffits installation? It just
imports "ffits" the way your own scripts do, so run it with the same
Python environment (venv) you use to run ffits.

Usage:
    python ff_fit_diagnostics.py struc.xyz wbo_file struc.hess -o report.pdf

    python ff_fit_diagnostics.py struc.xyz wbo_file struc.hess \\
        --maxit 1000 --stepsize 0.15 --threshold 0.0005 \\
        --repulsion-start 0.01 --vary-repulsion \\
        --collapse-threshold 0.05 -o report.pdf
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np


TERM_TYPES = ["bonds", "angles", "dihedrals", "repulsive"]


# --------------------------------------------------------------------------
# Step 1: build the Structure (xyz + WBO + reference Hessian -> Structure)
# --------------------------------------------------------------------------
def build_structure(xyz_file: str, wbo_file: str, hessian_file: str, bo_threshold: float):
    try:
        from ffits.datatype.structure_data import StructureBuilder
        from ffits.forcefield.python_interface.ff_energy import (
            complete_gradient,
            complete_hessian,
            energy_ff,
        )
        from ffits.external.xtb import readin_xyz
    except ImportError as exc:
        raise SystemExit(
            f"Could not import ffits' structure-building interface ({exc}).\n"
            "Make sure you are running this script with the same Python "
            "environment (venv) you use for ffits."
        )

    nat, _comment, xyz, atom_types = readin_xyz(str(xyz_file))

    builder = (
        StructureBuilder(
            xyz=xyz,
            xyz_filename=str(xyz_file),
            wbo_filename=str(wbo_file),
            hessian_filename=str(hessian_file),
            bondorder_threshold=bo_threshold,
        )
        .nat(nat)
        .atom_types(atom_types)
        .path()
        .hessian_from_file()
        .wbo_from_file()
        .ff_empty(
            energy_calculator=energy_ff,
            gradient_calculator=complete_gradient,
            hessian_calculator=complete_hessian,
        )
    )
    return builder.build()


# --------------------------------------------------------------------------
# Step 2: run fill_ff + fit_ff_to_hessian, snapshotting parameters and
# FF Hessians before and after the fit.
# --------------------------------------------------------------------------
def snapshot_parameters(ff) -> dict:
    """{"type": np.ndarray of the 'parameter' column} for each term type present."""
    snap = {}
    for term in TERM_TYPES:
        df = getattr(ff, term)
        snap[term] = df["parameter"].to_numpy(dtype=float, copy=True) if len(df) else np.array([])
    return snap


def offdiagonal_block_mask(dim: int) -> np.ndarray:
    """Boolean (dim, dim) mask that is False on the 3x3 per-atom diagonal
    blocks and True everywhere else -- i.e. the cross-atom Hessian entries
    that ff_fit_objective_function compares against the reference. Written
    out here rather than imported so this script has no dependency on the
    internal helper of the same purpose."""
    nat = dim // 3
    mask = np.ones((dim, dim), dtype=bool)
    for i in range(nat):
        mask[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = False
    return mask


def run_fit(struc, args) -> dict:
    try:
        from ffits.ts_guess.define_starting_parameters import fill_ff
    except ImportError as exc:
        raise SystemExit(f"Could not import ffits.ts_guess.define_starting_parameters.fill_ff ({exc}).")
    try:
        from ffits.ts_guess.parameterize_ff import fit_ff_to_hessian
    except ImportError as exc:
        raise SystemExit(f"Could not import ffits.ts_guess.parameterize_ff.fit_ff_to_hessian ({exc}).")

    ff = struc.ff
    info = struc.info

    try:
        fill_ff(
            ff,
            info,
            repulsive_start=args.repulsion_start,
            bo_threshold=args.bo_threshold,
        )
    except TypeError:
        # Older/newer fill_ff may not know every keyword above -- fall back
        # to the bare call rather than giving up.
        fill_ff(ff, info)

    dim = 3 * ff.nat
    mask = offdiagonal_block_mask(dim)
    hessian_ref = info.hessian

    params_before = snapshot_parameters(ff)
    hessian_ff_before = ff.get_hessian(info.fortran_xyz)

    fit_result = None
    try:
        fit_result = fit_ff_to_hessian(
            struc,
            maxit=args.maxit,
            stepsize=args.stepsize,
            threshold=args.threshold,
            constant_repulsion=not args.vary_repulsion,
        )
    except TypeError:
        # Signature drifted -- still try the fit with defaults only, rather
        # than aborting the whole report.
        print(
            "Warning: fit_ff_to_hessian() did not accept the usual keyword "
            "arguments on this ffits version; calling it with defaults.",
            file=sys.stderr,
        )
        fit_result = fit_ff_to_hessian(struc)

    params_after = snapshot_parameters(ff)
    hessian_ff_after = ff.get_hessian(info.fortran_xyz)

    return {
        "mask": mask,
        "hessian_ref": hessian_ref,
        "hessian_ff_before": hessian_ff_before,
        "hessian_ff_after": hessian_ff_after,
        "params_before": params_before,
        "params_after": params_after,
        "fit_result": fit_result,
        "ff": ff,
    }


# --------------------------------------------------------------------------
# Step 3: metrics
# --------------------------------------------------------------------------
def hessian_metrics(hessian_ff, hessian_ref, mask):
    ff_entries = hessian_ff[mask]
    ref_entries = hessian_ref[mask]
    diff = ff_entries - ref_entries
    rmsd = float(np.sqrt(np.mean(diff**2)))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((ref_entries - np.mean(ref_entries)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmsd": rmsd, "r2": r2, "ff_entries": ff_entries, "ref_entries": ref_entries}


def collapse_fractions(params_before: dict, params_after: dict, threshold: float):
    """Fraction of parameters per type that the FIT drove below `threshold`
    (started at/above it, ended below it -- a proxy for 'this term stopped
    contributing to the FF during fitting'), and the fraction that grew by
    more than 10x -- the two failure modes this was written to catch. A
    parameter that started below `threshold` and was simply left there
    (e.g. a repulsive term kept constant) is not counted as newly
    collapsed."""
    out = {}
    for term in TERM_TYPES:
        c0 = params_before.get(term, np.array([]))
        c1 = params_after.get(term, np.array([]))
        n = len(c1)
        if n == 0:
            out[term] = {"n": 0, "collapsed": 0.0, "blown_up": 0.0}
            continue
        newly_collapsed = (np.abs(c0) >= threshold) & (np.abs(c1) < threshold)
        collapsed = float(np.mean(newly_collapsed))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(c1) / np.where(np.abs(c0) > 1e-12, np.abs(c0), np.nan)
        blown_up = float(np.nanmean(ratio > 10.0)) if n else 0.0
        out[term] = {"n": n, "collapsed": collapsed, "blown_up": blown_up}
    return out


# --------------------------------------------------------------------------
# Step 4: rendering
# --------------------------------------------------------------------------
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "surface": "#fcfcfb",
}
TERM_COLORS = {
    "bonds": PALETTE["blue"],
    "angles": PALETTE["orange"],
    "dihedrals": PALETTE["aqua"],
    "repulsive": PALETTE["yellow"],
}


def render_report(data: dict, args, molecule_label: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink2"],
            "ytick.color": PALETTE["ink2"],
            "font.size": 10,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
        }
    )

    mask = data["mask"]
    metrics_before = hessian_metrics(data["hessian_ff_before"], data["hessian_ref"], mask)
    metrics_after = hessian_metrics(data["hessian_ff_after"], data["hessian_ref"], mask)
    collapse = collapse_fractions(data["params_before"], data["params_after"], args.collapse_threshold)

    figures = []

    # --- Figure 1: summary panel -----------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.axis("off")
    fit_result = data["fit_result"] or {}
    lines = [
        f"FF parametrization diagnostics -- {molecule_label}",
        "",
        f"Hessian RMSD before fit:  {metrics_before['rmsd']:.5f}",
        f"Hessian RMSD after fit:   {metrics_after['rmsd']:.5f}",
        f"R² before fit:            {metrics_before['r2']:.4f}",
        f"R² after fit:             {metrics_after['r2']:.4f}",
    ]
    if isinstance(fit_result, dict):
        it = fit_result.get("iterations")
        fr = fit_result.get("final_rmsd")
        fo = fit_result.get("final_objectiv_function", fit_result.get("final_objective_function"))
        if it is not None:
            lines.append(f"Iterations reported by fit_ff_to_hessian: {it}")
        if fr is not None:
            lines.append(f"final_rmsd reported by fit_ff_to_hessian: {fr:.5f}")
        if fo is not None:
            lines.append(f"final objective reported by fit_ff_to_hessian: {fo:.5f}")
    else:
        lines.append(
            "(fit_ff_to_hessian on this ffits version did not return a "
            "recognizable summary dict -- only before/after snapshots shown.)"
        )
    lines.append("")
    lines.append(f"Parameters the fit drove below |c|={args.collapse_threshold:g} ('collapsed'):")
    if not args.vary_repulsion:
        lines.append("(repulsive parameters were held constant -- pass --vary-repulsion to fit them too)")
    for term in TERM_TYPES:
        c = collapse[term]
        if c["n"] == 0:
            continue
        lines.append(
            f"  {term:<10s} n={c['n']:<5d} collapsed={c['collapsed']*100:5.1f}%   "
            f"grew >10x={c['blown_up']*100:5.1f}%"
        )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    figures.append(fig)

    # --- Figure 2: parity plots (before vs after) -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, metrics, title in zip(
        axes, [metrics_before, metrics_after], ["Before fit", "After fit"]
    ):
        ax.scatter(
            metrics["ref_entries"], metrics["ff_entries"], s=4, alpha=0.35, color=PALETTE["blue"]
        )
        lo = min(metrics["ref_entries"].min(), metrics["ff_entries"].min())
        hi = max(metrics["ref_entries"].max(), metrics["ff_entries"].max())
        ax.plot([lo, hi], [lo, hi], color=PALETTE["muted"], linewidth=1, linestyle="--")
        ax.set_title(f"{title}\nRMSD={metrics['rmsd']:.4f}  R²={metrics['r2']:.4f}")
        ax.set_xlabel("Reference Hessian entry")
    axes[0].set_ylabel("FF Hessian entry")
    fig.suptitle("Off-diagonal-block Hessian entries: FF vs. reference")
    fig.tight_layout()
    figures.append(fig)

    # --- Figure 3: parameter movement, faceted by term type ---------------
    present_terms = [t for t in TERM_TYPES if len(data["params_after"].get(t, [])) > 0]
    if present_terms:
        fig, axes = plt.subplots(1, len(present_terms), figsize=(3.2 * len(present_terms), 4.2), sharey=True)
        if len(present_terms) == 1:
            axes = [axes]
        for ax, term in zip(axes, present_terms):
            c0 = data["params_before"][term]
            c1 = data["params_after"][term]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    (np.abs(c0) > 1e-12) & (np.abs(c1) > 1e-300),
                    np.log10(np.abs(c1) / np.abs(c0)),
                    np.nan,
                )
            ratio = ratio[np.isfinite(ratio)]
            if len(ratio) == 0:
                ax.axis("off")
                ax.set_title(f"{term}\n(no finite ratios)")
                continue
            jitter = (np.random.default_rng(0).random(len(ratio)) - 0.5) * 0.5
            ax.scatter(jitter, ratio, s=10, alpha=0.5, color=TERM_COLORS.get(term, PALETTE["blue"]))
            ax.axhline(0, color=PALETTE["muted"], linewidth=1)
            ax.axhline(np.log10(args.collapse_threshold), color=PALETTE["critical"], linewidth=1, linestyle=":")
            ax.set_xticks([])
            ax.set_title(f"{term} (n={len(c1)})")
        axes[0].set_ylabel(r"$\log_{10}(|c_{final}| / |c_{start}|)$")
        fig.suptitle("Parameter movement during the fit (dotted line = collapse threshold)")
        fig.tight_layout()
        figures.append(fig)

    # --- output -------------------------------------------------------
    out_path = Path(args.out)
    if out_path.suffix.lower() == ".pdf":
        with PdfPages(out_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
    else:
        # PNG(s): treat --out as a stem, write one numbered file per figure.
        stem = out_path.with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        for idx, fig in enumerate(figures, start=1):
            fig.savefig(stem.parent / f"{stem.name}_{idx}.png", dpi=200)

    for fig in figures:
        plt.close(fig)

    return out_path


# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xyz", help="xyz coordinate file")
    p.add_argument("wbo", help="Wiberg bond order file (xtb wbo format)")
    p.add_argument("hessian", help="xtb Hessian file")
    p.add_argument("-o", "--out", default="ff_fit_diagnostics.pdf", help="Output path. '.pdf' -> one multi-page PDF, anything else -> numbered PNGs (default: %(default)s)")
    p.add_argument("--bo-threshold", type=float, default=0.0, help="Bond order threshold for connectivity (default: %(default)s)")
    p.add_argument("--repulsion-start", type=float, default=0.01, help="Starting value for repulsive parameters (default: %(default)s)")
    p.add_argument("--vary-repulsion", action="store_true", help="Let repulsive parameters be updated during the fit (default: kept constant)")
    p.add_argument("--maxit", type=int, default=1000, help="Max fit iterations (default: %(default)s)")
    p.add_argument("--stepsize", type=float, default=0.15, help="Newton step scaling (default: %(default)s)")
    p.add_argument("--threshold", type=float, default=0.0005, help="RMSD-change convergence threshold (default: %(default)s)")
    p.add_argument("--collapse-threshold", type=float, default=0.05, help="|c| below this after the fit counts as 'collapsed' (default: %(default)s)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    molecule_label = Path(args.xyz).stem

    try:
        struc = build_structure(args.xyz, args.wbo, args.hessian, args.bo_threshold)
        data = run_fit(struc, args)
        out_path = render_report(data, args, molecule_label)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(
            "\nff_fit_diagnostics.py failed -- see traceback above. If this "
            "points into ffits itself rather than this script, the ffits "
            "version in your venv may have changed fill_ff() or "
            "fit_ff_to_hessian() in a way this adapter does not know about yet."
        )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
