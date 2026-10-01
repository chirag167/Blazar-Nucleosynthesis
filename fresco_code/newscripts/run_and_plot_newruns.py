#!/usr/bin/env python3
"""
run_and_plot_newruns.py
=======================

Pipeline for the four reactions in runs/newruns/:

  1. Run FRESCO on each multi-energy .in file (skips if .out already exists).
  2. Parse the multi-energy output at the selected lab energies.
  3. Save individual figures (one per reaction, all selected energies overlaid).
  4. Save combined figures (grid of all four reactions).

Each reaction subdirectory must contain exactly one .in file.
FRESCO writes all 80 energies into a single .out file; this script
extracts only the eight energies listed in TARGET_MEV.

Usage
-----
    # run FRESCO and plot
    python3 scripts/run_and_plot_newruns.py --fresco /path/to/fresco

    # plot from existing .out files (no FRESCO call)
    python3 scripts/run_and_plot_newruns.py --no-run

    # point at a different newruns root
    python3 scripts/run_and_plot_newruns.py --newruns-dir runs/newruns --fresco fresco
"""

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TARGET_MEV = [5, 100, 150, 200, 250, 300, 350, 400]
COLOURS = plt.cm.turbo(np.linspace(0.05, 0.95, len(TARGET_MEV)))

# Matches "LABORATORY <proj> ENERGY =  100.0000  MeV." lines in FRESCO output.
_EHEAD = re.compile(
    r"LABORATORY\s+\S+\s+ENERGY\s*=\s*([0-9.]+)\s+MeV",
    re.IGNORECASE,
)

# Matches per-angle cross-section lines (same format as single-energy output).
_XS_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s+deg\.:"
    r"\s+X-S\s*=\s*([0-9.+\-Ee*]+)\s+mb/sr,\s*\n"
    r"\+\s*/R\s*=\s*([0-9.+\-Ee*]+)",
    re.MULTILINE,
)


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def run_fresco(fresco_exe: str, in_path: Path, out_path: Path) -> None:
    """Run FRESCO on in_path, writing stdout to out_path."""
    scratch = tempfile.mkdtemp(prefix="fresco_")
    try:
        deck = in_path.read_text(errors="ignore")
        with open(out_path, "w") as fout:
            result = subprocess.run(
                [fresco_exe],
                input=deck,
                stdout=fout,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=scratch,
            )
        if result.returncode != 0:
            print(f"    [warn] FRESCO exited {result.returncode} for {in_path.name}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def parse_multi_energy(out_path: Path) -> dict[float, tuple]:
    """
    Parse a multi-energy FRESCO output file.

    Returns {energy_mev: (thetas, xs_mb_sr, ratios)} for every energy block
    found in the file.
    """
    text = out_path.read_text(errors="ignore")

    # Split the file into per-energy blocks using the energy header as delimiter.
    boundaries = [(m.start(), float(m.group(1))) for m in _EHEAD.finditer(text)]
    if not boundaries:
        return {}

    blocks: dict[float, tuple] = {}
    for i, (start, energy) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end]

        thetas, xs_list, ratios = [], [], []
        for angle, xsval, rval in _XS_RE.findall(chunk):
            thetas.append(float(angle))
            xs_list.append(_safe_float(xsval))
            ratios.append(_safe_float(rval))

        if thetas:
            blocks[energy] = (thetas, xs_list, ratios)

    return blocks


def select_energies(blocks: dict[float, tuple]) -> dict[int, tuple]:
    """Pick the TARGET_MEV energies from the parsed blocks (nearest match)."""
    available = sorted(blocks.keys())
    selected: dict[int, tuple] = {}
    for target in TARGET_MEV:
        closest = min(available, key=lambda e: abs(e - target), default=None)
        if closest is not None and abs(closest - target) < 1.0:
            selected[target] = blocks[closest]
        else:
            print(f"    [warn] no block near {target} MeV (available: {available[:5]}...)")
    return selected


def plot_reaction(reaction: str, data: dict[int, tuple], save_dir: Path) -> None:
    fig, (ax_xs, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(reaction.replace("_", " "), fontsize=13)

    for mev, colour in zip(TARGET_MEV, COLOURS):
        if mev not in data:
            continue
        thetas, xs, ratios = data[mev]
        label = f"{mev} MeV"

        valid_xs = [(t, x) for t, x in zip(thetas, xs) if not math.isnan(x) and x > 0]
        if valid_xs:
            t_arr, x_arr = zip(*valid_xs)
            ax_xs.semilogy(t_arr, x_arr, color=colour, label=label, linewidth=1.2)

        valid_r = [(t, r) for t, r in zip(thetas, ratios) if not math.isnan(r)]
        if valid_r:
            t_arr, r_arr = zip(*valid_r)
            ax_r.plot(t_arr, r_arr, color=colour, label=label, linewidth=1.2)

    for ax, ylabel in [
        (ax_xs, r"$d\sigma/d\Omega$ (mb/sr)"),
        (ax_r, r"$\sigma/\sigma_{\rm Ruth}$"),
    ]:
        ax.set_xlabel(r"$\theta_{\rm lab}$ (deg)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 180)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out = save_dir / f"{reaction}_scan.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved {out.name}")


def plot_combined(all_data: dict[str, dict], save_dir: Path, quantity: str) -> None:
    reactions = list(all_data.keys())
    n = len(reactions)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (reaction, data) in enumerate(all_data.items()):
        ax = axes_flat[idx]
        ax.set_title(reaction.replace("_", " "), fontsize=9)

        for mev, colour in zip(TARGET_MEV, COLOURS):
            if mev not in data:
                continue
            thetas, xs, ratios = data[mev]
            vals = xs if quantity == "xs" else ratios
            use_log = quantity == "xs"

            valid = [(t, v) for t, v in zip(thetas, vals)
                     if not math.isnan(v) and (v > 0 if use_log else True)]
            if not valid:
                continue
            t_arr, v_arr = zip(*valid)
            if use_log:
                ax.semilogy(t_arr, v_arr, color=colour, label=f"{mev} MeV", linewidth=1.0)
            else:
                ax.plot(t_arr, v_arr, color=colour, label=f"{mev} MeV", linewidth=1.0)

        ax.set_xlim(0, 180)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel(r"$\theta_{\rm lab}$ (deg)", fontsize=8)
        ylabel = r"$d\sigma/d\Omega$ (mb/sr)" if quantity == "xs" else r"$\sigma/\sigma_{\rm Ruth}$"
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=c, linewidth=1.5, label=f"{m} MeV")
        for m, c in zip(TARGET_MEV, COLOURS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(TARGET_MEV),
               fontsize=8, title="E_lab", title_fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out = save_dir / f"combined_{quantity}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved combined: {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--newruns-dir", default="runs/newruns",
                    help="Directory containing reaction subdirs (default: runs/newruns)")
    ap.add_argument("--fresco", default="fresco",
                    help="FRESCO executable (default: fresco)")
    ap.add_argument("--no-run", action="store_true",
                    help="Skip running FRESCO; parse existing .out files only")
    ap.add_argument("--out-dir", default=None,
                    help="Where to save plots (default: same as --newruns-dir)")
    args = ap.parse_args()

    newruns_root = Path(args.newruns_dir)
    save_dir = Path(args.out_dir) if args.out_dir else newruns_root
    save_dir.mkdir(parents=True, exist_ok=True)

    reaction_dirs = sorted(
        d for d in newruns_root.iterdir()
        if d.is_dir() and list(d.glob("*.in"))
    )
    if not reaction_dirs:
        print(f"No reaction subdirectories with .in files found under {newruns_root}")
        sys.exit(1)

    fresco_available = shutil.which(args.fresco) or Path(args.fresco).exists()

    all_data: dict[str, dict] = {}

    for rdir in reaction_dirs:
        reaction = rdir.name
        in_files = list(rdir.glob("*.in"))
        if len(in_files) != 1:
            print(f"[{reaction}] expected 1 .in file, found {len(in_files)} — skipping")
            continue
        in_path = in_files[0]
        out_path = in_path.with_suffix(".out")

        print(f"\n=== {reaction} ===")

        if not args.no_run:
            if out_path.exists():
                print(f"  .out exists, skipping FRESCO run  (delete to re-run)")
            elif not fresco_available:
                print(f"  [warn] FRESCO '{args.fresco}' not found — skipping run; "
                      f"use --fresco /path/to/fresco or --no-run")
            else:
                print(f"  running FRESCO on {in_path.name} ...")
                run_fresco(args.fresco, in_path, out_path)
                print(f"  done -> {out_path.name}")

        if not out_path.exists():
            print(f"  no .out file found — skipping plots")
            continue

        print(f"  parsing {out_path.name} ...")
        blocks = parse_multi_energy(out_path)
        if not blocks:
            print(f"  no energy blocks parsed — is the run complete?")
            continue

        print(f"  found {len(blocks)} energy blocks: "
              f"{sorted(blocks)[:4]} ... {sorted(blocks)[-2:]}")
        data = select_energies(blocks)
        if not data:
            print(f"  none of the target energies found — skipping plots")
            continue

        all_data[reaction] = data
        plot_reaction(reaction, data, save_dir)

    if all_data:
        print(f"\nBuilding combined plots ({len(all_data)} reactions)...")
        plot_combined(all_data, save_dir, quantity="xs")
        plot_combined(all_data, save_dir, quantity="ratio")

    print("\nDone.")


if __name__ == "__main__":
    main()
