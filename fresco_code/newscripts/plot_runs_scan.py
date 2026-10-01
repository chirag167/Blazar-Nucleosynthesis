#!/usr/bin/env python3
"""
plot_runs_scan.py
=================

For every reaction subdirectory under runs/ that has an outputs/ folder,
parse FRESCO .out files at the eight selected lab energies and produce:

  1. One figure per reaction  -- all selected energies overlaid.
  2. One combined figure      -- grid of per-reaction subplots.

Both dσ/dΩ (mb/sr, log scale) and ratio-to-Rutherford panels are generated.

Usage
-----
    python3 scripts/plot_runs_scan.py
    python3 scripts/plot_runs_scan.py --runs-dir runs/ --out-dir plots/
    python3 scripts/plot_runs_scan.py --reactions p_4He_d_3He 4He_p_d_3He
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TARGET_MEV = [5, 100, 150, 200, 250, 300, 350, 400]

# Colour cycle matched to TARGET_MEV length
COLOURS = plt.cm.turbo(np.linspace(0.05, 0.95, len(TARGET_MEV)))

# Maps reaction key -> (ejectile, residual) as FRESCO writes them in the output header.
# Used to locate the transfer block; elastic blocks (with /R lines) are ignored.
TRANSFER_CHANNEL = {
    "he4_p_d_he3":    ("d",   "3He"),
    "he4_he4_n_be7":  ("n",   "7Be"),
    "he4_he4_p_li7":  ("p",   "7Li"),
    "li7_p_he4_he4":  ("4He", "4He"),
}

# Matches "CROSS SECTIONS FOR OUTGOING <p1> & <p2> in state" headers.
_HEADER_RE = re.compile(
    r"CROSS SECTIONS FOR OUTGOING\s+(\S+)\s+&\s+(\S+)\s+in state",
    re.IGNORECASE,
)

# Matches a transfer angle line (no /R continuation required).
_TRANSFER_XS_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s+deg\.:\s+X-S\s*=\s*([0-9.+\-Ee*]+)\s+mb/sr,",
    re.MULTILINE,
)


def _float_or_nan(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_transfer_xs(path: Path, ejec: str, resid: str) -> tuple[list[float], list[float]]:
    """Return (theta_deg, xs_mb_sr) for the transfer block matching ejec & resid.

    Elastic blocks (which carry /R lines) are skipped automatically because we
    select by header rather than by the presence of a /R continuation.
    Returns empty lists if the block is absent or has no data (e.g. below threshold).
    """
    text = path.read_text(errors="ignore")
    headers = list(_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        if m.group(1) == ejec and m.group(2) == resid:
            start = m.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            chunk = text[start:end]
            thetas, xs = [], []
            for angle, xsval in _TRANSFER_XS_RE.findall(chunk):
                thetas.append(float(angle))
                xs.append(_float_or_nan(xsval))
            return thetas, xs
    return [], []


def find_output(outdir: Path, reaction: str, mev: int) -> Path | None:
    """Return the .out file for the given energy, or None if absent."""
    candidates = [
        outdir / f"{reaction}_{mev}.0MeV.out",
        outdir / f"{reaction}_{mev}MeV.out",
        outdir / f"{reaction}_{float(mev):g}MeV.out",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def plot_reaction(reaction: str, data: dict, save_dir: Path) -> None:
    """Individual figure for one reaction: dσ/dΩ (transfer only, log scale)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(reaction.replace("_", " "), fontsize=13)

    for mev, colour in zip(TARGET_MEV, COLOURS):
        if mev not in data:
            continue
        thetas, xs = data[mev]
        label = f"{mev} MeV"
        valid = [(t, x) for t, x in zip(thetas, xs) if not math.isnan(x) and x > 0]
        if valid:
            t_arr, x_arr = zip(*valid)
            ax.semilogy(t_arr, x_arr, color=colour, label=label, linewidth=1.2)

    ax.set_xlabel(r"$\theta_{\rm lab}$ (deg)")
    ax.set_ylabel(r"$d\sigma/d\Omega$ (mb/sr)")
    ax.set_xlim(0, 180)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out = save_dir / f"{reaction}_scan.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_combined(all_data: dict[str, dict], save_dir: Path) -> None:
    """Combined grid of dσ/dΩ (log scale) for all reactions."""
    reactions = list(all_data.keys())
    n = len(reactions)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (reaction, data) in enumerate(all_data.items()):
        ax = axes_flat[idx]
        ax.set_title(reaction.replace("_", " "), fontsize=7)

        for mev, colour in zip(TARGET_MEV, COLOURS):
            if mev not in data:
                continue
            thetas, xs = data[mev]
            valid = [(t, x) for t, x in zip(thetas, xs) if not math.isnan(x) and x > 0]
            if not valid:
                continue
            t_arr, v_arr = zip(*valid)
            ax.semilogy(t_arr, v_arr, color=colour, label=f"{mev}", linewidth=0.9)

        ax.set_xlim(0, 180)
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(labelsize=6)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if n > 0:
        handles = [
            plt.Line2D([0], [0], color=c, linewidth=1.5, label=f"{m} MeV")
            for m, c in zip(TARGET_MEV, COLOURS)
        ]
        fig.legend(handles=handles, loc="lower right", ncol=len(TARGET_MEV) // 2,
                   fontsize=7, title="E_lab", title_fontsize=7)

    fig.supxlabel(r"$\theta_{\rm lab}$ (deg)", fontsize=9)
    fig.supylabel(r"$d\sigma/d\Omega$ (mb/sr)", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = save_dir / "combined_xs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="newruns",
                    help="Root directory containing reaction subdirs (default: newruns/)")
    ap.add_argument("--out-dir", default=None,
                    help="Where to save plots (default: same as --runs-dir)")
    ap.add_argument("--reactions", nargs="*", default=None,
                    help="Limit to these reaction keys (default: all)")
    args = ap.parse_args()

    runs_root = Path(args.runs_dir)
    save_dir = Path(args.out_dir) if args.out_dir else runs_root
    save_dir.mkdir(parents=True, exist_ok=True)

    # discover reaction directories
    candidates = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and (d / "outputs").is_dir()
        and any((d / "outputs").glob("*.out"))
    )
    if args.reactions:
        candidates = [d for d in candidates if d.name in args.reactions]

    if not candidates:
        print(f"No reaction directories found under {runs_root}")
        return

    print(f"Found {len(candidates)} reactions.  Energies: {TARGET_MEV} MeV\n")

    all_data: dict[str, dict] = {}

    for rdir in candidates:
        reaction = rdir.name
        channel = TRANSFER_CHANNEL.get(reaction)
        if channel is None:
            print(f"  [{reaction}] no transfer channel defined — skipping")
            continue
        ejec, resid = channel
        outdir = rdir / "outputs"
        data: dict[int, tuple] = {}

        for mev in TARGET_MEV:
            path = find_output(outdir, reaction, mev)
            if path is None:
                print(f"  [{reaction}] missing {mev} MeV output — skipped")
                continue
            try:
                thetas, xs = parse_transfer_xs(path, ejec, resid)
                if not thetas:
                    continue  # below threshold or empty block — skip silently
                data[mev] = (thetas, xs)
            except Exception as e:
                print(f"  [{reaction}] error reading {path.name}: {e}")

        if not data:
            print(f"  [{reaction}] no transfer data — skipping plots")
            continue

        all_data[reaction] = data
        print(f"  {reaction}: {sorted(data)} MeV loaded")
        plot_reaction(reaction, data, save_dir)

    print(f"\nBuilding combined plot ({len(all_data)} reactions)...")
    plot_combined(all_data, save_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
