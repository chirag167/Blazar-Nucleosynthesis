"""
plot_species.py

Plot a single species' raw abundance vs. time from run_famiano.py's
abundance_history.csv, with every recorded row shown as a visible marker.

Why this exists
----------------
plot_famiano.py / plot_famiano_linear.py overlay many species on a shared
log-y axis, which is the right call for a full-run overview but hides two
things that matter when you're trying to verify a fix rather than get an
overview:
  1. Whether the data actually contains points through a fast transient, or
     whether write_every_n_steps skipped over it (markers make missing
     resolution visually obvious -- a gap between dots -- instead of letting
     a smooth interpolated line paper over it).
  2. The real shape of a rise/decay in LINEAR abundance, without a log-y
     axis exaggerating the approach to zero into what looks like a cliff.

Defaults to a linear-linear plot of one species (n by default) with markers
at every CSV row; --logy switches the y-axis to log if you need to see a
tiny residual tail near zero.

Also supports --panels: a full run (e.g. 1e5 s) will always make any single
jet transition look like a vertical cliff in a single plot, simply because
the transient (a couple thousand seconds) is a tiny fraction of the total
plotted width -- that's arithmetic, not a bug, and no amount of zooming into
ONE window fixes how the OTHER transitions look. --panels sidesteps that by
reading the jet on/off boundaries straight out of config/run.json and laying
out one zoomed subplot per boundary, all in one image, so every transition
can be checked for smoothness at once instead of one manual --tmin/--tmax
call at a time.

Usage
-----
    python scripts/plot_species.py [--species n]
                                    [--csv outputs/abundance_history.csv]
                                    [--out outputs/species_n.png]
                                    [--tmin T0] [--tmax T1]
                                    [--logy]

    python scripts/plot_species.py --species n --panels
                                    [--run-config config/run.json]
                                    [--window 3000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def _read_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["step", "t_s"]).drop_duplicates("step").reset_index(drop=True)
    df = df.sort_values("t_s").reset_index(drop=True)
    return df


def _jet_boundaries_from_config(run_cfg_path: Path) -> list[float]:
    """Mirror run_famiano.py's jet_intervals parsing to find on/off edges."""
    run_cfg = json.loads(run_cfg_path.read_text())
    time_cfg = run_cfg["time"]
    if "jet_intervals" in time_cfg:
        intervals = [(float(a), float(b)) for a, b in time_cfg["jet_intervals"]]
    elif "jet_stop_s" in time_cfg:
        intervals = [(float(time_cfg["t0_s"]), float(time_cfg["jet_stop_s"]))]
    else:
        return []
    return sorted({b for on_s, off_s in intervals for b in (on_s, off_s)})


def plot_species(
    df: pd.DataFrame,
    species: str,
    out_path: Path,
    tmin: float | None = None,
    tmax: float | None = None,
    logy: bool = False,
) -> None:
    if species not in df.columns:
        raise SystemExit(
            f"'{species}' is not a column in this CSV. Available species: "
            f"{[c for c in df.columns if c not in ('step', 't_s', 'delta_m_over_m0')]}"
        )

    df = df[df["t_s"] >= 0].copy()
    if tmin is not None:
        df = df[df["t_s"] >= tmin]
    if tmax is not None:
        df = df[df["t_s"] <= tmax]
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise SystemExit("No rows in the selected t range.")

    t_s = df["t_s"].values
    y = pd.to_numeric(df[species], errors="coerce").values

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Every recorded row is drawn as a marker (not just an interpolated
    # line) so gaps in resolution -- e.g. write_every_n_steps skipping over
    # a fast transient -- are visible as widely-spaced dots joined by a
    # long straight segment, rather than hidden.
    ax.plot(t_s, y, "-o", color="black", ms=4, lw=1, mfc="tab:red", mec="black", mew=0.5)

    if logy:
        ax.set_yscale("log")
        pos = y[y > 0]
        if pos.size:
            ax.set_ylim(bottom=max(pos.min() * 0.5, 1e-25))

    ax.set_xlabel("Time (s)  [linear]", fontsize=12)
    ax.set_ylabel(f"Y({species})  " + ("[log]" if logy else "[linear]"), fontsize=12)

    window_note = ""
    if tmin is not None or tmax is not None:
        lo = f"{tmin:.4g}" if tmin is not None else "start"
        hi = f"{tmax:.4g}" if tmax is not None else "end"
        window_note = f"  (t = {lo} to {hi} s)"
    ax.set_title(f"Species '{species}' vs. time{window_note}  --  {len(df)} recorded rows", fontsize=12)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    print(f"  rows plotted : {len(df)}")
    print(f"  t range      : {t_s[0]:.4g} to {t_s[-1]:.4g} s")
    print(f"  Y range      : {pd.Series(y).min():.4g} to {pd.Series(y).max():.4g}")


def plot_species_panels(
    df: pd.DataFrame,
    species: str,
    out_path: Path,
    boundaries: list[float],
    window: float,
    logy: bool = False,
) -> None:
    if species not in df.columns:
        raise SystemExit(
            f"'{species}' is not a column in this CSV. Available species: "
            f"{[c for c in df.columns if c not in ('step', 't_s', 'delta_m_over_m0')]}"
        )
    if not boundaries:
        raise SystemExit("No jet transitions found in the run config -- nothing to panel.")

    n = len(boundaries)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=not logy)
    if n == 1:
        axes = [axes]

    t_full = df["t_s"].values
    y_full = pd.to_numeric(df[species], errors="coerce").values

    for ax, b in zip(axes, boundaries):
        mask = (t_full >= b - window) & (t_full <= b + window)
        t_w, y_w = t_full[mask], y_full[mask]
        ax.plot(t_w, y_w, "-o", color="black", ms=3.5, lw=1, mfc="tab:red", mec="black", mew=0.4)
        ax.axvline(b, color="gray", ls="--", lw=1, alpha=0.7)
        if logy:
            ax.set_yscale("log")
            pos = y_w[y_w > 0]
            if pos.size:
                ax.set_ylim(bottom=max(pos.min() * 0.5, 1e-25))
        ax.set_title(f"t = {b:.4g} s  ({len(t_w)} rows)", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, ls=":", lw=0.4, alpha=0.6)

    axes[0].set_ylabel(f"Y({species})  " + ("[log]" if logy else "[linear]"), fontsize=11)
    fig.suptitle(
        f"Species '{species}' zoomed around every jet transition (±{window:.4g} s each)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    print(f"  transitions panelled: {boundaries}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one species' abundance vs. time, linear axes, every row marked")
    parser.add_argument("--species", default="n")
    parser.add_argument("--csv", default=str(_ROOT / "outputs" / "abundance_history.csv"))
    parser.add_argument("--out", default=None, help="Defaults to outputs/species_<name>.png")
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--logy", action="store_true", help="Use a log y-axis instead of linear.")
    parser.add_argument(
        "--panels", action="store_true",
        help="Instead of one plot, lay out one zoomed subplot per jet on/off transition.",
    )
    parser.add_argument("--run-config", default=str(_ROOT / "config" / "run.json"))
    parser.add_argument("--window", type=float, default=3000.0, help="Half-width (s) of each panel, --panels only.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = _read_history(Path(args.csv))
    if args.panels:
        out_path = Path(args.out) if args.out else _ROOT / "outputs" / f"species_{args.species}_panels.png"
        boundaries = _jet_boundaries_from_config(Path(args.run_config))
        plot_species_panels(df, args.species, out_path, boundaries, args.window, logy=args.logy)
    else:
        out_path = Path(args.out) if args.out else _ROOT / "outputs" / f"species_{args.species}.png"
        plot_species(df, args.species, out_path, tmin=args.tmin, tmax=args.tmax, logy=args.logy)
