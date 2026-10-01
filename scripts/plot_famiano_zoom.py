"""
plot_famiano_zoom.py

Same log-log Famiano (2002) Fig. 3 style figure as plot_famiano.py, but with
an inset panel zoomed into one jet off->on transition on a LINEAR time axis.

Why this exists
----------------
On the full-range log-time plot, a species that decays away while the jet is
off and climbs back to its quasi-equilibrium value once the jet resumes
(e.g. free neutrons, 7Li, 7Be) does so over a few hundred to a few thousand
seconds -- fast relative to the full 1e5 s run, so it renders as a near-
vertical jump. That's a plotting-resolution artifact, not a discontinuity in
the underlying physics: the step-by-step evolution (see
outputs/abundance_history.csv) is a smooth dip-and-recovery. This script
keeps the standard full-range panel and adds a zoomed linear-time inset
(mirroring plot_famiano_linear.py's --tmin/--tmax window) directly on the
same figure, with the zoomed region boxed on the main panel, so the
"jump" is visibly resolved as a smooth ramp instead.

The zoom window defaults to bracketing the first jet off-interval found in
config/run.json's time.jet_intervals (padded before/after so the pre-shutoff
plateau, the off-decay, and the on-recovery are all visible); pass
--zoom-tmin/--zoom-tmax to override.

Usage
-----
    python scripts/plot_famiano_zoom.py [--csv outputs/abundance_history.csv]
                                        [--out outputs/famiano_abundance_evolution_zoom.png]
                                        [--run-config config/run.json]
                                        [--zoom-tmin T0] [--zoom-tmax T1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_famiano import _STYLE, _style_for, _read_history, _active_species  # noqa: E402


def _default_zoom_window(run_config_path: Path) -> tuple[float, float] | None:
    """Bracket the first jet off-interval from config/run.json, if present."""
    if not run_config_path.exists():
        return None
    cfg = json.loads(run_config_path.read_text())
    intervals = cfg.get("time", {}).get("jet_intervals")
    if not intervals or len(intervals) < 2:
        return None
    # First gap: end of interval 0 -> start of interval 1.
    gap_start = float(intervals[0][1])
    gap_end = float(intervals[1][0])
    gap_len = gap_end - gap_start
    if gap_len <= 0:
        return None
    tmin = max(0.0, gap_start - gap_len)
    tmax = gap_end + 2.0 * gap_len
    return tmin, tmax


def plot_abundances_with_zoom(
    df,
    out_path: Path,
    zoom_tmin: float,
    zoom_tmax: float,
) -> None:
    df_full = df[df["t_s"] > 0].copy().reset_index(drop=True)
    t_full = df_full["t_s"].values
    species = _active_species(df_full)

    y_H_full_raw = df_full["p"].astype(float).values
    y_H_full = y_H_full_raw.copy()
    y_H_full[y_H_full <= 0] = float("nan")

    fig, ax = plt.subplots(figsize=(11, 7))

    fallback_idx = 0
    style_used: dict[str, dict] = {}
    for sp in species:
        if sp == "p":
            continue
        y = df_full[sp].astype(float).values
        y_over_H = y / y_H_full
        if not (y_over_H > 0).any():
            continue
        kw = _style_for(sp, fallback_idx)
        if sp not in _STYLE:
            fallback_idx += 1
        style_used[sp] = kw
        ax.plot(t_full, y_over_H, **kw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel(r"$Y_i\,/\,Y_\mathrm{H}$", fontsize=13)

    finite_vals = [
        v for sp in style_used
        for v in [(df_full[sp].astype(float).values / y_H_full_raw)]
        if (v > 0).any()
        for v in [v[v > 0].min()]
    ]
    if finite_vals:
        ax.set_ylim(bottom=max(min(finite_vals) * 0.1, 1e-20))

    ax.legend(fontsize=8.5, loc="lower left", ncol=2, framealpha=0.7)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    fig.suptitle(
        r"Blazar Jet Nucleosynthesis — Famiano (2002) Fig. 3 (Model B)" "\n"
        r"BLR cloud  $n_0 = 10^{11}$ cm$^{-3}$,  $T = 10^4$ K;"
        r"  Jet: p + $^4$He at 100 MeV/nucleon",
        fontsize=11,
    )

    # --- Zoomed inset: linear time axis over one jet off->on transition ---
    df_zoom = df_full[(df_full["t_s"] >= zoom_tmin) & (df_full["t_s"] <= zoom_tmax)]
    if len(df_zoom) >= 2:
        axins = ax.inset_axes([0.56, 0.08, 0.42, 0.40])
        t_zoom = df_zoom["t_s"].values
        y_H_zoom_raw = df_zoom["p"].astype(float).values
        y_H_zoom = y_H_zoom_raw.copy()
        y_H_zoom[y_H_zoom <= 0] = float("nan")

        zoom_finite = []
        for sp, kw in style_used.items():
            y = df_zoom[sp].astype(float).values
            y_over_H = y / y_H_zoom
            if not (y_over_H > 0).any():
                continue
            axins.plot(t_zoom, y_over_H, **kw)
            pos = (df_zoom[sp].astype(float).values / y_H_zoom_raw)
            pos = pos[pos > 0]
            if pos.size:
                zoom_finite.append(pos)

        axins.set_yscale("log")
        axins.set_xlim(zoom_tmin, zoom_tmax)
        if zoom_finite:
            import numpy as np
            lo = min(float(v.min()) for v in zoom_finite) * 0.5
            hi = max(float(v.max()) for v in zoom_finite) * 2.0
            axins.set_ylim(lo, hi)
        axins.tick_params(labelsize=7.5)
        axins.set_title(
            f"linear zoom: t = {zoom_tmin:.3e}–{zoom_tmax:.3e} s\n"
            "(dip-and-recover is smooth, not a jump)",
            fontsize=8,
        )
        axins.grid(True, which="both", ls=":", lw=0.3, alpha=0.5)

        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.4", lw=0.8)
    else:
        print(f"Warning: zoom window [{zoom_tmin}, {zoom_tmax}] has <2 rows; skipping inset.")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Famiano abundance history with a zoomed linear-time inset"
    )
    parser.add_argument("--csv", default=str(_ROOT / "outputs" / "abundance_history.csv"))
    parser.add_argument(
        "--out",
        default=str(_ROOT / "outputs" / "famiano_abundance_evolution_zoom.png"),
    )
    parser.add_argument("--run-config", default=str(_ROOT / "config" / "run.json"))
    parser.add_argument("--zoom-tmin", type=float, default=None)
    parser.add_argument("--zoom-tmax", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = _read_history(Path(args.csv))

    if args.zoom_tmin is not None and args.zoom_tmax is not None:
        zoom_window = (args.zoom_tmin, args.zoom_tmax)
    else:
        zoom_window = _default_zoom_window(Path(args.run_config))
        if zoom_window is None:
            raise SystemExit(
                "Could not determine a default zoom window from "
                f"{args.run_config} (need >=2 entries in time.jet_intervals); "
                "pass --zoom-tmin/--zoom-tmax explicitly."
            )

    plot_abundances_with_zoom(df, Path(args.out), zoom_window[0], zoom_window[1])
