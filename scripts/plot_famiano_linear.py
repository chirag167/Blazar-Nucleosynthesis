"""
plot_famiano_linear.py

Plot the abundance evolution from run_famiano.py output, same data and
species styling as plot_famiano.py, but with a LINEAR time axis instead of
log-log.

Why this exists
----------------
plot_famiano.py matches Famiano (2002) Fig. 3's log-log convention, which is
the right call for showing the full 1e5 s run at a glance. But on a log time
axis, any transient that is fast relative to the *total* plotted range --
e.g. a short-lived species (n, 13N, ...) decaying away while the jet is off
and climbing back to equilibrium over a few hundred/thousand seconds once it
switches back on -- gets visually compressed into what looks like a vertical
jump, even though the underlying step-by-step evolution is smooth. This
script re-plots the same CSV on a linear time axis (optionally zoomed to a
window via --tmin/--tmax) so those transients can be inspected directly
instead of relying on log-axis intuition.

The y-axis stays log (abundances still span many orders of magnitude), so
this is a log-linear (semilogy) plot, not fully linear.

Usage
-----
    python scripts/plot_famiano_linear.py [--csv outputs/abundance_history.csv]
                                          [--out outputs/famiano_abundance_evolution_linear.png]
                                          [--tmin T0] [--tmax T1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Colour / line-style map for known species -- kept identical to
# plot_famiano.py so the two figures are visually comparable.
# ---------------------------------------------------------------------------

_STYLE: dict = {
    "p":    dict(color="tab:blue",    ls="-",   lw=2.0, label=r"$^{1}$H  (p)"),
    "d":    dict(color="tab:orange",  ls="--",  lw=1.5, label=r"D  ($^{2}$H)"),
    "3He":  dict(color="tab:green",   ls="-.",  lw=1.5, label=r"$^{3}$He"),
    "4He":  dict(color="tab:red",     ls="-",   lw=2.0, label=r"$^{4}$He"),
    "7Li":  dict(color="tab:purple",  ls="--",  lw=2.0, label=r"$^{7}$Li"),
    "7Be":  dict(color="tab:brown",   ls=":",   lw=1.5, label=r"$^{7}$Be"),
    "6Li":  dict(color="tab:pink",    ls="-",   lw=1.5, label=r"$^{6}$Li"),
    "n":    dict(color="black",       ls=":",   lw=1.5, label=r"n"),
    "t":    dict(color="tab:cyan",    ls="-.",  lw=1.2, label=r"$^{3}$H  (t)"),
    "10B":  dict(color="olive",       ls="--",  lw=1.2, label=r"$^{10}$B"),
    "11B":  dict(color="darkolivegreen", ls="-.", lw=1.2, label=r"$^{11}$B"),
    "11C":  dict(color="peru",        ls="-",   lw=1.2, label=r"$^{11}$C"),
    "12C":  dict(color="sienna",      ls="--",  lw=1.2, label=r"$^{12}$C"),
    "13C":  dict(color="saddlebrown", ls="-.",  lw=1.0, label=r"$^{13}$C"),
    "13N":  dict(color="teal",        ls=":",   lw=1.2, label=r"$^{13}$N"),
    "14N":  dict(color="darkcyan",    ls="--",  lw=1.0, label=r"$^{14}$N"),
    "15N":  dict(color="steelblue",   ls="-.",  lw=1.0, label=r"$^{15}$N"),
    "15O":  dict(color="tomato",      ls=":",   lw=1.0, label=r"$^{15}$O"),
    "16O":  dict(color="firebrick",   ls="--",  lw=1.0, label=r"$^{16}$O"),
}

# Fallback colour cycle for species not in _STYLE
_FALLBACK_COLORS = plt.cm.tab20.colors  # type: ignore[attr-defined]


def _style_for(sp: str, fallback_idx: int) -> dict:
    if sp in _STYLE:
        return dict(_STYLE[sp])
    color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
    return dict(color=color, ls="-", lw=1.0, label=sp)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["step", "t_s"]).drop_duplicates("step").reset_index(drop=True)
    df = df.sort_values("t_s").reset_index(drop=True)
    return df


def _active_species(df: pd.DataFrame) -> list[str]:
    """Return species columns that are non-zero at any recorded timestep."""
    skip = {"step", "t_s", "delta_m_over_m0"}
    active = []
    for col in df.columns:
        if col in skip:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if (vals > 0).any():
            active.append(col)
    return active


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_abundances(
    df: pd.DataFrame,
    out_path: Path,
    tmin: float | None = None,
    tmax: float | None = None,
) -> None:
    # Linear time axis: t=0 is well-defined (unlike log), so it's kept.
    # --tmin/--tmax let you zoom into one transition instead of squinting at
    # the full 1e5 s range.
    df = df[df["t_s"] >= 0].copy().reset_index(drop=True)
    if tmin is not None:
        df = df[df["t_s"] >= tmin]
    if tmax is not None:
        df = df[df["t_s"] <= tmax]
    df = df.reset_index(drop=True)
    t_s = df["t_s"].values

    species = _active_species(df)

    # --- Derive ΔM/M₀ conversion factor from the CSV column ---
    alpha: float | None = None
    if "delta_m_over_m0" in df.columns:
        dm = pd.to_numeric(df["delta_m_over_m0"], errors="coerce")
        mask = (df["t_s"] > 0) & (dm > 0)
        if mask.any():
            alpha = float((dm[mask] / df["t_s"][mask]).median())

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalise to hydrogen (Y/H), matching Famiano (2002) Fig. 3 y-axis convention.
    y_H_raw = pd.to_numeric(df["p"], errors="coerce").values.astype(float)
    y_H = np.where(y_H_raw > 0, y_H_raw, np.nan)

    fallback_idx = 0
    for sp in species:
        if sp == "p":
            continue  # Y_p / Y_H = 1 everywhere; omitted as in Famiano figs.
        y = pd.to_numeric(df[sp], errors="coerce").values.astype(float)
        y_over_H = y / y_H
        y_pos = np.where(y_over_H > 0, y_over_H, np.nan)
        if np.all(np.isnan(y_pos)):
            continue
        kw = _style_for(sp, fallback_idx)
        if sp not in _STYLE:
            fallback_idx += 1
        ax.plot(t_s, y_pos, **kw)

    # x-axis stays linear (the whole point of this script); y-axis stays log.
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)  [linear]", fontsize=13)
    ax.set_ylabel(r"$Y_i\,/\,Y_\mathrm{H}$", fontsize=13)

    # Tighten y-limits: smallest plotted value (within the window) floored at 1e-20
    finite_vals = []
    for sp in species:
        if sp == "p":
            continue
        y = pd.to_numeric(df[sp], errors="coerce").values.astype(float)
        y_over_H = y / y_H_raw
        pos = y_over_H[y_over_H > 0]
        if pos.size:
            finite_vals.append(pos.min())
    if finite_vals:
        ylo = max(min(finite_vals) * 0.1, 1e-20)
        ax.set_ylim(bottom=ylo)

    ax.legend(fontsize=9, loc="lower left", ncol=2, framealpha=0.7)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    # --- Top x-axis: ΔM/M₀ (still a valid linear rescaling of a linear axis) ---
    if alpha is not None and alpha > 0:
        ax2 = ax.secondary_xaxis(
            "top",
            functions=(lambda t: alpha * t, lambda dm: dm / alpha),
        )
        ax2.set_xlabel(r"$\Delta M\,/\,M_0$", fontsize=13)

    window_note = ""
    if tmin is not None or tmax is not None:
        lo = f"{tmin:.3e}" if tmin is not None else "start"
        hi = f"{tmax:.3e}" if tmax is not None else "end"
        window_note = f"  |  window: t = {lo} to {hi} s"

    fig.suptitle(
        r"Blazar Jet Nucleosynthesis — Famiano (2002) Fig. 3 (Model B), linear time" "\n"
        r"BLR cloud  $n_0 = 10^{11}$ cm$^{-3}$,  $T = 10^4$ K;"
        r"  Jet: p + $^4$He at 100 MeV/nucleon" + window_note,
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # --- Console summary ---
    print("\nWindow summary:")
    if len(df) == 0:
        print("  (no rows in the selected t range)")
        return
    first, last = df.iloc[0], df.iloc[-1]
    print(f"  t_start = {first['t_s']:.3e} s   t_end = {last['t_s']:.3e} s   rows = {len(df)}")
    for sp in species:
        y0 = float(pd.to_numeric(first[sp], errors="coerce"))
        yf = float(pd.to_numeric(last[sp], errors="coerce"))
        if y0 > 0:
            print(f"  {sp:6s}: Y_0 = {y0:.3e}  Y_f = {yf:.3e}  ratio = {yf/y0:.3e}")
        elif yf > 0:
            print(f"  {sp:6s}: Y_0 =   0        Y_f = {yf:.3e}  (produced in this window)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Famiano abundance history on a linear time axis"
    )
    parser.add_argument(
        "--csv",
        default=str(_ROOT / "outputs" / "abundance_history.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(_ROOT / "outputs" / "famiano_abundance_evolution_linear.png"),
    )
    parser.add_argument(
        "--tmin", type=float, default=None,
        help="Lower time bound (s) to zoom into, e.g. around one jet transition.",
    )
    parser.add_argument(
        "--tmax", type=float, default=None,
        help="Upper time bound (s) to zoom into.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.path.insert(0, str(_ROOT))
    df = _read_history(Path(args.csv))
    plot_abundances(df, Path(args.out), tmin=args.tmin, tmax=args.tmax)
