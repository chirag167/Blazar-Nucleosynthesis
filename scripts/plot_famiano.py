"""
plot_famiano.py

Plot the abundance evolution from run_famiano.py output.

Produces a single-panel log–log figure matching Famiano (2002) style:
  - Bottom x-axis : time [yr]          (log scale)
  - Top x-axis    : ΔM/M₀             (log scale, derived from CSV column)
  - y-axis        : abundance Yᵢ       (log scale)

Only species that are non-zero at any recorded timestep are plotted.

Usage
-----
    python scripts/plot_famiano.py [--csv outputs/abundance_history.csv]
                                   [--out outputs/famiano_abundance_evolution.png]
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
# Colour / line-style map for known species
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
    """Return species columns that are non-zero at any recorded time."""
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

def plot_abundances(df: pd.DataFrame, out_path: Path) -> None:
    # Drop t = 0 (undefined on log scale) and rows with non-positive time
    df = df[df["t_s"] > 0].copy().reset_index(drop=True)
    t_s = df["t_s"].values

    species = _active_species(df)

    # --- Derive ΔM/M₀ conversion factor from the CSV column ---
    # alpha = ΔM/M₀ per second  (constant: ΔM/M₀ = alpha × t_s)
    alpha: float | None = None
    if "delta_m_over_m0" in df.columns:
        dm = pd.to_numeric(df["delta_m_over_m0"], errors="coerce")
        mask = (df["t_s"] > 0) & (dm > 0)
        if mask.any():
            alpha = float((dm[mask] / t_s[mask]).median())

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(10, 6))

    fallback_idx = 0
    for sp in species:
        y = pd.to_numeric(df[sp], errors="coerce").values.astype(float)
        y_pos = np.where(y > 0, y, np.nan)
        if np.all(np.isnan(y_pos)):
            continue
        kw = _style_for(sp, fallback_idx)
        if sp not in _STYLE:
            fallback_idx += 1
        ax.plot(t_s, y_pos, **kw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel(r"Abundance $Y_i$  (per baryon)", fontsize=13)

    # Tighten y-limits: smallest plotted value floored at 1e-20
    finite_vals = []
    for sp in species:
        y = pd.to_numeric(df[sp], errors="coerce").values.astype(float)
        pos = y[y > 0]
        if pos.size:
            finite_vals.append(pos.min())
    if finite_vals:
        ylo = max(min(finite_vals) * 0.1, 1e-20)
        ax.set_ylim(bottom=ylo)

    ax.legend(fontsize=9, loc="lower left", ncol=2, framealpha=0.7)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    # --- Top x-axis: ΔM/M₀ ---
    if alpha is not None and alpha > 0:
        ax2 = ax.secondary_xaxis(
            "top",
            functions=(lambda t: alpha * t, lambda dm: dm / alpha),
        )
        ax2.set_xlabel(r"$\Delta M\,/\,M_0$", fontsize=13)

    fig.suptitle(
        "Blazar Jet Nucleosynthesis — Famiano (2002) Model B\n"
        r"(BLR cloud, $n = 10^{11}$ cm$^{-3}$, $T = 10^4$ K;"
        r" Jet: p + $^4$He at 100 MeV/nucleon, Group-2 reactions)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # --- Console summary ---
    print("\nFinal state summary:")
    last = df.iloc[-1]
    print(f"  t_final      = {last['t_s']:.3e} s")
    if "delta_m_over_m0" in df.columns:
        print(f"  ΔM/M₀ final  = {last['delta_m_over_m0']:.3e}")
    for sp in species:
        y0 = float(pd.to_numeric(df[sp], errors="coerce").iloc[0])
        yf = float(pd.to_numeric(last[sp], errors="coerce"))
        if y0 > 0:
            print(f"  {sp:6s}: Y_0 = {y0:.3e}  Y_f = {yf:.3e}  ratio = {yf/y0:.3e}")
        elif yf > 0:
            print(f"  {sp:6s}: Y_0 =   0        Y_f = {yf:.3e}  (produced)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Famiano abundance history")
    parser.add_argument(
        "--csv",
        default=str(_ROOT / "outputs" / "abundance_history.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(_ROOT / "outputs" / "famiano_abundance_evolution.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.path.insert(0, str(_ROOT))
    df = _read_history(Path(args.csv))
    plot_abundances(df, Path(args.out))
