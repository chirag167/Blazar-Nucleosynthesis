"""
plot_helium.py

Plot only the helium species (3He, 4He) from the abundance history CSV.

Usage
-----
    python scripts/plot_helium.py [--csv outputs/abundance_history.csv]
                                   [--out outputs/helium_abundance_evolution.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

_STYLE = {
    "4He": dict(color="tab:red",   ls="-",  lw=2.2, label=r"$^{4}$He"),
}


def plot_helium(df: pd.DataFrame, out_path: Path) -> None:
    df = df[df["t_s"] > 0].copy().reset_index(drop=True)
    t_s = df["t_s"].values

    fig, ax = plt.subplots(figsize=(10, 6))

    for sp, kw in _STYLE.items():
        y = pd.to_numeric(df[sp], errors="coerce").values.astype(float)
        y_pos = np.where(y > 0, y, np.nan)
        if np.all(np.isnan(y_pos)):
            continue
        ax.plot(t_s, y_pos, **kw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel(r"Abundance $Y_i$", fontsize=13)
    ax.set_title(r"$^{4}$He Abundance Evolution", fontsize=13)
    ax.legend(fontsize=11, loc="best", framealpha=0.7)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot helium abundance history")
    parser.add_argument("--csv", default=str(_ROOT / "outputs" / "abundance_history.csv"))
    parser.add_argument("--out", default=str(_ROOT / "outputs" / "4he_abundance_evolution.png"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["step", "t_s"]).drop_duplicates("step").sort_values("t_s").reset_index(drop=True)
    plot_helium(df, Path(args.out))
