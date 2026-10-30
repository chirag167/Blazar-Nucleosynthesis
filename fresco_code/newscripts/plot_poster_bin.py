#!/usr/bin/env python3
"""
Poster figures: single-reaction cross section vs energy, presentation quality.

Each reaction's newruns/bins/<reaction>_bins.csv is rendered as one large
plot, colour-coded by bin status (measured data, log-linear interpolation,
FRESCO-scaled, FRESCO-extrapolated) with a shaded band over any energy gap.

Output: newruns/bins/<reaction>_poster.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

NEWRUNS = Path(__file__).parent.parent / "newruns"
BINS_DIR = NEWRUNS / "bins"
COLOR_GRID = "#c9c9c9"

# Style + legend label per bin status (colours match plot_bins.py for
# cross-poster consistency).
STYLE = {
    "data":         dict(color="#2ca02c", marker="D", ms=10, ls="",
                          label="Measured cross section"),
    "interpolated": dict(color="#7570b3", marker="", ms=0, ls="-",
                          label="Log-linear interpolation"),
    "scaled":       dict(color="#1f77b4", marker="o", ms=7, ls="-",
                          label="FRESCO, scaled to data"),
    "extrapolated": dict(color="#ff7f0e", marker="s", ms=6, ls="--",
                          label="FRESCO, extrapolated beyond data"),
}
STATUS_ORDER = ["scaled", "extrapolated", "interpolated", "data"]

REACTIONS = [
    dict(
        name="p_d_pn_p",
        title=r"p + d $\to$ p + n + p Breakup Cross Section",
        subtitle=r"$\mathit{5\ MeV\ energy\ bins,\ 5\ to\ 400\ MeV\ -\ complete\ coverage,\ no\ gaps}$",
    ),
    dict(
        name="he4_he4_n_be7",
        title=r"$^4$He + $^4$He $\to$ n + $^7$Be Neutron-Stripping Cross Section",
        subtitle=r"$\mathit{FRESCO\ DWBA,\ scaled\ to\ data\ 65-155\ MeV,\ extrapolated\ 160-400\ MeV}$",
    ),
]


def gap_intervals(df):
    """Contiguous [start, end] energy ranges where status == gap."""
    gaps = df[df["status"] == "gap"]["E_MeV"].sort_values().tolist()
    if not gaps:
        return []
    step = 5.0
    intervals = []
    lo = hi = gaps[0]
    for e in gaps[1:]:
        if abs(e - hi - step) < 1e-6:
            hi = e
        else:
            intervals.append((lo, hi))
            lo = hi = e
    intervals.append((lo, hi))
    return intervals


def plot_reaction(cfg):
    reaction = cfg["name"]
    csv_path = BINS_DIR / f"{reaction}_bins.csv"
    out_path = BINS_DIR / f"{reaction}_poster.png"

    df = pd.read_csv(csv_path)
    df = df[df["reaction"] == reaction].sort_values("E_MeV")

    gaps = gap_intervals(df)

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Shade gap bands first, underneath everything else
    for lo, hi in gaps:
        ax.axvspan(lo - 2.5, hi + 2.5, facecolor="#e0e0e0", alpha=0.6, hatch="//",
                   edgecolor="#b0b0b0", lw=0, zorder=0)
    plotted = df.dropna(subset=["sigma_mb"])

    for status in STATUS_ORDER:
        sub = plotted[plotted["status"] == status]
        if sub.empty:
            continue
        style = STYLE[status]
        if status == "data":
            ax.errorbar(sub["E_MeV"], sub["sigma_mb"], yerr=sub["d_sigma_mb"],
                        fmt=style["marker"], color=style["color"], ms=style["ms"],
                        mec="white", mew=1.2, ecolor=style["color"], elinewidth=1.5,
                        capsize=4, zorder=3, label=style["label"])
            for _, row in sub.iterrows():
                ax.annotate(f"{row['sigma_mb']:.0f} mb",
                            (row["E_MeV"], row["sigma_mb"]),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", fontsize=10, color=style["color"])
        else:
            ax.plot(sub["E_MeV"], sub["sigma_mb"], color=style["color"],
                    marker=style["marker"], ms=style["ms"], ls=style["ls"],
                    lw=1.8, zorder=2, label=style["label"])

    ax.set_yscale("log")
    ax.set_xlim(-10, 420)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 2.2)
    ax.set_xlabel("$E_{\\mathrm{lab}}$ (MeV)")
    ax.set_ylabel(r"$\sigma$ (mb)")
    ax.set_title(f"{cfg['title']}\n{cfg['subtitle']}", pad=14)

    # Annotate gap bands now that the y-range is final
    for lo, hi in gaps:
        ax.text((lo + hi) / 2, ymin * 1.15, "no data",
                ha="center", va="bottom", fontsize=10, color="#808080",
                style="italic")

    ax.grid(True, which="both", ls=":", lw=0.6, color=COLOR_GRID, alpha=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
    for cfg in REACTIONS:
        plot_reaction(cfg)


if __name__ == "__main__":
    main()
