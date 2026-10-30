#!/usr/bin/env python3
"""
Plot binned cross sections (sigma vs E) for all Group I reactions.

Reads newruns/bins/all_reactions_bins.csv and produces a 4x3 grid of
subplots — one per reaction — showing sigma_mb vs E_MeV on a log-y scale.
Bin status is shown by colour:
  scaled       -> blue  (FRESCO normalised to data within data range)
  extrapolated -> orange (FRESCO normalised, beyond data range)
  data         -> green  (experimental data point)
  interpolated -> purple (log-linear interpolation between data points)
  gap          -> grey dashed line at the x-axis (no value available)

Output: newruns/bins/bins_overview.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

NEWRUNS = Path(__file__).parent.parent / "newruns"
BINS_CSV = NEWRUNS / "bins" / "all_reactions_bins.csv"
OUT_PNG  = NEWRUNS / "bins" / "bins_overview.png"

# Colour/marker scheme keyed by status
STYLE = {
    "scaled":       dict(color="#1f77b4", marker="o", ls="-",  ms=4, lw=1.5, zorder=3),
    "extrapolated": dict(color="#ff7f0e", marker="s", ls="--", ms=3, lw=1.2, zorder=2),
    "data":         dict(color="#2ca02c", marker="D", ls="",   ms=5, lw=0,   zorder=4),
    "interpolated": dict(color="#9467bd", marker="",  ls="-",  ms=0, lw=1.5, zorder=2),
}

# Human-readable reaction labels (LaTeX)
LABELS = {
    # FRESCO reactions
    "he4_he4_n_be7":   r"$^4$He+$^4$He $\to$ n+$^7$Be",
    "he4_he4_p_li7":   r"$^4$He+$^4$He $\to$ p+$^7$Li",
    "he4_p_d_he3":     r"$^4$He+p $\to$ d+$^3$He",
    "li7_p_he4_he4":   r"$^7$Li+p $\to$ $^4$He+$^4$He",
    # Pre-existing data-only reactions
    "he4_he4_n_li7":   r"$^4$He+$^4$He $\to$ n+$^7$Li",
    "he4_p_p_he4":     r"$^4$He+p $\to$ p+$^4$He",
    "p_d_dp_n":        r"p+d $\to$ dp+n",
    "p_d_n_d":         r"p+d $\to$ n+d",
    "p_d_n_p":         r"p+d $\to$ n+p",
    "p_he4_p_he4":     r"p+$^4$He $\to$ p+$^4$He",
    "p_li7_he4_he4":   r"p+$^7$Li $\to$ $^4$He+$^4$He",
    # Meyer (1972) p+He4 breakup channels
    "he4_p_he3_np":    r"$^4$He+p $\to$ $^3$He+n+p",
    "he4_p_h3_2p":     r"$^4$He+p $\to$ $^3$H+2p",
    "he4_p_dd":        r"$^4$He+p $\to$ d+d+p",
    "he4_p_d_np":      r"$^4$He+p $\to$ d+n+2p",
    "he4_p_2n3p":      r"$^4$He+p $\to$ 2n+3p",
    # Meyer (1972) p+d inelastic
    "p_d_inel":        r"p+d $\to$ inelastic",
}

REACTION_ORDER = list(LABELS.keys())


def plot_reaction(ax, df_rxn, rxn):
    grid = np.arange(5.0, 405.0, 5.0)
    gap_mask = df_rxn["status"] == "gap"

    # Draw a faint grey band at the bottom to indicate gap coverage
    if gap_mask.any():
        gap_e = df_rxn.loc[gap_mask, "E_MeV"].values
        for e in gap_e:
            ax.axvline(e, color="#cccccc", lw=0.5, zorder=0)

    # Plot each non-gap status group
    for status, style in STYLE.items():
        sub = df_rxn[df_rxn["status"] == status].dropna(subset=["sigma_mb"])
        if sub.empty:
            continue
        kw = {k: v for k, v in style.items() if k != "zorder"}
        ax.plot(sub["E_MeV"], sub["sigma_mb"],
                zorder=style["zorder"], **kw)

        # Error bars where d_sigma is available
        esub = sub.dropna(subset=["d_sigma_mb"])
        if not esub.empty:
            ax.errorbar(esub["E_MeV"], esub["sigma_mb"],
                        yerr=esub["d_sigma_mb"],
                        fmt="none", ecolor=style["color"],
                        elinewidth=0.8, capsize=2, zorder=style["zorder"])

    ax.set_title(LABELS.get(rxn, rxn), fontsize=8, pad=3)
    ax.set_yscale("log")
    ax.set_xlim(0, 410)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)


def make_legend(fig):
    handles = []
    for status, style in STYLE.items():
        h = mlines.Line2D([], [],
                          color=style["color"],
                          marker=style.get("marker", ""),
                          linestyle=style["ls"] if style["ls"] else "None",
                          markersize=5,
                          label=status)
        handles.append(h)
    gap_h = mlines.Line2D([], [], color="#cccccc", lw=1.5, label="gap")
    handles.append(gap_h)
    fig.legend(handles=handles, loc="lower right",
               fontsize=8, framealpha=0.9, ncol=len(handles))


def main():
    df = pd.read_csv(BINS_CSV)

    n_rxn = len(REACTION_ORDER)
    ncols = 3
    nrows = (n_rxn + ncols - 1) // ncols   # ceil division

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, nrows * 3.2),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for i, rxn in enumerate(REACTION_ORDER):
        ax = axes_flat[i]
        df_rxn = df[df["reaction"] == rxn].copy()
        if df_rxn.empty:
            ax.set_visible(False)
            continue
        plot_reaction(ax, df_rxn, rxn)

    # Hide any unused subplot slots
    for j in range(len(REACTION_ORDER), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared axis labels
    fig.supxlabel("$E_{\\mathrm{lab}}$ (MeV)", fontsize=10)
    fig.supylabel(r"$\sigma$ (mb)", fontsize=10)
    fig.suptitle("Group I Reaction Cross Sections — 5 MeV Bins", fontsize=12, y=1.01)

    make_legend(fig)

    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
