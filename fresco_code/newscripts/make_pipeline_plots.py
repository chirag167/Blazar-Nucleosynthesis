#!/usr/bin/env python3
"""
Small, textless plots for the 5 Computational Pipeline steps on the poster.

Each one is real data pulled straight from the file that step actually
produces or consumes, normalized to fit inside a badge circle with no axes,
ticks, or labels -- pure decoration, not a figure meant to be read on its
own.

  1. Gather Experimental Cross Sections  -> data/CrossSections/pd_total_cross_sections.csv
  2. Generate Computational Cross Sections -> fresco_code/newruns/he4_p_d_he3/he4_p_d_he3_total_xs.csv
  3. Energy-Bin Distributions             -> fresco_code/newruns/bins/p_d_pn_p_bins.csv
  4. Abundance Equation Source Terms      -> outputs/abundance_history.csv (d[7Li]/dt)
  5. Total Abundance Evolution            -> outputs/abundance_history.csv (Y/Y_H)

Output: outputs/pipeline_plots/{01..05}_<step>.png  (+ a contact-sheet preview)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "outputs" / "pipeline_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAROON = "#500000"
MAROON_LIGHT = "#9b3a3a"
GRAY = "#8a8a8a"
LO, HI = -0.8, 0.8


def norm(arr, log=False, lo=LO, hi=HI, floor=1e-30):
    a = np.asarray(arr, dtype=float)
    if log:
        a = np.log10(np.clip(a, floor, None))
    amin, amax = np.nanmin(a), np.nanmax(a)
    if amax - amin < 1e-12:
        return np.full_like(a, (lo + hi) / 2)
    return lo + (a - amin) / (amax - amin) * (hi - lo)


def new_canvas():
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=250)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    badge = Circle((0, 0), 1.0, facecolor=MAROON, alpha=0.07,
                   edgecolor=MAROON, lw=2.2, zorder=0)
    ax.add_patch(badge)
    clip = Circle((0, 0), 0.97, transform=ax.transData)
    return fig, ax, clip


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {path}")


# 1. Gather Experimental Cross Sections -> raw literature p+d compilation
def plot_gather():
    df = pd.read_csv(ROOT / "data/CrossSections/pd_total_cross_sections.csv")
    fig, ax, clip = new_canvas()
    x = norm(df["E_MeV"], log=True)
    y = norm(df["sigma_mb"], log=True)
    sc = ax.scatter(x, y, s=22, color=MAROON, alpha=0.75, linewidths=0, zorder=2)
    sc.set_clip_path(clip)
    save(fig, "01_gather_experimental_cross_sections.png")


# 2. Generate Computational Cross Sections -> FRESCO DWBA excitation function
def plot_fresco():
    df = pd.read_csv(ROOT / "fresco_code/newruns/he4_p_d_he3/he4_p_d_he3_total_xs.csv")
    df = df[df["source"] == "fresco"].sort_values("E_MeV")  # drop the literature anchor points
    fig, ax, clip = new_canvas()
    x = norm(df["E_MeV"])
    y = norm(df["sigma_mb"], log=True)
    ln, = ax.plot(x, y, color=MAROON, lw=3.5, solid_capstyle="round", zorder=2)
    ln.set_clip_path(clip)
    save(fig, "02_generate_computational_cross_sections.png")


# 3. Calculate Energy-Bin Total Cross Section Distributions -> the 5 MeV bin table
def plot_bins():
    df = pd.read_csv(ROOT / "fresco_code/newruns/bins/p_d_pn_p_bins.csv")
    df = df[df["reaction"] == "p_d_pn_p"].sort_values("E_MeV")
    group_size = 8  # 8 x 5 MeV bins per bar -> 10 bars
    grouped = df["sigma_mb"].groupby(df.index // group_size).mean()
    fig, ax, clip = new_canvas()
    x = norm(np.arange(len(grouped)))
    heights = norm(grouped.values, lo=0.0, hi=1.5)
    w = (x[1] - x[0]) * 0.7
    for xi, h in zip(x, heights):
        r = Rectangle((xi - w / 2, -0.8), w, h, facecolor=MAROON, alpha=0.85,
                      edgecolor=MAROON, lw=1.0, zorder=2)
        r.set_clip_path(clip)
        ax.add_patch(r)
    save(fig, "03_energy_bin_distributions.png")


# 4. Compute Abundance Equation Source Terms -> d[7Li]/dt from the running network
def plot_source_terms():
    df = pd.read_csv(ROOT / "outputs/abundance_history.csv")
    t, Y = df["t_s"].values, df["7Li"].values
    dydt = np.gradient(Y, t)
    mask = t > 0
    fig, ax, clip = new_canvas()
    x = norm(t[mask], log=True)
    y = norm(dydt[mask], log=True)
    ln, = ax.plot(x, y, color=MAROON, lw=3.5, solid_capstyle="round", zorder=2)
    ln.set_clip_path(clip)
    save(fig, "04_abundance_source_terms.png")


# 5. Compute Total Abundance Evolution -> Y/Y_H growth curves for enhanced species
def plot_evolution():
    df = pd.read_csv(ROOT / "outputs/abundance_history.csv")
    t = df["t_s"].values
    fig, ax, clip = new_canvas()
    mask = t > 0
    x = norm(t[mask], log=True)
    for col, color, lw in [("7Li", MAROON, 3.5), ("7Be", MAROON_LIGHT, 2.8),
                            ("10B", GRAY, 2.2)]:
        ratio = df[col].values[mask] / df["p"].values[mask]
        y = norm(ratio, log=True, floor=1e-30)
        ln, = ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=2)
        ln.set_clip_path(clip)
    save(fig, "05_total_abundance_evolution.png")


def make_contact_sheet():
    from matplotlib import image as mpimg
    files = sorted(OUT_DIR.glob("0*.png"))
    fig, axes = plt.subplots(1, len(files), figsize=(2.4 * len(files), 2.6), dpi=200)
    for ax, f in zip(axes, files):
        ax.imshow(mpimg.imread(f))
        ax.axis("off")
        ax.set_title(f.stem.split("_", 1)[1].replace("_", " "), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "contact_sheet.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'contact_sheet.png'}")


def main():
    plot_gather()
    plot_fresco()
    plot_bins()
    plot_source_terms()
    plot_evolution()
    make_contact_sheet()


if __name__ == "__main__":
    main()
