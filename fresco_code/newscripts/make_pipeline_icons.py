#!/usr/bin/env python3
"""
Small flat-line icons for the 5 Computational Pipeline steps on the poster.

Pure decoration: one icon per step, same visual weight (maroon stroke on a
faint maroon badge circle, transparent background) so they drop in next to
the existing [1]-[5] step boxes without clashing with the poster's palette.

Output: outputs/pipeline_icons/{01..05}_<step>.png  (+ a contact-sheet preview)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Polygon
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "pipeline_icons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAROON = "#500000"
MAROON_FAINT = "#500000"
LW = 4.5


def new_canvas():
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=250)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    badge = Circle((0, 0), 1.0, facecolor=MAROON_FAINT, alpha=0.07,
                   edgecolor=MAROON, lw=2.2, zorder=0)
    ax.add_patch(badge)
    return fig, ax


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {path}")


# 1. Gather Experimental Cross Sections -> data points + magnifying glass
def icon_gather():
    fig, ax = new_canvas()
    pts = np.array([[-0.55, -0.35], [-0.15, 0.05], [0.15, -0.5],
                     [0.55, 0.1], [-0.5, 0.45], [0.5, -0.55]])
    ax.scatter(pts[:, 0], pts[:, 1], s=40, color=MAROON, zorder=2)
    glass_c = (0.05, 0.05)
    ax.add_patch(Circle(glass_c, 0.48, facecolor="none",
                        edgecolor=MAROON, lw=LW, zorder=3))
    handle_start = (glass_c[0] + 0.48 * np.cos(np.radians(45)),
                    glass_c[1] - 0.48 * np.sin(np.radians(45)))
    ax.plot([handle_start[0], 0.78], [handle_start[1] - 0.02, -0.78],
            color=MAROON, lw=LW + 1, solid_capstyle="round", zorder=3)
    save(fig, "01_gather_experimental_cross_sections.png")


# 2. Generate Computational Cross Sections (FRESCO) -> chip with coupled wave
def icon_fresco():
    fig, ax = new_canvas()
    r = 0.42
    ax.add_patch(Rectangle((-r, -r), 2 * r, 2 * r, facecolor="none",
                           edgecolor=MAROON, lw=LW, joinstyle="round", zorder=2))
    pin_len = 0.22
    for x in (-0.22, 0.22):
        ax.plot([x, x], [r, r + pin_len], color=MAROON, lw=LW, zorder=2)
        ax.plot([x, x], [-r, -r - pin_len], color=MAROON, lw=LW, zorder=2)
    for y in (-0.22, 0.22):
        ax.plot([r, r + pin_len], [y, y], color=MAROON, lw=LW, zorder=2)
        ax.plot([-r, -r - pin_len], [y, y], color=MAROON, lw=LW, zorder=2)
    t = np.linspace(-r + 0.08, r - 0.08, 100)
    wave = 0.16 * np.sin(2 * np.pi * (t + r) / (1.3 * r))
    ax.plot(t, wave, color=MAROON, lw=LW - 1, solid_capstyle="round", zorder=3)
    save(fig, "02_generate_computational_cross_sections.png")


# 3. Calculate Energy-Bin Total Cross Section Distributions -> histogram
def icon_binning():
    fig, ax = new_canvas()
    heights = [0.25, 0.55, 0.85, 0.45, 0.65]
    xs = np.linspace(-0.72, 0.72, len(heights))
    w = 0.24
    base = -0.55
    for x, h in zip(xs, heights):
        ax.add_patch(Rectangle((x - w / 2, base), w, h, facecolor=MAROON,
                               alpha=0.85, edgecolor=MAROON, lw=1.5, zorder=2))
    ax.plot([-0.85, 0.85], [base, base], color=MAROON, lw=LW, zorder=3,
            solid_capstyle="round")
    ax.plot([-0.85, -0.85], [base, 0.95], color=MAROON, lw=LW, zorder=3,
            solid_capstyle="round")
    save(fig, "03_energy_bin_distributions.png")


# 4. Compute Abundance Equation Source Terms -> sigma feeding an arrow into a box
def icon_source_terms():
    fig, ax = new_canvas()
    ax.text(-0.55, 0.02, r"$\Sigma$", fontsize=42, color=MAROON,
            ha="center", va="center", fontweight="bold", zorder=2)
    arrow = FancyArrowPatch((-0.18, 0.0), (0.32, 0.0), arrowstyle="-|>",
                            mutation_scale=22, color=MAROON, lw=LW, zorder=2)
    ax.add_patch(arrow)
    ax.add_patch(Rectangle((0.38, -0.32), 0.42, 0.64, facecolor="none",
                           edgecolor=MAROON, lw=LW, joinstyle="round", zorder=2))
    ax.plot([0.48, 0.7], [0.1, 0.1], color=MAROON, lw=LW - 1.5,
            solid_capstyle="round", zorder=3)
    ax.plot([0.48, 0.62], [-0.1, -0.1], color=MAROON, lw=LW - 1.5,
            solid_capstyle="round", zorder=3)
    save(fig, "04_abundance_source_terms.png")


# 5. Compute Total Abundance Evolution -> rising curve + atom marker
def icon_evolution():
    fig, ax = new_canvas()
    ax.plot([-0.75, -0.75], [-0.75, 0.85], color=MAROON, lw=LW - 1.5,
            solid_capstyle="round", zorder=2)
    ax.plot([-0.75, 0.85], [-0.75, -0.75], color=MAROON, lw=LW - 1.5,
            solid_capstyle="round", zorder=2)
    t = np.linspace(-0.65, 0.65, 100)
    curve = -0.6 + 1.25 / (1 + np.exp(-6 * t))
    ax.plot(t, curve, color=MAROON, lw=LW, solid_capstyle="round", zorder=3)
    ax.add_patch(Circle((0.65, curve[-1]), 0.07, facecolor=MAROON,
                        edgecolor="none", zorder=4))
    for angle in (20, 80, 140):
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (0.65, curve[-1]), 0.34, 0.13, angle=angle, facecolor="none",
            edgecolor=MAROON, lw=1.6, zorder=3))
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
    icon_gather()
    icon_fresco()
    icon_binning()
    icon_source_terms()
    icon_evolution()
    make_contact_sheet()


if __name__ == "__main__":
    main()
