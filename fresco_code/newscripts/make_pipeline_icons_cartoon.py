#!/usr/bin/env python3
"""
Small, cartoony icons for the 5 Computational Pipeline steps on the poster.

Flat "sticker" style: bold dark outlines, solid fills, rounded corners, a
small gloss highlight -- friendly rather than technical, but kept inside the
poster's maroon/cream palette so it doesn't clash with the TAMU branding.

Output: outputs/pipeline_icons/{01..05}_<step>.png  (+ a contact-sheet preview)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyBboxPatch, Ellipse, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "pipeline_icons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAROON = "#500000"
MAROON_LIGHT = "#8B3A3A"
OUTLINE = "#2E0000"
CREAM = "#F5E9D3"
BADGE_BG = "#FFFBF3"
LW = 7


def new_canvas():
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=250)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    shadow = Circle((0.04, -0.05), 1.0, facecolor="#000000", alpha=0.10, zorder=0)
    ax.add_patch(shadow)
    badge = Circle((0, 0), 1.0, facecolor=BADGE_BG, edgecolor=OUTLINE,
                   lw=5, zorder=1)
    ax.add_patch(badge)
    clip = Circle((0, 0), 0.97, transform=ax.transData)
    return fig, ax, clip


def rounded_rect(xy, w, h, rounding=0.08, **kw):
    x, y = xy
    return FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={rounding}",
                          **kw)


def gloss(ax, center, w=0.5, h=0.2, angle=-25, xy_offset=(-0.15, 0.2)):
    e = Ellipse((center[0] + xy_offset[0], center[1] + xy_offset[1]), w, h,
               angle=angle, facecolor="white", alpha=0.35, zorder=5, lw=0)
    ax.add_patch(e)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {path}")


# 1. Gather Experimental Cross Sections -> chunky magnifying glass over data dots
def icon_gather():
    fig, ax, clip = new_canvas()
    dot_xy = [(-0.62, -0.15), (-0.25, 0.35), (0.05, -0.55), (0.55, -0.15), (-0.55, 0.55)]
    for x, y in dot_xy:
        ax.add_patch(Circle((x, y), 0.09, facecolor=MAROON_LIGHT, edgecolor=OUTLINE,
                            lw=3.5, zorder=2))
        ax.add_patch(Circle((x - 0.03, y + 0.03), 0.025, facecolor="white",
                            alpha=0.6, zorder=3, lw=0))

    lens_c = (0.02, 0.05)
    ax.add_patch(Circle(lens_c, 0.5, facecolor=CREAM, edgecolor=OUTLINE, lw=LW, zorder=4))
    gloss(ax, lens_c, w=0.4, h=0.16, xy_offset=(-0.16, 0.2))

    handle_dir = np.array([np.cos(np.radians(-45)), np.sin(np.radians(-45))])
    p0 = lens_c + handle_dir * 0.5
    p1 = lens_c + handle_dir * 0.98
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=MAROON, lw=LW + 4,
            solid_capstyle="round", zorder=3)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=OUTLINE, lw=LW + 8,
            solid_capstyle="round", zorder=2)
    save(fig, "01_gather_experimental_cross_sections.png")


# 2. Generate Computational Cross Sections -> friendly rounded chip with a wave + LED
def icon_fresco():
    fig, ax, clip = new_canvas()
    r = 0.45
    body = rounded_rect((-r, -r), 2 * r, 2 * r, rounding=0.14,
                        facecolor=MAROON, edgecolor=OUTLINE, lw=LW, zorder=2)
    ax.add_patch(body)

    pin_len, pin_w = 0.2, 0.07
    for x in (-0.24, 0.24):
        ax.add_patch(rounded_rect((x - pin_w / 2, r), pin_w, pin_len, rounding=0.02,
                                  facecolor=MAROON_LIGHT, edgecolor=OUTLINE, lw=2.5, zorder=1))
        ax.add_patch(rounded_rect((x - pin_w / 2, -r - pin_len), pin_w, pin_len, rounding=0.02,
                                  facecolor=MAROON_LIGHT, edgecolor=OUTLINE, lw=2.5, zorder=1))
    for y in (-0.24, 0.24):
        ax.add_patch(rounded_rect((r, y - pin_w / 2), pin_len, pin_w, rounding=0.02,
                                  facecolor=MAROON_LIGHT, edgecolor=OUTLINE, lw=2.5, zorder=1))
        ax.add_patch(rounded_rect((-r - pin_len, y - pin_w / 2), pin_len, pin_w, rounding=0.02,
                                  facecolor=MAROON_LIGHT, edgecolor=OUTLINE, lw=2.5, zorder=1))

    t = np.linspace(-r + 0.1, r - 0.1, 100)
    wave = 0.16 * np.sin(2 * np.pi * (t + r) / (1.2 * r))
    ax.plot(t, wave, color=CREAM, lw=6, solid_capstyle="round", zorder=3)

    ax.add_patch(Circle((0, 0.28), 0.07, facecolor="#F4C05A", edgecolor=OUTLINE,
                        lw=3, zorder=3))
    save(fig, "02_generate_computational_cross_sections.png")


# 3. Calculate Energy-Bin Total Cross Section Distributions -> chunky rounded bars
def icon_binning():
    fig, ax, clip = new_canvas()
    heights = [0.35, 0.62, 0.9, 0.5, 0.72]
    xs = np.linspace(-0.72, 0.72, len(heights))
    w = 0.26
    base = -0.6
    colors = [MAROON, MAROON_LIGHT] * 3
    for x, h, c in zip(xs, heights, colors):
        ax.add_patch(rounded_rect((x - w / 2, base), w, h, rounding=0.05,
                                  facecolor=c, edgecolor=OUTLINE, lw=4, zorder=2))
        ax.add_patch(Ellipse((x - 0.05, base + h - 0.06), w * 0.35, 0.04,
                             facecolor="white", alpha=0.35, zorder=3, lw=0))
    baseline, = ax.plot([-0.88, 0.88], [base, base], color=OUTLINE, lw=LW,
                        solid_capstyle="round", zorder=4)
    baseline.set_clip_path(clip)
    save(fig, "03_energy_bin_distributions.png")


# 4. Compute Abundance Equation Source Terms -> bold Sigma -> arrow -> box
def icon_source_terms():
    fig, ax, clip = new_canvas()
    ax.text(-0.55, 0.0, r"$\Sigma$", fontsize=54, color=MAROON,
            ha="center", va="center", fontweight="bold", zorder=3,
            path_effects=[pe.withStroke(linewidth=4, foreground=OUTLINE)])
    arrow = FancyArrowPatch((-0.14, 0.0), (0.28, 0.0), arrowstyle="-|>",
                            mutation_scale=28, color=MAROON,
                            linewidth=LW - 1, zorder=2,
                            capstyle="round", joinstyle="round")
    ax.add_patch(arrow)

    box = rounded_rect((0.34, -0.34), 0.5, 0.68, rounding=0.08, facecolor=CREAM,
                       edgecolor=OUTLINE, lw=LW, zorder=2)
    ax.add_patch(box)
    for yy, ww in [(0.1, 0.32), (-0.1, 0.24)]:
        ax.plot([0.44, 0.44 + ww], [yy, yy], color=MAROON, lw=6,
                solid_capstyle="round", zorder=3)
    save(fig, "04_abundance_source_terms.png")


# 5. Compute Total Abundance Evolution -> bouncy rising curve to a little atom
def icon_evolution():
    fig, ax, clip = new_canvas()
    xax, = ax.plot([-0.7, -0.7], [-0.7, 0.85], color=OUTLINE, lw=LW - 1,
                   solid_capstyle="round", zorder=2)
    yax, = ax.plot([-0.7, 0.85], [-0.7, -0.7], color=OUTLINE, lw=LW - 1,
                   solid_capstyle="round", zorder=2)
    xax.set_clip_path(clip)
    yax.set_clip_path(clip)

    t = np.linspace(-0.6, 0.6, 100)
    curve = -0.55 + 1.15 / (1 + np.exp(-6 * t))
    curve_line, = ax.plot(t, curve, color=MAROON, lw=LW + 1,
                          solid_capstyle="round", zorder=3)
    curve_line.set_clip_path(clip)

    cx, cy = 0.58, curve[-1] - 0.05
    for angle in (25, 90, 145):
        orbit = Ellipse((cx, cy), 0.36, 0.14, angle=angle, facecolor="none",
                        edgecolor=MAROON_LIGHT, lw=3, zorder=3)
        orbit.set_clip_path(clip)
        ax.add_patch(orbit)
    ax.add_patch(Circle((cx, cy), 0.1, facecolor="#F4C05A", edgecolor=OUTLINE,
                        lw=3.5, zorder=4))
    ax.add_patch(Circle((cx - 0.03, cy + 0.03), 0.03, facecolor="white",
                        alpha=0.7, zorder=5, lw=0))
    save(fig, "05_total_abundance_evolution.png")


def make_contact_sheet():
    from matplotlib import image as mpimg
    files = sorted(OUT_DIR.glob("0*.png"))
    fig, axes = plt.subplots(1, len(files), figsize=(2.6 * len(files), 2.8), dpi=200)
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
