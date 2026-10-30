#!/usr/bin/env python3
"""
Regenerate the three "Steps: Obtaining Non-thermal Abundance Evolution" figures
at font sizes legible on a printed 48 x 36 in poster.

The originals (fresco_code/newruns/plots/he4_he4_n_be7_total_xs.png,
fresco_code/newruns/bins/he4_he4_n_be7_poster.png, and
outputs/famiano_abundance_evolution_*.png) were sized for on-screen viewing --
axis ticks, legend text, and figure titles are set with matplotlib's small
defaults (8-13pt) and get illegible once blown up to poster scale.

Sizing: the plots sit at a MEASURED placeholder width of 7 in (19-26 in
across the 48-in poster), with a caption running beside each one. Font sizes
are chosen as literal target print points and figsize width is set to that
same 7 in, so the point values below ARE the printed size (no scale-up
correction needed) -- this only holds if the image is placed at ~7 in wide;
resize the constants here if the placeholder changes.

Output: outputs/poster_plots/{01..03}_*.png
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "outputs" / "poster_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "fresco_code" / "newscripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_total_xs as ptx          # noqa: E402
import plot_famiano as pf            # noqa: E402

FIG_WIDTH_IN = 7.0  # measured placeholder width -- keep figsize width == this

KINK_NOTE = "data ends here — 1/E extrapolation\n(same method as Famiano et al. 2002)"


def annotate_extrapolation_boundary(ax, e_boundary, y_frac=0.34, ha="left", x_offset_frac=0.02):
    """Mark where data-anchored values give way to the 1/E extrapolation --
    matches the fallback used in Famiano et al. (2002, Sec. 3) for reactions
    without high-energy data."""
    ax.axvline(e_boundary, color="#888888", ls=":", lw=1.6, zorder=1)
    ylo, yhi = ax.get_ylim()
    y_text = 10 ** (np.log10(ylo) + y_frac * (np.log10(yhi) - np.log10(ylo)))
    xlo, xhi = ax.get_xlim()
    x_text = e_boundary + x_offset_frac * (xhi - xlo)
    ax.text(x_text, y_text, KINK_NOTE, fontsize=9, color="#666666",
            ha=ha, va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))


POSTER_RC = {
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
}


# 1. FRESCO normalised cross section for he4_he4_n_be7, with experimental anchors
def make_fresco_normalized():
    reaction = "he4_he4_n_be7"
    csv_path = ROOT / "fresco_code/newruns" / reaction / f"{reaction}_total_xs.csv"
    fresco_rows, exp_rows = ptx.load_total_xs(csv_path)

    extrap = ptx.EXTRAP_MODE.get(reaction, "power_law")
    norm = ptx.compute_normalization(fresco_rows, exp_rows, extrap_mode=extrap)

    pretty = ptx.REACTION_LABELS.get(reaction, reaction.replace("_", " "))

    with plt.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 5.3))
        ax.set_title(pretty, pad=10)

        E = ptx.convert_energy(norm["E"], reaction, "lab")
        sn, ext = norm["sigma_norm"], norm["is_extrap"]

        # sigma_norm cancels fresco_sg algebraically in-range (sigma_norm =
        # fresco_sg * (sigma_exp_interp/fresco_sg) = sigma_exp_interp), so
        # this curve IS the data, log-log interpolated -- label it as such
        # rather than implying it's an independent FRESCO prediction.
        m_in = np.isfinite(sn) & ~ext
        ax.semilogy(E[m_in], sn[m_in], color="steelblue", lw=2.2,
                    label="Normalized to data", zorder=3)

        m_ex = np.isfinite(sn) & ext
        join_E = [E[m_in][-1], E[m_ex][0]]
        join_sn = [sn[m_in][-1], sn[m_ex][0]]
        ax.semilogy(join_E, join_sn, color="steelblue", lw=1.8, ls="--", zorder=3)
        ax.semilogy(E[m_ex], sn[m_ex], color="steelblue", lw=1.8, ls="--",
                    label="1/E extrapolation (beyond data)", zorder=3)

        # Plot all experimental points under one general label rather than the
        # raw per-file source name (e.g. "4He_4Hen_7Be_1") plot_total_xs.py uses.
        E_exp = np.array([r["E"] for r in exp_rows])
        sigma_exp = np.array([r["sigma"] for r in exp_rows])
        d_exp = np.array([r["d_sigma"] for r in exp_rows])
        order = np.argsort(E_exp)
        ax.errorbar(E_exp[order], sigma_exp[order],
                    yerr=np.where(np.isfinite(d_exp[order]), d_exp[order], 0),
                    fmt="o", color="steelblue", ms=8, capsize=4,
                    label="Experimental data", zorder=5)

        ax.set_xlabel(ptx.XLABELS["lab"])
        ax.set_ylabel(r"$\sigma$ (mb)")
        ax.set_xlim(0, 400)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
        annotate_extrapolation_boundary(ax, norm["E_hi"], y_frac=0.42)

        fig.tight_layout()
        out = OUT_DIR / "01_fresco_normalized_he4_he4_n_be7.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved: {out}  ({FIG_WIDTH_IN} x 5.3 in @ 300dpi)")


# 2. Same reaction, split into measured (scaled) vs extrapolated segments + gap band
def make_measured_extrapolated_split():
    reaction = "he4_he4_n_be7"
    csv_path = ROOT / "fresco_code/newruns/bins" / f"{reaction}_bins.csv"
    df = pd.read_csv(csv_path)
    df = df[df["reaction"] == reaction].sort_values("E_MeV")

    gaps = df[df["status"] == "gap"]["E_MeV"].sort_values().tolist()
    gap_lo, gap_hi = (gaps[0], gaps[-1]) if gaps else (None, None)

    plotted = df.dropna(subset=["sigma_mb"])
    scaled = plotted[plotted["status"] == "scaled"]
    extrap = plotted[plotted["status"] == "extrapolated"]

    with plt.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 5.3))

        if gap_lo is not None:
            ax.axvspan(gap_lo - 2.5, gap_hi + 2.5, facecolor="#e0e0e0", alpha=0.6,
                       hatch="//", edgecolor="#b0b0b0", lw=0, zorder=0)

        # Each point is a 5 MeV bin value: draw as a staircase (step function)
        # so each bin is flat across its width and steps to the next bin's
        # value at the shared edge, rather than a diagonal interpolation.
        half = 2.5

        def stairs_of(sub):
            E = sub["E_MeV"].values
            edges = np.append(E - half, E[-1] + half)
            return edges, sub["sigma_mb"].values

        if not scaled.empty:
            edges, vals = stairs_of(scaled)
            ax.stairs(vals, edges, baseline=None, color="#1f77b4", lw=2.5,
                      zorder=2, label="FRESCO, scaled to data")
        if not extrap.empty:
            edges, vals = stairs_of(extrap)
            ax.stairs(vals, edges, baseline=None, color="#ff7f0e", lw=2.5,
                      ls="--", zorder=2, label="FRESCO, extrapolated beyond data")

        ax.set_yscale("log")
        ax.set_xlim(-10, 420)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 2.0)
        if gap_lo is not None:
            ax.text((gap_lo + gap_hi) / 2, ymin * 1.2, "no data", ha="center",
                    va="bottom", fontsize=12, color="#808080", style="italic")

        ax.set_xlabel(r"$E_{\mathrm{lab}}$ (MeV)")
        ax.set_ylabel(r"$\sigma$ (mb)")
        ax.set_title(r"$^4$He + $^4$He $\to$ n + $^7$Be" "\n"
                     r"$\mathit{scaled\ 65\!-\!155\ MeV,\ extrapolated\ 160\!-\!400\ MeV}$",
                     pad=10, fontsize=15)
        ax.grid(True, which="both", ls=":", lw=0.7, color="#c9c9c9", alpha=0.8)
        ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
        boundary = float(extrap["E_MeV"].min())
        annotate_extrapolation_boundary(ax, boundary, y_frac=0.42)

        fig.tight_layout()
        out = OUT_DIR / "02_measured_extrapolated_split_he4_he4_n_be7.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved: {out}  ({FIG_WIDTH_IN} x 5.3 in @ 300dpi)")


# 3. Famiano (2002) Fig. 3 (Model B) abundance-evolution benchmark
def make_abundance_evolution():
    df = pf._read_history(ROOT / "outputs" / "abundance_history.csv")
    with plt.rc_context(POSTER_RC):
        df2 = df[df["t_s"] > 0].copy().reset_index(drop=True)
        t_s = df2["t_s"].values
        species = pf._active_species(df2)

        alpha = None
        if "delta_m_over_m0" in df2.columns:
            dm = pd.to_numeric(df2["delta_m_over_m0"], errors="coerce")
            mask = (df2["t_s"] > 0) & (dm > 0)
            if mask.any():
                alpha = float((dm[mask] / t_s[mask]).median())

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 5.3))
        y_H_raw = pd.to_numeric(df2["p"], errors="coerce").values.astype(float)
        y_H = np.where(y_H_raw > 0, y_H_raw, np.nan)

        fallback_idx = 0
        for sp in species:
            if sp == "p":
                continue
            y = pd.to_numeric(df2[sp], errors="coerce").values.astype(float)
            y_pos = np.where((y / y_H) > 0, y / y_H, np.nan)
            if np.all(np.isnan(y_pos)):
                continue
            kw = pf._style_for(sp, fallback_idx)
            if sp not in pf._STYLE:
                fallback_idx += 1
            kw["lw"] = kw.get("lw", 1.5) * 1.3
            ax.plot(t_s, y_pos, **kw)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$Y_i\,/\,Y_\mathrm{H}$")

        finite_vals = []
        for sp in species:
            if sp == "p":
                continue
            y = pd.to_numeric(df2[sp], errors="coerce").values.astype(float)
            pos = (y / y_H_raw)
            pos = pos[pos > 0]
            if pos.size:
                finite_vals.append(pos.min())
        if finite_vals:
            ax.set_ylim(bottom=max(min(finite_vals) * 0.1, 1e-20))

        ax.legend(fontsize=10, loc="lower left", ncol=2, framealpha=1.0,
                  handlelength=1.6, columnspacing=1.0, labelspacing=0.55)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)

        if alpha is not None and alpha > 0:
            ax2 = ax.secondary_xaxis(
                "top", functions=(lambda t: alpha * t, lambda dm: dm / alpha))
            ax2.set_xlabel(r"$\Delta M\,/\,M_0$", fontsize=15)
            ax2.tick_params(labelsize=12)

        ax.set_title(
            r"Blazar Jet Nucleosynthesis — Famiano (2002) Fig. 3 (Model B)" "\n"
            r"BLR cloud  $n_0 = 10^{11}$ cm$^{-3}$,  $T = 10^4$ K;"
            r"  Jet: p + $^4$He at 100 MeV/nucleon",
            fontsize=12, pad=28,
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.74, bottom=0.11)
        out = OUT_DIR / "03_abundance_evolution.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved: {out}  ({FIG_WIDTH_IN} x 5.3 in @ 300dpi)")


def main():
    make_fresco_normalized()
    make_measured_extrapolated_split()
    make_abundance_evolution()


if __name__ == "__main__":
    main()
