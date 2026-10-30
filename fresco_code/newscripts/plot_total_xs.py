#!/usr/bin/env python3
"""
plot_total_xs.py
================

Makes two sets of figures from the *_total_xs.csv files produced by integrate_xs.py.

When a reaction has both FRESCO σ(E) and experimental (E, σ) data, the FRESCO
curve is σ_tot-normalised before plotting:

    s(E) = σ_exp_interp(E) / σ_FRESCO(E)          (log-log interpolation)

Boundary handling:
  • E < E_lo_exp  →  scale = 0; FRESCO not shown  (clamped)
  • E_lo ≤ E ≤ E_hi  →  log-log interpolated scale (solid line)
  • E > E_hi_exp  →  scale held at s(E_hi); FRESCO shown dashed (flagged)

Scale factors are written to  <out-dir>/<reaction>_scale_factors.csv.

Figures produced:
  1. Per-reaction: absolute σ(E) vs E — normalised FRESCO + experimental points.
  2. Combined overlay: σ/σ_peak vs E for all reactions.
  3. Grid: same normalised curves as subplot grid.

Usage
-----
    python scripts/newscripts/plot_total_xs.py
    python scripts/newscripts/plot_total_xs.py --runs-dir newruns --out-dir plots/
    python scripts/newscripts/plot_total_xs.py --reactions he4_p_d_he3 he4_he4_n_be7
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

plt.rcParams.update({"font.size": 10})

REACTION_LABELS = {
    "he4_p_d_he3":   r"$^4$He(p,d)$^3$He",
    "he4_he4_n_be7": r"$^4$He($^4$He,n)$^7$Be",
    "he4_he4_p_li7": r"$^4$He($^4$He,p)$^7$Li",
    "he4_he4_n_li7": r"$^4$He($^4$He,n)$^7$Li",
    "li7_p_he4_he4": r"$^7$Li(p,$\alpha$)$\alpha$",
    "p_li7_he4_he4": r"p($^7$Li,$\alpha$)$\alpha$",
    "p_d_n_p":       r"p(d,n)p",
    "p_d_n_d":       r"p(d,n)d",
    "p_d_dp_n":      r"p(d,dp)n",
    "he4_p_p_he4":   r"$^4$He(p,p)$^4$He",
    "p_he4_p_he4":   r"p($^4$He,p)$^4$He",
}

# --- kinematics support for the --x-axis option -----------------------------
AMU_MEV = 931.49410242  # MeV per atomic mass unit

# Mass number A for each nuclide token that can appear in a run-dir name.
# Used only to build the x-axis (cm / per-nucleon); A is an adequate
# approximation for the rest mass (m ~= A * AMU_MEV).
NUCLIDE_A = {
    "n": 1, "p": 1, "d": 2, "t": 3,
    "he3": 3, "he4": 4,
    "li6": 6, "li7": 7,
    "be7": 7, "be9": 9, "be10": 10,
    "b10": 10, "b11": 11,
    "c12": 12, "c13": 13, "c14": 14,
    "n13": 13, "n14": 14, "n15": 15,
    "o15": 15, "o16": 16, "o17": 17, "o18": 18,
    "f17": 17, "f18": 18,
    "ne20": 20, "ne21": 21,
}

XLABELS = {
    "lab":         r"$E_{\rm lab}$ (MeV)",
    "cm":          r"$E_{\rm cm}$ (MeV)",
    "per_nucleon": r"$E_{\rm lab}/A$ (MeV/nucleon)",
}

# Extrapolation mode above the highest measured energy point.
# "fresco_shape" – follow the FRESCO angular-distribution shape (scale held constant).
# "power_law"    – extend as σ ∝ 1/E anchored to the last measured point.
EXTRAP_MODE = {
    "he4_p_d_he3": "fresco_shape",   # DWBA shape is trusted for this channel
    # all other reactions default to "power_law" (αα, cluster, multi-body)
}

EXP_MARKERS = ["o", "s", "^", "D", "v", "P", "*"]
EXP_COLOURS = plt.cm.tab10(np.linspace(0, 0.9, 10))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_total_xs(csv_path):
    """
    Return (fresco_rows, exp_rows).  Rows with empty/non-numeric E or σ ≤ 0 are dropped.
    Each row is a dict: E, sigma, d_sigma (nan if absent), coverage, [source for exp].
    """
    fresco, exp = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            e_str = row.get("E_MeV", "").strip()
            if not e_str:
                continue
            try:
                e     = float(e_str)
                sigma = float(row["sigma_mb"])
            except (ValueError, KeyError):
                continue
            if not math.isfinite(sigma) or sigma <= 0:
                continue
            d_str   = row.get("d_sigma_mb", "").strip()
            d_sigma = float(d_str) if d_str else float("nan")
            cov     = float(row.get("coverage_frac", 1.0))
            rec     = {"E": e, "sigma": sigma, "d_sigma": d_sigma, "coverage": cov}
            if row["source"] == "fresco":
                fresco.append(rec)
            else:
                rec["source"] = row["source"]
                exp.append(rec)
    return fresco, exp


# ---------------------------------------------------------------------------
# Energy-axis conversion
# ---------------------------------------------------------------------------

def reaction_kinematics(reaction):
    """
    Parse a run-dir name of the form ``{target}_{projectile}_{...}`` and return
    ``(A_projectile, A_target)``.  Returns None if either nuclide is unknown.

    The CSV energy column is the *projectile* lab kinetic energy on a target
    at rest, so the second token is the projectile.
    """
    toks = reaction.split("_")
    if len(toks) < 2:
        return None
    a_target = NUCLIDE_A.get(toks[0])
    a_proj   = NUCLIDE_A.get(toks[1])
    if a_target is None or a_proj is None:
        return None
    return a_proj, a_target


def convert_energy(E_lab, reaction, mode):
    """
    Convert projectile lab kinetic energy (MeV) to the requested x-axis.

    mode = "lab"          -> unchanged
           "per_nucleon"  -> E_lab / A_projectile
           "cm"           -> relativistic CM kinetic energy
                             sqrt(s) - (m_p + m_t),
                             s = m_p^2 + m_t^2 + 2 m_t (E_lab + m_p)

    Falls back to the lab energy (unchanged) when the nuclides are unknown.
    """
    E_lab = np.asarray(E_lab, dtype=float)
    if mode == "lab" or E_lab.size == 0:
        return E_lab
    kin = reaction_kinematics(reaction)
    if kin is None:
        return E_lab
    a_p, a_t = kin
    if mode == "per_nucleon":
        return E_lab / a_p
    if mode == "cm":
        m_p = a_p * AMU_MEV
        m_t = a_t * AMU_MEV
        s = m_p**2 + m_t**2 + 2.0 * m_t * (E_lab + m_p)
        return np.sqrt(s) - (m_p + m_t)
    return E_lab


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_single_point_norm(fresco_rows, exp_rows):
    """
    Single-point anchoring: when there is exactly one experimental σ_tot,
    compute scale = σ_exp / σ_fresco(E_anchor) and apply it uniformly to the
    whole FRESCO curve (no clamping below E_anchor).

    Returns a norm dict compatible with save_scale_factors / plot_reaction,
    with an extra ``single_point`` key set to True.
    """
    valid_exp = [
        r for r in exp_rows
        if math.isfinite(r.get("E", float("nan"))) and r["sigma"] > 0
    ]
    if len(valid_exp) != 1 or not fresco_rows:
        return None

    ep           = valid_exp[0]
    E_anchor     = ep["E"]
    sigma_anchor = ep["sigma"]

    fresco_E  = np.array([r["E"]     for r in fresco_rows])
    fresco_sg = np.array([r["sigma"] for r in fresco_rows])

    valid = (fresco_sg > 0) & np.isfinite(fresco_sg)
    if not np.any(valid):
        return None

    log_sig_anchor = float(np.interp(
        math.log(E_anchor),
        np.log(fresco_E[valid]),
        np.log(fresco_sg[valid]),
    ))
    sigma_fresco_anchor = math.exp(log_sig_anchor)
    scale_factor = sigma_anchor / sigma_fresco_anchor

    scale      = np.full_like(fresco_E, scale_factor)
    sigma_norm = np.where(fresco_sg > 0, fresco_sg * scale_factor, float("nan"))

    return {
        "E":             fresco_E,
        "sigma_fresco":  fresco_sg,
        "sigma_norm":    sigma_norm,
        "scale":         scale,
        "is_extrap":     np.zeros(len(fresco_E), dtype=bool),
        "E_lo":          E_anchor,
        "E_hi":          E_anchor,
        "single_point":  True,
        "anchor_E":      E_anchor,
        "anchor_sigma":  sigma_anchor,
        "anchor_scale":  scale_factor,
    }


def compute_normalization(fresco_rows, exp_rows, extrap_mode="fresco_shape"):
    """
    Compute per-energy scale factors  s(E) = σ_exp_interp(E) / σ_FRESCO(E).

    Returns a dict, or None if either side has no usable data.

    extrap_mode : "fresco_shape" – hold scale constant above E_hi (follow FRESCO shape).
                  "power_law"   – extend as σ ∝ 1/E anchored to the last measured point.

    Keys:
        E               – FRESCO energy grid (array)
        sigma_fresco    – raw FRESCO σ (array)
        sigma_norm      – normalised FRESCO σ (array, nan where clamped below E_lo)
        scale           – s(E)  (array)
        is_extrap       – boolean mask: True for E > E_hi
        E_lo, E_hi      – energy bounds of the experimental data
        extrap_mode     – the extrap_mode used
    """
    valid_exp = sorted(
        (r["E"], r["sigma"])
        for r in exp_rows
        if math.isfinite(r.get("E", float("nan"))) and r["sigma"] > 0
    )
    if not valid_exp or not fresco_rows:
        return None

    exp_E  = np.array([p[0] for p in valid_exp])
    exp_sg = np.array([p[1] for p in valid_exp])

    fresco_E  = np.array([r["E"]     for r in fresco_rows])
    fresco_sg = np.array([r["sigma"] for r in fresco_rows])

    E_lo, E_hi = float(exp_E[0]), float(exp_E[-1])

    # --- log-log interpolation of experimental σ onto FRESCO E grid -----------
    # np.interp operates on log-E / log-σ; left/right handle boundary:
    #   left  = -inf → exp(−∞) = 0  (clamp below E_lo)
    #   right = log(exp_sg[-1])      (hold last experimental σ above E_hi)
    log_sg_interp = np.interp(
        np.log(fresco_E),
        np.log(exp_E),
        np.log(exp_sg),
        left=float("-inf"),
        right=float(np.log(exp_sg[-1])),
    )
    sigma_exp_interp = np.exp(log_sg_interp)

    # Enforce clamp: any FRESCO point strictly below E_lo gets scale = 0
    below = fresco_E < E_lo
    sigma_exp_interp[below] = 0.0

    is_extrap = fresco_E > E_hi

    # --- scale factor ----------------------------------------------------------
    # s = σ_exp_interp / σ_fresco.  NaN where σ_fresco = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(fresco_sg > 0, sigma_exp_interp / fresco_sg, float("nan"))

    # Hold the scale constant above E_hi: find last in-range scale and broadcast
    in_range = (~below) & (~is_extrap)
    if np.any(in_range):
        last_valid_idx = np.where(in_range)[0][-1]
        s_hold = scale[last_valid_idx]
    else:
        s_hold = float("nan")

    scale[is_extrap] = s_hold
    sigma_norm = fresco_sg * scale
    sigma_norm[below] = float("nan")   # don't plot below E_lo

    # Power-law extrapolation: replace FRESCO shape above E_hi with σ = A/E,
    # anchored to the last measured point so the join is continuous.
    if extrap_mode == "power_law" and np.any(is_extrap):
        power_law_A = float(exp_sg[-1]) * float(exp_E[-1])
        sigma_norm[is_extrap] = power_law_A / fresco_E[is_extrap]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale[is_extrap] = np.where(
                fresco_sg[is_extrap] > 0,
                sigma_norm[is_extrap] / fresco_sg[is_extrap],
                float("nan"),
            )

    return {
        "E":             fresco_E,
        "sigma_fresco":  fresco_sg,
        "sigma_norm":    sigma_norm,
        "scale":         scale,
        "is_extrap":     is_extrap,
        "E_lo":          E_lo,
        "E_hi":          E_hi,
        "extrap_mode":   extrap_mode,
    }


def save_scale_factors(reaction, norm, save_dir):
    """Write <reaction>_scale_factors.csv to save_dir."""
    if norm is None:
        return
    out = save_dir / f"{reaction}_scale_factors.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E_MeV", "sigma_fresco_mb", "sigma_norm_mb", "scale_factor", "extrapolated"])
        for i, e in enumerate(norm["E"]):
            def fmt(v):
                return f"{v:.6g}" if math.isfinite(v) else "nan"
            w.writerow([
                f"{e:.3f}",
                fmt(norm["sigma_fresco"][i]),
                fmt(norm["sigma_norm"][i]),
                fmt(norm["scale"][i]),
                "1" if norm["is_extrap"][i] else "0",
            ])
    print(f"  scale factors -> {out.name}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_exp(ax, exp_rows, start_idx=0, reaction=None, x_axis="lab"):
    """Plot experimental points onto ax.  Returns next colour/marker index."""
    sources = {}
    for r in exp_rows:
        sources.setdefault(r["source"], []).append(r)

    idx = start_idx
    for src, rows in sorted(sources.items()):
        rows.sort(key=lambda r: r["E"])
        ee  = convert_energy([r["E"] for r in rows], reaction, x_axis)
        ss  = np.array([r["sigma"]    for r in rows])
        dd  = np.array([r["d_sigma"]  for r in rows])
        covs = np.array([r["coverage"] for r in rows])
        partial  = covs < 0.9
        has_err  = np.isfinite(dd)
        marker   = EXP_MARKERS[idx % len(EXP_MARKERS)]
        colour   = EXP_COLOURS[idx % len(EXP_COLOURS)]
        full_lbl = src
        trunc_lbl = f"{src} (lower bound)"

        if np.any(~partial):
            m = ~partial
            ax.errorbar(ee[m], ss[m],
                        yerr=np.where(has_err[m], dd[m], 0),
                        fmt=marker, color=colour, capsize=3, ms=6,
                        label=full_lbl, zorder=5)
            full_lbl = "_nolegend_"

        if np.any(partial):
            m = partial
            ax.errorbar(ee[m], ss[m],
                        yerr=np.where(has_err[m], dd[m], 0),
                        fmt=marker, color=colour, capsize=3, ms=6,
                        markerfacecolor="none", mew=1.5,
                        label=trunc_lbl, zorder=5)
        idx += 1
    return idx


def plot_reaction(reaction, fresco_rows, exp_rows, norm, save_dir, x_axis="lab"):
    """Absolute-σ figure for one reaction, with normalised FRESCO curve."""
    if not fresco_rows and not exp_rows:
        return

    pretty = REACTION_LABELS.get(reaction, reaction.replace("_", " "))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(pretty, fontsize=12)

    if fresco_rows:
        if norm is not None:
            E   = convert_energy(norm["E"], reaction, x_axis)
            sn  = norm["sigma_norm"]
            ext = norm["is_extrap"]

            if norm.get("single_point"):
                # Constant scale applied to whole curve
                ok = np.isfinite(sn)
                if np.any(ok):
                    ax.semilogy(E[ok], sn[ok], color="steelblue", lw=1.5, zorder=3,
                                label=f"FRESCO (single-point anchored, {norm['anchor_E']:.0f} MeV)")
            else:
                # In-range portion (solid)
                m_in = np.isfinite(sn) & ~ext
                if np.any(m_in):
                    ax.semilogy(E[m_in], sn[m_in],
                                color="steelblue", lw=1.5, label="FRESCO (normalised)", zorder=3)

                # Extrapolated portion (dashed)
                m_ex = np.isfinite(sn) & ext
                if np.any(m_ex):
                    em = norm.get("extrap_mode", "fresco_shape")
                    extrap_lbl = (
                        f"1/E extrap. > {norm['E_hi']:.0f} MeV"
                        if em == "power_law"
                        else f"FRESCO (extrap. > {norm['E_hi']:.0f} MeV)"
                    )
                    # Connect last in-range point to first extrap point for visual continuity
                    if np.any(m_in):
                        join_E  = [E[m_in][-1],  E[m_ex][0]]
                        join_sn = [sn[m_in][-1], sn[m_ex][0]]
                        ax.semilogy(join_E, join_sn, color="steelblue", lw=1.0,
                                    ls="--", zorder=3)
                    ax.semilogy(E[m_ex], sn[m_ex],
                                color="steelblue", lw=1.0, ls="--",
                                label=extrap_lbl, zorder=3)
        else:
            # No experimental data to normalise against — show raw curve
            fe = convert_energy([r["E"] for r in fresco_rows], reaction, x_axis)
            fs = np.array([r["sigma"] for r in fresco_rows])
            ax.semilogy(fe, fs, color="steelblue", lw=1.5, label="FRESCO (raw)", zorder=3)

    _plot_exp(ax, exp_rows, reaction=reaction, x_axis=x_axis)

    ax.set_xlabel(XLABELS.get(x_axis, XLABELS["lab"]))
    ax.set_ylabel(r"$\sigma$ (mb)")
    ax.set_yscale("log")
    ax.set_xlim(0, 400)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = save_dir / f"{reaction}_total_xs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out.name}")


# ---------------------------------------------------------------------------
# Combined normalised figures
# ---------------------------------------------------------------------------

def _split_norm_curve(fresco_rows, norm):
    """
    Return (E, sigma, is_extrap) for the normalised FRESCO curve, masked to
    finite sigma.  is_extrap is True for points above the highest measured
    energy (so callers can render them dashed).  Falls back to the raw FRESCO
    curve (all in-range) when there is no normalisation.
    """
    if norm is not None:
        E  = np.asarray(norm["E"],          dtype=float)
        sn = np.asarray(norm["sigma_norm"], dtype=float)
        ex = np.asarray(norm["is_extrap"],  dtype=bool)
        ok = np.isfinite(sn)
        return E[ok], sn[ok], ex[ok]
    if fresco_rows:
        fe = np.array([r["E"]     for r in fresco_rows], dtype=float)
        fs = np.array([r["sigma"] for r in fresco_rows], dtype=float)
        return fe, fs, np.zeros(len(fe), dtype=bool)
    return np.array([]), np.array([]), np.array([], dtype=bool)


def plot_combined_normalised(all_data, all_norms, save_dir, x_axis="lab"):
    """σ/σ_peak overlay and subplot grid using normalised FRESCO curves."""
    reactions = list(all_data.keys())
    if not reactions:
        return
    colours = plt.cm.turbo(np.linspace(0.05, 0.95, len(reactions)))

    # Warn once for any reaction whose nuclides we can't map (CM / per-nucleon
    # silently fall back to lab for those).
    if x_axis != "lab":
        for reaction in reactions:
            if reaction_kinematics(reaction) is None:
                print(f"  [warn] {reaction}: unknown nuclide token -- "
                      f"plotted on lab energy despite --x-axis {x_axis}")

    def _draw_split(ax, E_x, y, is_ex, colour, lw, label=None):
        """Solid line for measured region, dashed for extrapolated region."""
        m_in, m_ex = ~is_ex, is_ex
        if np.any(m_in):
            ax.plot(E_x[m_in], y[m_in], color=colour, lw=lw, label=label)
            label = None  # consume label on the solid segment
        if np.any(m_ex):
            if np.any(m_in):  # bridge the gap so the curve reads continuously
                ax.plot([E_x[m_in][-1], E_x[m_ex][0]],
                        [y[m_in][-1],   y[m_ex][0]],
                        color=colour, lw=lw * 0.85, ls="--")
            ax.plot(E_x[m_ex], y[m_ex], color=colour, lw=lw * 0.85, ls="--",
                    label=label)

    # --- combined overlay ---
    fig_all, ax_all = plt.subplots(figsize=(9, 5))
    ax_all.set_title("Normalised total cross sections", fontsize=12)

    for (reaction, (fresco_rows, exp_rows)), colour in zip(all_data.items(), colours):
        pretty       = REACTION_LABELS.get(reaction, reaction.replace("_", " "))
        norm         = all_norms.get(reaction)
        fe, fs, fex  = _split_norm_curve(fresco_rows, norm)

        all_sigmas = list(fs) + [r["sigma"] for r in exp_rows]
        if not all_sigmas:
            continue
        peak = max(all_sigmas)

        if len(fe):
            fe_x = convert_energy(fe, reaction, x_axis)
            _draw_split(ax_all, fe_x, fs / peak, fex, colour, lw=1.5, label=pretty)

        sources = {}
        for r in exp_rows:
            sources.setdefault(r["source"], []).append(r)
        for _, rows in sorted(sources.items()):
            rows.sort(key=lambda r: r["E"])
            ee = convert_energy([r["E"] for r in rows], reaction, x_axis)
            ss = np.array([r["sigma"]  for r in rows]) / peak
            dd = np.array([r["d_sigma"] for r in rows]) / peak
            has_err = np.isfinite(dd)
            ax_all.errorbar(ee, ss,
                            yerr=np.where(has_err, dd, 0),
                            fmt="o", color=colour, capsize=3, ms=5, zorder=5,
                            label="_nolegend_" if len(fe) else f"{pretty} (exp)")

    ax_all.set_xlabel(XLABELS.get(x_axis, XLABELS["lab"]))
    ax_all.set_ylabel(r"$\sigma\,/\,\sigma_{\rm peak}$")
    ax_all.set_yscale("log")
    ax_all.set_xlim(0, 400)
    ax_all.set_ylim(bottom=1e-6)
    ax_all.grid(True, which="both", alpha=0.3)
    ax_all.legend(fontsize=7, ncol=2, loc="upper right")
    fig_all.tight_layout()
    out_all = save_dir / "all_reactions_normalised_xs.png"
    fig_all.savefig(out_all, dpi=150)
    plt.close(fig_all)
    print(f"  saved {out_all.name}")

    # --- subplot grid ---
    n_cols = min(len(reactions), 4)
    n_rows = math.ceil(len(reactions) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 2.8))
    axes_flat = np.array(axes).flatten()

    for idx, (reaction, (fresco_rows, exp_rows)) in enumerate(all_data.items()):
        ax          = axes_flat[idx]
        pretty      = REACTION_LABELS.get(reaction, reaction.replace("_", " "))
        colour      = colours[idx]
        norm        = all_norms.get(reaction)
        fe, fs, fex = _split_norm_curve(fresco_rows, norm)

        all_sigmas = list(fs) + [r["sigma"] for r in exp_rows]
        if not all_sigmas:
            ax.set_visible(False)
            continue
        peak = max(all_sigmas)

        if len(fe):
            fe_x = convert_energy(fe, reaction, x_axis)
            _draw_split(ax, fe_x, fs / peak, fex, colour, lw=1.2)

        sources = {}
        for r in exp_rows:
            sources.setdefault(r["source"], []).append(r)
        for _, rows in sorted(sources.items()):
            rows.sort(key=lambda r: r["E"])
            ee = convert_energy([r["E"] for r in rows], reaction, x_axis)
            ss = np.array([r["sigma"]   for r in rows]) / peak
            dd = np.array([r["d_sigma"] for r in rows]) / peak
            has_err = np.isfinite(dd)
            ax.errorbar(ee, ss, yerr=np.where(has_err, dd, 0),
                        fmt="o", color=colour, capsize=2, ms=4, zorder=5)

        ax.set_title(pretty, fontsize=7)
        ax.set_yscale("log")
        ax.set_xlim(0, 400)
        ax.set_ylim(bottom=1e-6)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(XLABELS.get(x_axis, XLABELS["lab"]), fontsize=7)
        ax.set_ylabel(r"$\sigma/\sigma_{\rm peak}$", fontsize=7)

    for idx in range(len(all_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    out_grid = save_dir / "all_reactions_normalised_grid.png"
    fig.savefig(out_grid, dpi=150)
    plt.close(fig)
    print(f"  saved {out_grid.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir",  default=str(_PROJECT_ROOT / "newruns"))
    ap.add_argument("--out-dir",   default=str(_PROJECT_ROOT / "plots"))
    ap.add_argument("--reactions", nargs="*", default=None)
    ap.add_argument("--x-axis", choices=["lab", "cm", "per_nucleon"], default="lab",
                    help="energy axis: lab (default), cm (relativistic CM kinetic "
                         "energy), or per_nucleon (E_lab / A_projectile). "
                         "cm/per_nucleon make inverse-kinematics pairs comparable.")
    args = ap.parse_args()

    runs_root = Path(args.runs_dir)
    save_dir  = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(d for d in runs_root.iterdir() if d.is_dir())
    if args.reactions:
        candidates = [d for d in candidates if d.name in args.reactions]

    all_data  = {}
    all_norms = {}

    for rdir in candidates:
        reaction = rdir.name
        csv_path = rdir / f"{reaction}_total_xs.csv"
        if not csv_path.exists():
            continue
        fresco_rows, exp_rows = load_total_xs(csv_path)
        if not fresco_rows and not exp_rows:
            continue

        valid_exp = [
            r for r in exp_rows
            if math.isfinite(r.get("E", float("nan"))) and r["sigma"] > 0
        ]
        if len(valid_exp) == 1:
            norm = compute_single_point_norm(fresco_rows, exp_rows)
            if norm is not None:
                print(f"  [{reaction}] single-point anchoring at {norm['anchor_E']:.1f} MeV "
                      f"(scale={norm['anchor_scale']:.4g})")
                save_scale_factors(reaction, norm, save_dir)
            else:
                print(f"  [{reaction}] single-point anchoring failed -- FRESCO shown raw")
        else:
            extrap = EXTRAP_MODE.get(reaction, "power_law")
            norm = compute_normalization(fresco_rows, exp_rows, extrap_mode=extrap)
            if norm is not None:
                print(f"  [{reaction}] normalising FRESCO ({norm['extrap_mode']}): "
                      f"E_lo={norm['E_lo']:.1f} MeV, E_hi={norm['E_hi']:.1f} MeV")
                save_scale_factors(reaction, norm, save_dir)
            else:
                print(f"  [{reaction}] no experimental sigma(E) -- FRESCO shown raw")

        all_data[reaction]  = (fresco_rows, exp_rows)
        all_norms[reaction] = norm
        plot_reaction(reaction, fresco_rows, exp_rows, norm, save_dir, x_axis=args.x_axis)

    plot_combined_normalised(all_data, all_norms, save_dir, x_axis=args.x_axis)
    print("Done.")


if __name__ == "__main__":
    main()