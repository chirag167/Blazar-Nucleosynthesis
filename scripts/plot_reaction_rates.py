"""
plot_reaction_rates.py

For one or more species, plot how fast each reaction makes or destroys it
over the course of a run: dY/dt broken down by reaction, versus time.

Each figure has two panels sharing the time axis:
    top    : the species' abundance Y/H
    bottom : |dY/dt| (Y/H per second) for each contributing process, on a
             log scale -- solid lines make the species, dashed lines destroy
             it -- plus the net rate actually seen in the run (black)

Contributions are rebuilt from the saved abundances
(outputs/abundance_history.csv by default) with the same rate library and
decay table the simulation uses:
    - thermonuclear reactions: n * <sigma v>(T9) * Y_i * Y_j / (1 + delta_ij)
      times the species' net stoichiometry, with the cloud density at each
      row taken as n0 * (1 + DM/M0) (constant-volume growth, as in
      run_famiano.py)
    - radioactive decay: lambda * Y
    - "jet-cloud + mixing (inferred)": the net change seen between saved
      rows minus the two terms above. It lumps the jet-cloud reactions with
      the dilution and jet-composition mixing from the cloud's mass growth,
      which cannot be separated after the fact. Saved rows are 100 steps
      apart, so this line is coarse, especially near jet on/off switches.

Stretches where DM/M0 stays flat (jet switched off) are shaded.

Usage:
    python scripts/plot_reaction_rates.py 7Li t 3He
    python scripts/plot_reaction_rates.py 7Li --top 6 --csv path/to/history.csv
Writes outputs/dydt_plots/dydt_<species>.png for each species (folder created if missing).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "core"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rate_library import build_thermonuclear_library  # noqa: E402
from decay import DECAY_TABLE  # noqa: E402

_ELEM = ["n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg"]
_LIGHT = {"n": "n", "p": "p", "d": "d", "t": "t", "4He": r"\alpha"}


def _tex(sp: str, A: dict, Z: dict, light: bool = False) -> str:
    if light and sp in _LIGHT:
        return _LIGHT[sp]
    if sp == "n":
        return "n"
    return f"^{{{A[sp]}}}\\mathrm{{{_ELEM[Z[sp]]}}}"


def _reaction_label(reactants: dict, products: dict, A: dict, Z: dict, label: str | None) -> str:
    """target(projectile, ejectiles)residual in mathtext, heaviest reactant as target."""
    def expand(d):
        return sorted([s for s, m in d.items() for _ in range(m)], key=lambda s: (A.get(s, 0), s), reverse=True)
    r, p = expand(reactants), expand(products)
    to_2a = ""
    if label and "[8Be->2a]" in label:
        # 8Be products were rewritten as 2 alpha at ingestion; show the original channel.
        p = [x for x in p]
        for _ in range(2):
            p.remove("4He")
        p = sorted(p + ["8Be"], key=lambda s: (A.get(s, 8), s), reverse=True)
        to_2a = r"\to 2\alpha"
    target, proj = r[0], r[1]
    residual, ej = p[0], p[1:]

    def t(sp, light=False):
        if sp == "8Be":
            return r"^{8}\mathrm{Be}"
        return _tex(sp, A, Z, light)
    ej_s = "".join(t(e, True) for e in ej) if ej else r"\gamma"
    return f"${t(target)}({t(proj, True)},{ej_s}){t(residual, residual == '4He')}{to_2a}$"


def _decay_label(parent: str, ch, A: dict, Z: dict) -> str:
    return f"${_tex(parent, A, Z)}$ decay ({ch.mode.split(' ')[0]})"


def _jet_off_spans(t: np.ndarray, dm: np.ndarray) -> list[tuple[float, float]]:
    flat = np.diff(dm) <= 1e-12 * np.maximum(dm[1:], 1.0)
    spans, start = [], None
    for k, f in enumerate(flat):
        if f and start is None:
            start = t[k]
        elif not f and start is not None:
            spans.append((start, t[k])); start = None
    if start is not None:
        spans.append((start, t[-1]))
    return [(a, b) for a, b in spans if b > a and t[0] < a]


def contributions(species: str, data: dict, lib, A: dict, Z: dict, n0: float, T9: float):
    """Return {label: dY/dt array (per baryon, signed)} for thermonuclear + decay."""
    t = data["t_s"]
    n = n0 * (1.0 + data["delta_m_over_m0"])
    out: dict[str, np.ndarray] = {}
    for r in lib.reactions:
        net = r.products_stoich.get(species, 0) - r.reactants_stoich.get(species, 0)
        if net == 0:
            continue
        ys = [data.get(sp) for sp, m in r.reactants_stoich.items() for _ in range(m)]
        if any(y is None for y in ys):
            continue
        sv = float(r.rate_func(T9))
        delta = 1 if len(r.reactants_stoich) == 1 else 0
        rate = net * n * sv * ys[0] * ys[1] / (1 + delta)
        if np.any(rate != 0):
            lab = _reaction_label(r.reactants_stoich, r.products_stoich, A, Z, r.label)
            out[lab] = out.get(lab, 0.0) + rate
    for parent, ch in DECAY_TABLE.items():
        net = ch.daughters.get(species, 0) - (1 if parent == species else 0)
        if net == 0 or parent not in data:
            continue
        rate = net * (math.log(2.0) / ch.half_life_s) * data[parent]
        if np.any(rate != 0):
            out[_decay_label(parent, ch, A, Z)] = rate
    return out


def plot_species(species: str, data: dict, lib, A: dict, Z: dict, n0: float, T9: float,
                 top: int, out_path: Path, csv_name: str) -> None:
    t = data["t_s"]
    YH = data["p"]
    m = t > 0
    contrib = contributions(species, data, lib, A, Z, n0, T9)

    Y = data[species]
    observed = np.gradient(Y, t)
    modelled = sum(contrib.values()) if contrib else np.zeros_like(Y)
    contrib["jet-cloud + mixing (inferred)"] = observed - modelled

    # Rank by the largest share of total |rate| each process reaches at any time.
    total_abs = np.sum([np.abs(v) for v in contrib.values()], axis=0)
    total_abs[total_abs == 0] = np.inf
    peak_share = {k: float(np.max(np.abs(v)[m] / total_abs[m])) for k, v in contrib.items()}
    ranked = sorted(contrib, key=lambda k: -peak_share[k])
    shown = [k for k in ranked if peak_share[k] > 0.01][:top]
    if "jet-cloud + mixing (inferred)" not in shown:
        shown.append("jet-cloud + mixing (inferred)")
    rest = [k for k in contrib if k not in shown]

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(10, 8.5),
                                   gridspec_kw={"height_ratios": [1, 2.2]})
    for a, b in _jet_off_spans(t, data["delta_m_over_m0"]):
        for ax in (ax0, ax1):
            ax.axvspan(a, b, color="0.85", zorder=0, lw=0)
    ax0.plot(t[m], (Y / YH)[m], color="black", lw=1.8)
    ax0.set_yscale("log")
    ax0.set_ylabel(r"$Y/Y_\mathrm{H}$")
    sp_tex = "$" + _tex(species, A, Z) + "$" if species not in ("d", "t", "n", "p") else species
    ax0.set_title(f"{sp_tex}: abundance and the reactions that change it")
    ax0.grid(alpha=0.25, which="both", lw=0.5)

    colors = plt.cm.tab10.colors + plt.cm.Dark2.colors
    floor = 1e-40
    for i, k in enumerate(shown):
        v = contrib[k] / YH
        c = "0.35" if k.startswith("jet-cloud") else colors[i % len(colors)]
        pos = np.where(v > 0, v, np.nan)
        neg = np.where(v < 0, -v, np.nan)
        ax1.plot(t[m], np.maximum(pos[m], floor), "-", color=c, lw=1.6, label=k)
        ax1.plot(t[m], np.maximum(neg[m], floor), "--", color=c, lw=1.6)
    if rest:
        v = sum(contrib[k] for k in rest) / YH
        ax1.plot(t[m], np.where(v > 0, v, np.nan)[m], "-", color="0.7", lw=1, label=f"{len(rest)} smaller processes")
        ax1.plot(t[m], np.where(v < 0, -v, np.nan)[m], "--", color="0.7", lw=1)
    ax1.plot(t[m], np.abs(observed / YH)[m], ":", color="black", lw=1.6, label="|net| (seen in run)")

    vals = np.concatenate([np.abs(contrib[k] / YH)[m] for k in shown] + [np.abs(observed / YH)[m]])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        hi = np.nanmax(vals)
        ax1.set_ylim(hi * 1e-12, hi * 5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"$|dY/dt|$  ($Y/Y_\mathrm{H}$ per s)")
    ax1.grid(alpha=0.25, which="both", lw=0.5)
    ax1.legend(fontsize=8.5, loc="lower left", ncol=2, framealpha=0.9,
               title="solid = makes it, dashed = destroys it", title_fontsize=8.5)
    fig.text(0.01, 0.005, f"From {csv_name}; T9 = {T9:g}, n = {n0:.2g}(1+ΔM/M₀) cm⁻³. "
             "Grey bands: jet off. 'Inferred' = net change between saved rows minus thermonuclear + decay.",
             fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot dY/dt by reaction over time for chosen species.")
    ap.add_argument("species", nargs="+", help="species names as in config/species.json, e.g. 7Li t 3He")
    ap.add_argument("--csv", default=str(_ROOT / "outputs" / "abundance_history.csv"))
    ap.add_argument("--top", type=int, default=8, help="number of processes drawn individually")
    ap.add_argument("--outdir", default=str(_ROOT / "outputs" / "dydt_plots"))
    args = ap.parse_args()

    species_cfg = json.load(open(_ROOT / "config" / "species.json"))
    A = {k: int(v["A"]) for k, v in species_cfg.items() if not k.startswith("_")}
    Z = {k: int(v["Z"]) for k, v in species_cfg.items() if not k.startswith("_")}
    cloud = json.load(open(_ROOT / "config" / "cloud.json", encoding="utf-8"))
    n0, T9 = float(cloud["density_cm3"]), float(cloud["temperature_K"]) / 1e9

    import pandas as pd
    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset="t_s", keep="last").sort_values("t_s")
    data = {c: df[c].to_numpy(dtype=float) for c in df.columns if c != "step"}

    lib, _ = build_thermonuclear_library()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for sp in args.species:
        if sp not in data:
            print(f"skip {sp}: not a column in {args.csv}")
            continue
        out = Path(args.outdir) / f"dydt_{sp}.png"
        plot_species(sp, data, lib, A, Z, n0, T9, args.top, out, Path(args.csv).name)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
