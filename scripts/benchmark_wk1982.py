"""
benchmark_wk1982.py

Reproduce the warm-CNO network benchmark Famiano et al. (2002, Section 3)
used to validate their thermonuclear network: Wiescher & Kettner (1982,
ApJ 263, 891), Figs. 2-6.

Setup (W&K Section III):
    - constant density rho = 100 g cm^-3 and constant T9 in {0.05, 0.1,
      0.13, 0.16, 0.2}
    - initial composition solar (Cameron 1973)
    - run until hydrogen is depleted to 10% of its initial value (or the
      end of W&K's plotted time axis)
    - plot mass fractions X_i = A_i * Y_i of the O, F and Ne isotopes

Two rate sets are compared:
    A  "library"      : core/rate_library.py as-is (NACRE / CF88 / Bao),
                        which has NO proton captures on 17F or 18F
    B  "library + WK" : A plus W&K's own rates for 17F(p,g)18Ne (Table 2),
                        18F(p,a)15O (Table 4) and 18F(p,g)19Ne (Table 6)

Both add 18Ne and 19Ne (and their beta+ decays) to the species list.
Nothing in config/ or core/decay.py is modified: the extended species
list is written to a temporary file and the extra decays live in this
script.

Integration uses scipy's BDF (stiff) solver on the same RHS that
run_famiano.py's thermonuclear step uses (compute_thermonuclear_dydt),
so the test isolates the rates, species and decays from the step-size
controller.

Usage:
    python scripts/benchmark_wk1982.py
Writes outputs/wk1982_benchmark.png and prints a summary table.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "core"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rate_library import build_thermonuclear_library, _LogLinearTabulatedRate  # noqa: E402
from thermonuclear import ThermonuclearReaction, compute_thermonuclear_dydt  # noqa: E402
from decay import DECAY_TABLE, DecayChannel  # noqa: E402

RHO_G_CM3 = 100.0
N_A = 6.02214076e23
N_BARYON_CM3 = RHO_G_CM3 * N_A  # baryon number density, the CloudState.density_cm3 convention

# W&K Figs. 2-6: temperature -> (log t_min, log t_max) of the plotted axis.
CASES = {
    0.05: (4.0, 9.5),
    0.10: (0.5, 5.5),
    0.13: (0.5, 4.5),
    0.16: (0.5, 4.5),
    0.20: (0.5, 4.5),
}

PLOT_SPECIES = ["16O", "17O", "18O", "17F", "18F", "19F", "18Ne", "19Ne"]

# Extra species W&K track that config/species.json does not.
EXTRA_SPECIES = {"18Ne": {"A": 18, "Z": 10}, "19Ne": {"A": 19, "Z": 10}, "22Ne": {"A": 22, "Z": 10}}

EXTRA_DECAYS = {
    "18Ne": DecayChannel(1.672, {"18F": 1}, "beta+"),
    "19Ne": DecayChannel(17.22, {"19F": 1}, "beta+"),
}

# Solar composition, Cameron (1973), number abundances relative to Si = 1e6,
# split into isotopes with solar isotopic ratios.
_CAMERON_ELEMENTS = {"H": 3.18e10, "He": 2.21e9, "C": 1.18e7, "N": 3.74e6, "O": 2.15e7, "F": 2450.0, "Ne": 3.44e6}
_ISOTOPE_FRACTIONS = {
    "H": {"p": 1.0},
    "He": {"4He": 1.0},
    "C": {"12C": 0.9890, "13C": 0.0110},
    "N": {"14N": 0.99634, "15N": 0.00366},
    "O": {"16O": 0.99759, "17O": 0.00037, "18O": 0.00204},
    "F": {"19F": 1.0},
    "Ne": {"20Ne": 0.9048, "21Ne": 0.0027, "22Ne": 0.0925},
}

# W&K tabulated rates N_A<sv> [cm^3 mol^-1 s^-1], column 2 of Tables 2, 4, 6.
_WK_T9 = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.30, 0.40, 0.50, 0.60])
_WK_RATES = {
    # 17F(p,g)18Ne, Table 2 (truncated to the shared T9 grid; Table 2 also
    # lists 0.8 and 1.0, outside every benchmark temperature)
    ("17F", "p", "18Ne", None): [3.9e-15, 2.3e-12, 1.3e-10, 2.2e-9, 1.9e-8, 1.1e-7, 4.5e-7,
                                  1.5e-6, 4.1e-6, 1.5e-4, 1.6e-3, 9.4e-3, 4.1e-2],
    # 18F(p,a)15O, Table 4
    ("18F", "p", "15O", "4He"): [1.1e-11, 2.1e-10, 7.6e-9, 7.9e-8, 4.3e-7, 5.1e-6, 1.1e-4,
                                  1.3e-3, 9.7e-3, 4.0, 79.0, 480.0, 1.6e3],
    # 18F(p,g)19Ne, Table 6
    ("18F", "p", "19Ne", None): [1.1e-14, 4.7e-12, 2.6e-10, 4.5e-9, 4.0e-8, 2.3e-7, 1.4e-6,
                                  5.6e-6, 2.7e-5, 8.4e-3, 0.21, 1.6, 6.1],
}


def _wk_reactions() -> list[ThermonuclearReaction]:
    out = []
    for (target, proj, residual, ejectile), rates in _WK_RATES.items():
        products = {residual: 1}
        if ejectile:
            products[ejectile] = products.get(ejectile, 0) + 1
        label = f"{target}({proj},{'a' if ejectile else 'g'}){residual} [W&K 1982]"
        out.append(ThermonuclearReaction(
            reactants_stoich={target: 1, proj: 1},
            products_stoich=products,
            rate_func=_LogLinearTabulatedRate(_WK_T9, np.asarray(rates)),
            label=label,
            source="Wiescher & Kettner 1982",
        ))
    return out


def _build_library(species_json: Path, add_wk: bool):
    lib, _ = build_thermonuclear_library(species_json=species_json)
    if add_wk:
        for rxn in _wk_reactions():
            lib.add_reaction(rxn)
    return lib


def _initial_Y(species: list[str], A: dict[str, int]) -> np.ndarray:
    counts = {}
    for elem, n_elem in _CAMERON_ELEMENTS.items():
        for iso, frac in _ISOTOPE_FRACTIONS[elem].items():
            counts[iso] = n_elem * frac
    total_mass = sum(n * A[iso] for iso, n in counts.items())
    return np.array([counts.get(sp, 0.0) / total_mass for sp in species])


def _decay_rhs(Y: np.ndarray, index: dict[str, int], table: dict) -> np.ndarray:
    dYdt = np.zeros_like(Y)
    for parent, ch in table.items():
        i = index.get(parent)
        if i is None:
            continue
        r = np.log(2.0) / ch.half_life_s * Y[i]
        dYdt[i] -= r
        for d, m in ch.daughters.items():
            j = index.get(d)
            if j is not None:
                dYdt[j] += m * r
    return dYdt


def run_case(T9: float, lib, species: list[str], A: np.ndarray, Y0: np.ndarray, log_t_max: float):
    state = SimpleNamespace(cloud=SimpleNamespace(
        species=species, Y=Y0.copy(), density_cm3=N_BARYON_CM3, temperature_K=T9 * 1e9,
    ))
    index = {sp: i for i, sp in enumerate(species)}
    decays = {**DECAY_TABLE, **EXTRA_DECAYS}
    i_H = index["p"]
    Y_H0 = Y0[i_H]

    def rhs(_t, Y):
        Y = np.maximum(Y, 0.0)
        return compute_thermonuclear_dydt(state, lib, T9=T9, Y=Y) + _decay_rhs(Y, index, decays)

    def h_depleted(_t, Y):
        return Y[i_H] - 0.1 * Y_H0
    h_depleted.terminal = True

    t_eval = np.logspace(-2, log_t_max, 400)
    sol = solve_ivp(rhs, (0.0, 10 ** log_t_max), Y0, method="BDF", t_eval=t_eval,
                    rtol=1e-6, atol=1e-30, events=h_depleted)
    X = np.maximum(sol.y, 0.0) * A[:, None]
    return sol.t, X, sol


def _tex_label(sp: str) -> str:
    """'18Ne' -> '$^{18}$Ne'."""
    mass = sp.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return f"$^{{{mass}}}${sp[len(mass):]}" if mass else sp


def _label_lines(ax, curves: dict[str, tuple[np.ndarray, np.ndarray]], colors: dict[str, tuple],
                 xlim: tuple[float, float], ylim: tuple[float, float], min_dy: float = 0.45) -> None:
    """Put each species' name beside the right-most visible point of its curve,
    nudging labels apart vertically so they don't overlap."""
    anchors = []
    for sp, (x, y) in curves.items():
        vis = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        if not vis.any():
            continue
        k = np.flatnonzero(vis)[-1]
        anchors.append([sp, x[k], y[k]])
    # Labels within ~10% of the axis width of each other in x compete for the
    # same vertical space; separate them bottom-up, then clamp to the axis.
    anchors.sort(key=lambda a: a[2])
    dx_close = 0.1 * (xlim[1] - xlim[0])
    for i in range(1, len(anchors)):
        for j in range(i):
            if abs(anchors[i][1] - anchors[j][1]) < dx_close and anchors[i][2] - anchors[j][2] < min_dy:
                anchors[i][2] = anchors[j][2] + min_dy
    top = ylim[1] - 0.3
    for i in reversed(range(len(anchors))):
        limit = top if i == len(anchors) - 1 else anchors[i + 1][2] - min_dy
        anchors[i][2] = min(anchors[i][2], limit)
    for sp, x, y in anchors:
        ax.annotate(_tex_label(sp), xy=(x, y), xytext=(4, 0), textcoords="offset points",
                    color=colors[sp], fontsize=9, fontweight="bold", ha="left", va="center",
                    annotation_clip=False)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species_cfg = json.load(open(_ROOT / "config" / "species.json"))
    species_cfg.update(EXTRA_SPECIES)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(species_cfg, f)
        species_json = Path(f.name)

    libs = {"library": _build_library(species_json, add_wk=False),
            "library + WK": _build_library(species_json, add_wk=True)}

    species = [k for k in species_cfg if not k.startswith("_")]
    A_map = {k: int(v["A"]) for k, v in species_cfg.items() if not k.startswith("_")}
    A = np.array([A_map[s] for s in species], dtype=float)
    Y0 = _initial_Y(species, A_map)
    idx = {s: i for i, s in enumerate(species)}

    print("Initial log X:", {s: round(float(np.log10(A[idx[s]] * Y0[idx[s]])), 2)
                             for s in PLOT_SPECIES if Y0[idx[s]] > 0})

    colors = dict(zip(PLOT_SPECIES, plt.cm.tab10.colors))
    fig, axes = plt.subplots(len(CASES), 1, figsize=(8, 4 * len(CASES)))
    for ax, (T9, (lt0, lt1)) in zip(axes, CASES.items()):
        print(f"\n=== T9 = {T9}, rho = {RHO_G_CM3:g} g/cm^3 ===")
        curves = {}
        for name, style in (("library + WK", "-"), ("library", ":")):
            t, X, sol = run_case(T9, libs[name], species, A, Y0, lt1)
            stop = (f"H at 10% at log t = {np.log10(sol.t_events[0][0]):.2f}"
                    if sol.t_events[0].size else "reached end of plotted axis")
            print(f"  [{name}] {sol.message} ({stop}, {sol.nfev} RHS calls)")
            m = t > 0
            for sp in PLOT_SPECIES:
                x, y = np.log10(t[m]), np.log10(np.maximum(X[idx[sp], m], 1e-30))
                ax.plot(x, y, style, color=colors[sp])
                if style == "-":
                    curves[sp] = (x, y)
            # Summary: log X at a few times along W&K's axis
            for lt in np.arange(np.ceil(lt0), lt1 + 1e-9, 1.0):
                k = np.searchsorted(np.log10(t[m]), lt)
                if k >= m.sum():
                    break
                row = "  ".join(f"{sp}={np.log10(max(X[idx[sp], m][k], 1e-30)):6.1f}" for sp in PLOT_SPECIES)
                print(f"    log t={lt:4.1f}: {row}")
        ax.set_xlim(lt0, lt1)
        ax.set_ylim(-12.5, -1.5)
        ax.set_xlabel("log t (s)")
        ax.set_ylabel("log X")
        ax.set_title(f"T9 = {T9}, rho = 100 g cm$^{{-3}}$  (solid: library + W&K rates, dotted: library only)")
        ax.grid(alpha=0.3)
        _label_lines(ax, curves, colors, (lt0, lt1), (-12.5, -1.5))
    fig.tight_layout()
    out = _ROOT / "outputs" / "wk1982_benchmark.png"
    fig.savefig(out, dpi=120)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
