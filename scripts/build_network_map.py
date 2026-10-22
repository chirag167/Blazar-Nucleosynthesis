"""
build_network_map.py

Build an interactive reaction-network map (outputs/reaction_network_map.html):
a chart of the nuclides where clicking a nucleus lists every reaction that
makes or destroys it, ranked by its actual flux at a snapshot of a run.

Data sources:
    - thermonuclear reactions: core/rate_library.build_thermonuclear_library()
    - radioactive decays:      core/decay.DECAY_TABLE
    - jet-cloud reactions:     core/reactions.ReactionLibrary (same loader
                               arguments as run_famiano.py); listed but not
                               ranked, since their rates depend on the jet
                               spectrum rather than on cloud abundances alone
    - abundances:              last row of the history CSV (default
                               outputs/abundance_history.csv)

The cloud density at the snapshot is taken as n0 * (1 + DM/M0) from
config/cloud.json and the CSV's delta_m_over_m0 column (exact for the
constant-volume growth run_famiano.py uses; for a jet-free run DM/M0 ~ 0).

Usage:
    python scripts/build_network_map.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "core"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rate_library import build_thermonuclear_library  # noqa: E402
from decay import DECAY_TABLE  # noqa: E402
from reactions import ReactionLibrary  # noqa: E402

N_A = 6.02214076e23
_TEMPLATE = Path(__file__).resolve().parent / "network_map_template.html"
_JET_PROJECTILES = ["p", "n", "d", "t", "3He", "4He", "6Li", "7Li"]  # as run_famiano.py
_INJECTED_PROJECTILES = {"p", "4He"}  # jet species; secondaries are not followed as projectiles

_ORIG_8BE = re.compile(r"^(?P<lhs>[^>]+)->(?P<rhs>.+?) \[8Be->2a\]$")


def _parse_side(side: str) -> dict[str, int]:
    """Parse one side of ThermonuclearReaction.name(), e.g. '4He+8Be',
    '2d' (2 x d) or '34He' (3 x 4He)."""
    out: dict[str, int] = {}
    for tok in side.split("+"):
        if re.fullmatch(r"\d+(4He|3He)", tok) and len(tok) > 3:
            mult, sp = int(tok[:-3]), tok[-3:]
        elif re.fullmatch(r"\d[a-z]", tok):
            mult, sp = int(tok[0]), tok[1:]
        else:
            mult, sp = 1, tok
        out[sp] = out.get(sp, 0) + mult
    return out


def _notation(reactants: dict[str, int], products: dict[str, int], A: dict[str, int]) -> dict:
    """target(projectile, ejectiles)residual pieces, heaviest reactant as target."""
    rlist = [sp for sp, m in reactants.items() for _ in range(m)]
    rlist.sort(key=lambda s: (A.get(s, 0), s), reverse=True)
    target, projectile = rlist[0], rlist[1]
    plist = [sp for sp, m in products.items() for _ in range(m)]
    plist.sort(key=lambda s: (A.get(s, 0), s), reverse=True)
    residual, ejectiles = plist[0], plist[1:]
    return {"target": target, "projectile": projectile, "ejectiles": ejectiles, "residual": residual}


def _load_snapshot(csv_path: Path) -> tuple[dict[str, float], float, float]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    t = float(last["t_s"])
    dm = float(last.get("delta_m_over_m0", 0.0) or 0.0)
    Y = {k: float(v) for k, v in last.items() if k not in ("step", "t_s", "delta_m_over_m0")}
    return Y, t, dm


def build(csv_path: Path, out_path: Path) -> None:
    species_cfg = json.load(open(_ROOT / "config" / "species.json"))
    cloud_cfg = json.load(open(_ROOT / "config" / "cloud.json", encoding="utf-8"))
    A = {k: int(v["A"]) for k, v in species_cfg.items() if not k.startswith("_")}
    Z = {k: int(v["Z"]) for k, v in species_cfg.items() if not k.startswith("_")}

    Y, t_snap, dm = _load_snapshot(csv_path)
    n0 = float(cloud_cfg["density_cm3"])
    n = n0 * (1.0 + dm)
    T9 = float(cloud_cfg["temperature_K"]) / 1e9
    Y_H = Y.get("p", 0.75) or 0.75

    reactions = []

    # --- thermonuclear ---
    lib, _ = build_thermonuclear_library()
    for i, r in enumerate(lib.reactions):
        sv = float(r.rate_func(T9))                     # per-pair <sigma v> [cm^3/s]
        prod_display = dict(r.products_stoich)
        note = ""
        if r.label and "[8Be->2a]" in r.label:
            m = _ORIG_8BE.match(r.label)
            if m:
                prod_display = _parse_side(m.group("rhs"))
                note = "8Be breaks up into 2α immediately"
        delta = 1 if len(r.reactants_stoich) == 1 else 0
        ys = [Y.get(sp, 0.0) for sp, mult in r.reactants_stoich.items() for _ in range(mult)]
        events = n * sv * ys[0] * ys[1] / (1 + delta) if len(ys) == 2 else 0.0  # per baryon per s
        reactions.append({
            "id": f"tn{i}", "kind": "thermonuclear", "source": r.source,
            "reactants": r.reactants_stoich, "products": r.products_stoich,
            "nota": _notation(r.reactants_stoich, prod_display, A),
            "note": note, "rate": sv * N_A, "events": events / Y_H,
        })

    # --- decays ---
    for parent, ch in DECAY_TABLE.items():
        lam = math.log(2.0) / ch.half_life_s
        reactions.append({
            "id": f"dk_{parent}", "kind": "decay", "source": "decay table (core/decay.py)",
            "reactants": {parent: 1}, "products": dict(ch.daughters),
            "mode": ch.mode, "half_life_s": ch.half_life_s,
            "events": lam * Y.get(parent, 0.0) / Y_H,
        })

    # --- jet-cloud ---
    jet_lib = ReactionLibrary.from_directories(
        base_dir=_ROOT, projectile_species=_JET_PROJECTILES, target_species=None,
        include_group1=True, include_group2=True,
    )
    seen = set()
    for j, r in enumerate(jet_lib.reactions):
        key = r.name()
        if key in seen:
            continue
        seen.add(key)
        reactants = {k: v for k, v in r.reactants_stoich.items()}
        products = {k: v for k, v in r.products_stoich.items()}
        if not all(sp in A for sp in list(reactants) + list(products)):
            continue
        reactions.append({
            "id": f"jc{j}", "kind": "jet", "source": f"jet-cloud {r.group}",
            "reactants": reactants, "products": products,
            "nota": {"target": r.target, "projectile": r.projectile,
                     "ejectiles": [sp for sp, m in products.items() if sp != r.residual for _ in range(m)]
                                  + [r.residual] * (products.get(r.residual, 1) - 1),
                     "residual": r.residual},
            "active": r.projectile in _INJECTED_PROJECTILES,
            "elastic": bool(getattr(r, "is_elastic", False)),
        })

    used = {sp for r in reactions for side in ("reactants", "products") for sp in r[side]}
    used |= {sp for sp, y in Y.items() if y > 0}
    species = [{"name": sp, "A": A[sp], "Z": Z[sp], "N": A[sp] - Z[sp],
                "y": Y.get(sp, 0.0) / Y_H, "inNetwork": sp in Y,
                "decay": (DECAY_TABLE[sp].mode if sp in DECAY_TABLE else None),
                "half_life_s": (DECAY_TABLE[sp].half_life_s if sp in DECAY_TABLE else None)}
               for sp in A if sp in used]

    data = {
        "snapshot": {"csv": str(csv_path.relative_to(_ROOT)) if csv_path.is_relative_to(_ROOT) else str(csv_path),
                     "t_s": t_snap, "dm_over_m0": dm, "n_cm3": n, "T9": T9},
        "species": species,
        "reactions": reactions,
    }
    html = _TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    out_path.write_text(html, encoding="utf-8")
    n_active = sum(1 for r in reactions if r.get("events", 0) > 0)
    print(f"Wrote {out_path}: {len(species)} species, {len(reactions)} reactions "
          f"({n_active} with nonzero flux at t = {t_snap:.4g} s).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", default=str(_ROOT / "outputs" / "abundance_history.csv"))
    ap.add_argument("--out", default=str(_ROOT / "outputs" / "reaction_network_map.html"))
    args = ap.parse_args()
    build(Path(args.csv).resolve(), Path(args.out).resolve())


if __name__ == "__main__":
    main()
