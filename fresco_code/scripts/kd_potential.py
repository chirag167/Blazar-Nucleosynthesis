#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from string import Template
import math
import sys

PARTICLES = {
    "n":   {"name": "n",   "z": 0, "a": 1, "spin": 0.5},
    "p":   {"name": "p",   "z": 1, "a": 1, "spin": 0.5},
    "d":   {"name": "d",   "z": 1, "a": 2, "spin": 1.0},
    "t":   {"name": "t",   "z": 1, "a": 3, "spin": 0.5},
    "he3": {"name": "3He", "z": 2, "a": 3, "spin": 0.5},
    "a":   {"name": "4He", "z": 2, "a": 4, "spin": 0.0},
    "li7": {"name": "7Li", "z": 3, "a": 7, "spin": 1.5},
}


def normalize_projectile(projectile: str) -> str:
    p = projectile.lower()

    aliases = {
        "proton": "p",
        "neutron": "n",
        "deuteron": "d",
        "triton": "t",
        "h3": "t",
        "3h": "t",
        "he3": "he3",
        "3he": "he3",
        "alpha": "a",
        "he4": "a",
        "4he": "a",
        "li7": "li7",
        "7li": "li7",
    }

    return aliases.get(p, p)


def target_name_and_spin(z: int, a: int) -> tuple[str, float]:
    if z == 1 and a == 1:
        return "p", 0.5
    if z == 1 and a == 2:
        return "d", 1.0
    if z == 1 and a == 3:
        return "t", 0.5
    if z == 2 and a == 3:
        return "3He", 0.5
    if z == 2 and a == 4:
        return "4He", 0.0
    if z == 3 and a == 7:
        return "7Li", 1.5

    return f"A{a}Z{z}", 0.0

def simple_cluster_parameters(projectile: str, z: int, a: int, energy_mev: float) -> dict[str, float]:
    """
    Generic placeholder Woods-Saxon potential for d, t, 3He, 4He, 7Li projectiles.

    This makes FRESCO run for entrance-channel studies, but it is not a
    validated global optical model. Replace these values with literature
    optical potentials for quantitative work.
    """

    projectile = normalize_projectile(projectile)
    pinfo = PARTICLES[projectile]

    A_p = float(pinfo["a"])
    Z_p = float(pinfo["z"])
    A_t = float(a)

    if a <= 0 or z < 0 or a < z:
        raise ValueError("Invalid target: require A > 0, Z >= 0, and A >= Z.")

    target_name, target_j = target_name_and_spin(z, a)

    # Approximate touching Coulomb radius.
    coul_r = 1.25 * (A_p ** (1.0 / 3.0) + A_t ** (1.0 / 3.0))

    # Placeholder optical potential.
    vv = 70.0
    rv = 1.25
    av = 0.65

    wv = 15.0
    rwv = 1.25
    awv = 0.65

    wd = 5.0
    rwd = 1.30
    awd = 0.65

    # Spin-orbit off for spin-0 alpha, small placeholder otherwise.
    if pinfo["spin"] == 0.0:
        vso = 0.0
    else:
        vso = 3.0

    rso = 1.0
    aso = 0.65

    return {
        "ENERGY": energy_mev,

        "PROJECTILE_NAME": pinfo["name"],
        "PROJECTILE_MASS": pinfo["a"],
        "PROJECTILE_Z": pinfo["z"],
        "PROJECTILE_J": pinfo["spin"],

        "TARGET_NAME": target_name,
        "TARGET_A": a,
        "TARGET_Z": z,
        "TARGET_J": target_j,

        "COUL_R": coul_r,

        "VV": vv,
        "RV": rv,
        "AV": av,

        "WV": wv,
        "RWV": rwv,
        "AWV": awv,

        "WD": wd,
        "RWD": rwd,
        "AWD": awd,

        "VSO": vso,
        "RSO": rso,
        "ASO": aso,
    }

def kd_parameters(projectile: str, z: int, a: int, energy_mev: float) -> dict[str, float]:
    projectile = normalize_projectile(projectile)
    """
    Approximate implementation of the global Koning-Delaroche 2003
    neutron/proton optical model potential.

    Based on the global OMP formulas in Koning & Delaroche,
    Nucl. Phys. A 713 (2003) 231-310.

    This returns parameters in the same names used by reaction.in.tpl.
    """

    if projectile not in PARTICLES:
        raise ValueError(f"Unsupported projectile: {projectile}")

    if projectile in {"p", "n"}:
        # Keep your existing KD logic for nucleons.
        pass
    else:
        return simple_cluster_parameters(projectile, z, a, energy_mev)

    """
    if not (0.001 <= energy_mev <= 200.0):
        raise ValueError("KD global potential is intended for about 0.001 <= E <= 200 MeV.")

    if not (24 <= a <= 209):
        raise ValueError("KD global potential is intended for about 24 <= A <= 209.")

    """

    E = energy_mev
    A = float(a)
    Z = float(z)
    N = A - Z
    alpha = (N - Z) / A

    A13 = A ** (1.0 / 3.0)
    A_m13 = A ** (-1.0 / 3.0)
    A_m23 = A ** (-2.0 / 3.0)
    A_m53 = A ** (-5.0 / 3.0)

    # Shared geometry parameters from global KD OMP.
    rv = 1.3039 - 0.4054 * A_m13
    av = 0.6778 - 1.487e-4 * A

    rwd = 1.3424 - 0.01585 * A13

    # KD uses different surface diffuseness for neutron and proton.
    if projectile == "n":
        awd = 0.5446 - 1.656e-4 * A
    else:
        awd = 0.5187 + 5.205e-4 * A

    rso = 1.1854 - 0.647 * A_m13
    aso = 0.59

    # Coulomb radius parameter for protons.
    rc = 1.198 + 0.697 * A_m23 + 12.994 * A_m53
    coul_r = rc * A13

    # The code template uses separate volume-imaginary radius parameters.
    # KD global geometry uses the same geometry for the real volume and
    # imaginary volume central terms.
    rwv = rv
    awv = av

    if projectile == "n":
        Ef = -11.2814 + 0.02646 * A

        v1 = 59.30 - 21.0 * alpha - 0.024 * A
        v2 = 0.007228 - 1.48e-6 * A
        v3 = 1.994e-5 - 2.0e-8 * A
        v4 = 7.0e-9

        w1 = 12.195 + 0.0167 * A
        w2 = 73.55 + 0.0795 * A

        d1 = 16.0 - 16.0 * alpha
        d2 = 0.0180 + 0.003802 / (1.0 + math.exp((A - 156.0) / 8.0))
        d3 = 11.5

        vso1 = 5.922 + 0.0030 * A
        vso2 = 0.0040

        x = E - Ef

        vv = v1 * (1.0 - v2 * x + v3 * x**2 - v4 * x**3)
        wv = w1 * x**2 / (x**2 + w2**2)
        wd = d1 * x**2 / (x**2 + d3**2) * math.exp(-d2 * x)
        vso = vso1 * math.exp(-vso2 * x)

    else:
        Ef = -8.4075 + 0.01378 * A

        v1 = 59.30 + 21.0 * alpha - 0.024 * A
        v2 = 0.007067 + 4.23e-6 * A
        v3 = 1.729e-5 + 1.136e-8 * A
        v4 = 7.0e-9

        w1 = 14.667 + 0.009629 * A
        w2 = 73.55 + 0.0795 * A

        d1 = 16.0 + 16.0 * alpha
        d2 = 0.0180 + 0.003802 / (1.0 + math.exp((A - 156.0) / 8.0))
        d3 = 11.5

        vso1 = 5.922 + 0.0030 * A
        vso2 = 0.0040

        x = E - Ef

        # KD proton Coulomb correction.
        Vc = 1.73 / rc * Z * A_m13

        vv = (
            v1 * (1.0 - v2 * x + v3 * x**2 - v4 * x**3)
            + Vc * v1 * (v2 - 2.0 * v3 * x + 3.0 * v4 * x**2)
        )

        wv = w1 * x**2 / (x**2 + w2**2)
        wd = d1 * x**2 / (x**2 + d3**2) * math.exp(-d2 * x)
        vso = vso1 * math.exp(-vso2 * x)

    pinfo = PARTICLES[projectile]
    target_name, target_j = target_name_and_spin(z, a)

    return {
        "ENERGY": energy_mev,

        "PROJECTILE_NAME": pinfo["name"],
        "PROJECTILE_MASS": pinfo["a"],
        "PROJECTILE_Z": pinfo["z"],
        "PROJECTILE_J": pinfo["spin"],

        "TARGET_NAME": target_name,
        "TARGET_A": a,
        "TARGET_Z": z,
        "TARGET_J": target_j,

        "COUL_R": coul_r,

        "VV": vv,
        "RV": rv,
        "AV": av,

        "WV": wv,
        "RWV": rwv,
        "AWV": awv,

        "WD": wd,
        "RWD": rwd,
        "AWD": awd,

        "VSO": vso,
        "RSO": rso,
        "ASO": aso,
    }


def render_template(template_path: Path, output_path: Path, values: dict[str, float], projectile: str) -> None:
    text = template_path.read_text()

    for key in sorted(values.keys(), key=len, reverse=True):
        val = values[key]
        if isinstance(val, float):
            text = text.replace(key, f"{val:.6g}")
        else:
            text = text.replace(key, str(val))

    output_path.write_text(text)
    unreplaced = [
        token for token in [
            "ENERGY",
            "PROJECTILE_NAME",
            "PROJECTILE_MASS",
            "PROJECTILE_Z",
            "PROJECTILE_J",
            "TARGET_NAME",
            "TARGET_A",
            "TARGET_Z",
            "TARGET_J",
            "COUL_R",
            "VV", "RV", "AV",
            "WV", "RWV", "AWV",
            "WD", "RWD", "AWD",
            "VSO", "RSO", "ASO",
        ]
        if token in text
    ]

    if unreplaced:
        raise ValueError(f"Unreplaced template placeholders: {unreplaced}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projectile",
        required=True,
        choices=["p", "n", "d", "t", "he3", "3He", "a", "alpha", "he4", "4He", "li7", "7Li"],
    )
    parser.add_argument("--z", required=True, type=int)
    parser.add_argument("--a", required=True, type=int)
    parser.add_argument("--energy", required=True, type=float)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    params = kd_parameters(args.projectile, args.z, args.a, args.energy)
    render_template(args.template, args.output, params, args.projectile)


if __name__ == "__main__":
    main()