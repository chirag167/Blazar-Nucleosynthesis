#!/usr/bin/env python3
"""
Generate FRESCO NAMELIST input files for the two-body transfer reactions
from Famiano et al. (2002) Table 2 Group I.

Only reactions with two-body final states are supported here.  The
remaining Group I reactions have 3+ body exits and cannot be computed
with FRESCO.

Usage:
    python generate_transfer_input.py --reaction LABEL --energy E_MeV --output FILE
    python generate_transfer_input.py --list
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kd_potential import kd_parameters, simple_cluster_parameters, normalize_projectile

# AME mass excesses Delta (MeV)
DELTA = {
    "n":    8.07131,
    "p":    7.28897,
    "d":   13.13572,
    "t":   14.94981,   # 3H
    "he3": 14.93121,   # 3He
    "a":    2.42492,   # 4He
    "li7": 14.90788,   # 7Li
    "be7": 15.76910,   # 7Be
}


def coul_r(a1, a2, r0=1.25):
    """Total Coulomb touching radius R_C = r0*(A1^1/3 + A2^1/3) in fm."""
    return r0 * (a1 ** (1.0 / 3.0) + a2 ** (1.0 / 3.0))


def get_omp(proj_key, targ_z, targ_a, energy_mev):
    """Return OMP parameter dict for projectile proj_key on target (Z, A)."""
    pnorm = normalize_projectile(proj_key)
    if pnorm in ("p", "n"):
        return kd_parameters(pnorm, targ_z, targ_a, energy_mev)
    return simple_cluster_parameters(pnorm, targ_z, targ_a, energy_mev)


# ---------------------------------------------------------------------------
# Reaction database.
#
# Each reaction is a two-body transfer A+B -> C+D treated in one-step DWBA
# (finite-range, KIND=7 coupling in FRESCO).
#
# Partition 1 = entrance channel (proj, targ)
# Partition 2 = exit channel    (proj, targ)
# qval        = ground-state reaction Q-value in MeV
#
# overlaps: two entries, one for each side of the transfer.
#   kn      : overlap index (kn=1 or 2)
#   in_     : 1 = projectile side, 2 = target side
#   kbpot   : bound-state potential kp index (3 or 4)
#   nn      : number of radial nodes including origin (1 = lowest state)
#   l       : orbital angular momentum of transferred cluster
#   sn      : intrinsic spin of transferred cluster
#   j       : j = l+sn (jj coupling, KIND=0 overlap)
#   be      : separation energy (positive = bound), MeV
#   core_a  : mass number of core nucleus (for Coulomb in bound-state pot)
#   core_z  : charge of core nucleus
#   xfer_z  : charge of transferred cluster (0 -> no Coulomb in bound-state pot)
#   comment : human-readable label
#
# cfps: spectroscopic amplitudes (amplitude = sqrt(SF), NOT SF itself).
#   in_  : 1=projectile side, 2=target side
#   ib   : state index of composite nucleus (1 = ground state)
#   ia   : state index of core nucleus      (1 = ground state)
#   kn   : overlap index this amplitude belongs to
#   a    : spectroscopic amplitude (1.0 = unit SF for closed/simple shells)
# ---------------------------------------------------------------------------

REACTIONS = {

    # -----------------------------------------------------------------------
    # Reaction 1: p + 4He -> d + 3He
    # Transferred particle: neutron (A=1, Z=0)
    # Spectroscopic structure:
    #   4He = 3He + n  (1s1/2 neutron, BE = S_n(4He) = 20.578 MeV)
    #   d   = p  + n  (3S1 deuteron,  BE = 2.225 MeV)
    # -----------------------------------------------------------------------
    "p_4He_d_3He": {
        "description": "p + 4He -> d + 3He (neutron pickup)",
        "entrance": {
            "proj": {"key": "p",   "name": "p",   "a": 1, "z": 1, "spin": 0.5},
            "targ": {"key": "a",   "name": "4He", "a": 4, "z": 2, "spin": 0.0},
        },
        "exit": {
            "proj": {"key": "d",   "name": "d",   "a": 2, "z": 1, "spin": 1.0},
            "targ": {"key": "he3", "name": "3He", "a": 3, "z": 2, "spin": 0.5},
        },
        "qval": DELTA["p"] + DELTA["a"] - DELTA["d"] - DELTA["he3"],
        "xfer": {"name": "n", "a": 1, "z": 0},
        "overlaps": [
            {
                # n bound in 4He with 3He as core (target side: part1-targ=4He, part2-targ=3He)
                "kn": 1, "in_": 2, "kbpot": 3,
                "nn": 1, "l": 0, "sn": 0.5, "j": 0.5,
                "be": DELTA["he3"] + DELTA["n"] - DELTA["a"],  # S_n(4He) = 20.578 MeV
                "core_a": 3, "core_z": 2, "xfer_z": 0,
                "comment": "n in 4He: composite=4He, core=3He",
            },
            {
                # n bound in d with p as core (proj side: part1-proj=p, part2-proj=d)
                "kn": 2, "in_": 1, "kbpot": 4,
                "nn": 1, "l": 0, "sn": 0.5, "j": 0.5,
                "be": DELTA["p"] + DELTA["n"] - DELTA["d"],    # S_n(d) = 2.225 MeV
                "core_a": 1, "core_z": 1, "xfer_z": 0,
                "comment": "n in d: composite=d, core=p",
            },
        ],
        "cfps": [
            {"in_": 2, "ib": 1, "ia": 1, "kn": 1, "a": 1.0},
            {"in_": 1, "ib": 1, "ia": 1, "kn": 2, "a": 1.0},
        ],
        "energy_range": [5.0, 400.0, 5.0],
    },

    # -----------------------------------------------------------------------
    # Reaction 2: 4He + 4He -> n + 7Be
    # Transferred particle: 3He cluster (A=3, Z=2)
    # Spectroscopic structure:
    #   7Be = 4He + 3He  (1p3/2, BE = S_3He(7Be) = 1.587 MeV)
    #   4He = n   + 3He  (1s1/2, BE = S_3He(4He) = 20.578 MeV)
    # -----------------------------------------------------------------------
    "4He_4He_n_7Be": {
        "description": "4He + 4He -> n + 7Be (3He cluster transfer)",
        "entrance": {
            "proj": {"key": "a",   "name": "4He", "a": 4, "z": 2, "spin": 0.0},
            "targ": {"key": "a",   "name": "4He", "a": 4, "z": 2, "spin": 0.0},
        },
        "exit": {
            "proj": {"key": "n",   "name": "n",   "a": 1, "z": 0, "spin": 0.5},
            "targ": {"key": "be7", "name": "7Be", "a": 7, "z": 4, "spin": 1.5},
        },
        "qval": 2.0 * DELTA["a"] - DELTA["n"] - DELTA["be7"],
        "xfer": {"name": "3He", "a": 3, "z": 2},
        "overlaps": [
            {
                # 3He bound in 7Be with 4He as core (target side: part1-targ=4He, part2-targ=7Be)
                # 7Be = 4He + 3He; parity of 7Be (3/2-) requires l=1
                "kn": 1, "in_": 2, "kbpot": 3,
                "nn": 1, "l": 1, "sn": 0.5, "j": 1.5,
                "be": DELTA["a"] + DELTA["he3"] - DELTA["be7"],  # S_3He(7Be) = 1.587 MeV
                "core_a": 4, "core_z": 2, "xfer_z": 2,
                "comment": "3He in 7Be: composite=7Be, core=4He",
            },
            {
                # 3He bound in 4He with n as core (proj side: part1-proj=4He, part2-proj=n)
                # 4He = n + 3He; parity of 4He (0+) with 3He(1/2+) and n(1/2+) requires l=0
                "kn": 2, "in_": 1, "kbpot": 4,
                "nn": 1, "l": 0, "sn": 0.5, "j": 0.5,
                "be": DELTA["n"] + DELTA["he3"] - DELTA["a"],    # S_3He(4He) = 20.578 MeV
                "core_a": 1, "core_z": 0, "xfer_z": 2,
                "comment": "3He in 4He: composite=4He, core=n (no Coulomb: neutral core)",
            },
        ],
        "cfps": [
            {"in_": 2, "ib": 1, "ia": 1, "kn": 1, "a": 1.0},
            {"in_": 1, "ib": 1, "ia": 1, "kn": 2, "a": 1.0},
        ],
        "energy_range": [10.0, 400.0, 5.0],
    },

    # -----------------------------------------------------------------------
    # Reaction 3: 4He + 4He -> p + 7Li
    # Transferred particle: triton t = 3H (A=3, Z=1)
    # Spectroscopic structure:
    #   7Li = 4He + t  (1p3/2, BE = S_t(7Li) = 2.467 MeV)
    #   4He = p   + t  (1s1/2, BE = S_t(4He) = 19.814 MeV)
    # -----------------------------------------------------------------------
    "4He_4He_p_7Li": {
        "description": "4He + 4He -> p + 7Li (triton transfer)",
        "entrance": {
            "proj": {"key": "a",   "name": "4He", "a": 4, "z": 2, "spin": 0.0},
            "targ": {"key": "a",   "name": "4He", "a": 4, "z": 2, "spin": 0.0},
        },
        "exit": {
            "proj": {"key": "p",   "name": "p",   "a": 1, "z": 1, "spin": 0.5},
            "targ": {"key": "li7", "name": "7Li", "a": 7, "z": 3, "spin": 1.5},
        },
        "qval": 2.0 * DELTA["a"] - DELTA["p"] - DELTA["li7"],
        "xfer": {"name": "t", "a": 3, "z": 1},
        "overlaps": [
            {
                # t bound in 7Li with 4He as core (target side: part1-targ=4He, part2-targ=7Li)
                # 7Li = 4He + t; parity of 7Li (3/2-) requires l=1
                "kn": 1, "in_": 2, "kbpot": 3,
                "nn": 1, "l": 1, "sn": 0.5, "j": 1.5,
                "be": DELTA["a"] + DELTA["t"] - DELTA["li7"],    # S_t(7Li) = 2.467 MeV
                "core_a": 4, "core_z": 2, "xfer_z": 1,
                "comment": "t in 7Li: composite=7Li, core=4He",
            },
            {
                # t bound in 4He with p as core (proj side: part1-proj=4He, part2-proj=p)
                # 4He = p + t; parity of 4He (0+) with p(1/2+) and t(1/2+) requires l=0
                "kn": 2, "in_": 1, "kbpot": 4,
                "nn": 1, "l": 0, "sn": 0.5, "j": 0.5,
                "be": DELTA["p"] + DELTA["t"] - DELTA["a"],      # S_t(4He) = 19.814 MeV
                "core_a": 1, "core_z": 1, "xfer_z": 1,
                "comment": "t in 4He: composite=4He, core=p",
            },
        ],
        "cfps": [
            {"in_": 2, "ib": 1, "ia": 1, "kn": 1, "a": 1.0},
            {"in_": 1, "ib": 1, "ia": 1, "kn": 2, "a": 1.0},
        ],
        "energy_range": [10.0, 400.0, 5.0],
    },

    # -----------------------------------------------------------------------
    # Reaction 4: p + 7Li -> 4He + 4He
    # Transferred particle: triton t = 3H (A=3, Z=1)
    # Same spectroscopic structure as reaction 3 but with reversed role of
    # entrance/exit.
    #
    # Note: the exit channel has two identical 4He nuclei.  FRESCO treats
    # them as distinguishable here (no Bose symmetrisation).  This gives
    # the direct amplitude only; the exchange amplitude (incident particle
    # scattering at pi-theta) is not included.  Full symmetrised calculation
    # requires fexch=.true. in the exit-channel &STATES block.
    # -----------------------------------------------------------------------
    "p_7Li_4He_4He": {
        "description": "p + 7Li -> 4He + 4He (triton transfer, exothermic)",
        "entrance": {
            "proj": {"key": "p",   "name": "p",   "a": 1, "z": 1, "spin": 0.5},
            "targ": {"key": "li7", "name": "7Li", "a": 7, "z": 3, "spin": 1.5},
        },
        "exit": {
            "proj": {"key": "a",   "name": "4He", "a": 4, "z": 2, "spin": 0.0},
            "targ": {"key": "a",   "name": "4He", "a": 4, "z": 2, "spin": 0.0},
        },
        "qval": DELTA["p"] + DELTA["li7"] - 2.0 * DELTA["a"],
        "xfer": {"name": "t", "a": 3, "z": 1},
        "overlaps": [
            {
                # t bound in 7Li with 4He as core (target side: part1-targ=7Li, part2-targ=4He)
                "kn": 1, "in_": 2, "kbpot": 3,
                "nn": 1, "l": 1, "sn": 0.5, "j": 1.5,
                "be": DELTA["a"] + DELTA["t"] - DELTA["li7"],    # S_t(7Li) = 2.467 MeV
                "core_a": 4, "core_z": 2, "xfer_z": 1,
                "comment": "t in 7Li: composite=7Li, core=4He",
            },
            {
                # t bound in 4He with p as core (proj side: part1-proj=p, part2-proj=4He)
                "kn": 2, "in_": 1, "kbpot": 4,
                "nn": 1, "l": 0, "sn": 0.5, "j": 0.5,
                "be": DELTA["p"] + DELTA["t"] - DELTA["a"],      # S_t(4He) = 19.814 MeV
                "core_a": 1, "core_z": 1, "xfer_z": 1,
                "comment": "t in 4He: composite=4He, core=p",
            },
        ],
        "cfps": [
            {"in_": 2, "ib": 1, "ia": 1, "kn": 1, "a": 1.0},
            {"in_": 1, "ib": 1, "ia": 1, "kn": 2, "a": 1.0},
        ],
        "energy_range": [5.0, 400.0, 5.0],
    },
}


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def _pot_lines_omp(kp, omp, proj_a, targ_a, targ_z):
    """Return &POT lines for one channel's optical potential."""
    rc = coul_r(proj_a, targ_a)
    lines = [
        f" &POT kp={kp} type=0 p1={targ_a} p2={targ_z} p3={rc:.4f} /",
        (f" &POT kp={kp} type=1"
         f" p1={omp['VV']:.3f} p2={omp['RV']:.4f} p3={omp['AV']:.4f}"
         f" p4={omp['WV']:.3f} p5={omp['RWV']:.4f} p6={omp['AWV']:.4f} /"),
        (f" &POT kp={kp} type=2"
         f" p4={omp['WD']:.3f} p5={omp['RWD']:.4f} p6={omp['AWD']:.4f} /"),
    ]
    if omp["VSO"] != 0.0:
        lines.append(
            f" &POT kp={kp} type=3"
            f" p1={omp['VSO']:.3f} p2={omp['RSO']:.4f} p3={omp['ASO']:.4f} /"
        )
    return lines


def _ws_threshold(l, xfer_a, core_a, r0=1.25):
    """Minimum WS well depth (MeV, positive) needed for the first l-wave bound state.

    Uses the infinite square well (ISW) formula:
        V_threshold = (kR_n)^2 * hbar^2 / (2 * mu * R^2)
    where kR_n is the n-th zero of the spherical Bessel function j_{l-1}
    (for l=0, kR = pi; for l=1, kR = 4.493).
    mu is the reduced mass of the (transferred + core) system in amu.
    """
    # zeros of j_l(x) for l=0,1 (first bound-state crossing for ISW)
    j_zeros = {0: 3.14159, 1: 4.493}
    kR_threshold = j_zeros.get(l, 4.493)

    mu_amu = xfer_a * core_a / (xfer_a + core_a)      # reduced mass (treat all nucleons as 1 amu)
    A_composite = xfer_a + core_a
    R = r0 * A_composite ** (1.0 / 3.0)               # WS radius in fm
    hbar2_over_2mu = 20.91 / mu_amu                    # MeV fm^2  (20.91 = hbar^2 c^2 / (2 * m_u * c^2))
    return kR_threshold ** 2 * hbar2_over_2mu / R ** 2


def _pot_lines_bound(kp, ovl, xfer_a):
    """Return &POT lines for one bound-state potential.

    Coulomb is included only when both core and transferred particle are charged.
    The real WS depth is an initial guess; isc=1 in the &OVERLAP block tells
    FRESCO to vary it to reproduce the specified binding energy BE.

    EIGCC's node-counting iteration (bisection + Newton–Raphson) requires that
    the initial potential V0_init already supports EXACTLY ONE bound state in the
    target l-channel.  If V0_init is too shallow (no bound state) or too deep
    (multiple bound states), FRESCO oscillates and hits the iteration limit.

    The safe window is  V_threshold_1st < |V0_init| < V_threshold_2nd.
    We choose |V0_init| = 1.5 * V_threshold_1st, which sits mid-window for
    any of the light-nucleus overlaps used here.
    """
    # FRESCO convention: type=1 potential V(r) = -p1 * WS(r).
    # p1 must be POSITIVE; FRESCO applies the minus sign internally.
    # Choose p1 = 1.5 * ISW_threshold so exactly one bound state exists,
    # placing the isc=1 starting point safely inside [threshold_1, threshold_2).
    thresh = _ws_threshold(ovl["l"], xfer_a, ovl["core_a"])
    v0_init = 1.5 * thresh  # positive (FRESCO negates to make it attractive)

    lines = []
    if ovl["core_z"] > 0 and ovl["xfer_z"] > 0:
        rc = coul_r(xfer_a, ovl["core_a"])
        lines.append(
            f" &POT kp={kp} type=0"
            f" p1={ovl['core_a']} p2={ovl['core_z']} p3={rc:.4f} /"
        )
    lines.append(f" &POT kp={kp} type=1 p1={v0_init:.1f} p2=1.25 p3=0.65 /")
    return lines


def generate_input(rxn_label, energy_mev):
    """Return a complete FRESCO NAMELIST input string for the given reaction and energy."""
    r = REACTIONS[rxn_label]
    ep = r["entrance"]["proj"]
    et = r["entrance"]["targ"]
    xp = r["exit"]["proj"]
    xt = r["exit"]["targ"]
    xfer_a = r["xfer"]["a"]
    qval = r["qval"]

    omp1 = get_omp(ep["key"], et["z"], et["a"], energy_mev)
    omp2 = get_omp(xp["key"], xt["z"], xt["a"], energy_mev)

    jtmax = max(20, min(200, int(energy_mev * 0.4)))

    lines = []
    lines.append(f"{r['description']} at E_lab={energy_mev:.1f} MeV")
    lines.append("NAMELIST")

    # &FRESCO: rnl and hnl are required for KIND=7 finite-range transfer integrals
    lines.append(
        f" &FRESCO hcm=0.02 rmatch=30.0 rnl=6.0 hnl=0.10"
        f" jtmin=0.0 jtmax={jtmax:.1f} absend=-10.0"
        f" thmin=0.0 thmax=180.0 thinc=2.0 smats=2 xstabl=1 elab={energy_mev} /"
    )
    lines.append("")

    # Partition 1 (entrance)
    lines.append(
        f" &PARTITION namep='{ep['name']}' massp={ep['a']} zp={ep['z']}"
        f" namet='{et['name']}' masst={et['a']} zt={et['z']} qval=0.0 nex=1 /"
    )
    lines.append(
        f" &STATES jp={ep['spin']:.1f} bandp=1 ep=0.0"
        f" jt={et['spin']:.1f} bandt=1 et=0.0 cpot=1 /"
    )
    lines.append("")

    # Partition 2 (exit)
    lines.append(
        f" &PARTITION namep='{xp['name']}' massp={xp['a']} zp={xp['z']}"
        f" namet='{xt['name']}' masst={xt['a']} zt={xt['z']}"
        f" qval={qval:.4f} nex=1 /"
    )
    lines.append(
        f" &STATES jp={xp['spin']:.1f} bandp=1 ep=0.0"
        f" jt={xt['spin']:.1f} bandt=1 et=0.0 cpot=2 /"
    )
    lines.append("")
    lines.append(" &PARTITION /")
    lines.append("")

    # kp=1 entrance OMP
    lines += _pot_lines_omp(1, omp1, ep["a"], et["a"], et["z"])
    lines.append("")

    # kp=2 exit OMP
    lines += _pot_lines_omp(2, omp2, xp["a"], xt["a"], xt["z"])
    lines.append("")

    # kp=3 and kp=4 bound-state potentials (one per overlap)
    for ovl in r["overlaps"]:
        lines += _pot_lines_bound(ovl["kbpot"], ovl, xfer_a)
        lines.append("")

    lines.append(" &POT /")
    lines.append("")

    # Overlaps (isc=1 -> FRESCO varies the WS depth to match the binding energy BE)
    for ovl in r["overlaps"]:
        lines.append(
            f" &OVERLAP kn1={ovl['kn']} kn2={ovl['kn']} ic1=1 ic2=2 in={ovl['in_']}"
            f" kind=0 nn={ovl['nn']} l={ovl['l']} sn={ovl['sn']:.1f} j={ovl['j']:.1f}"
            f" ib=1 kbpot={ovl['kbpot']} be={ovl['be']:.4f} isc=1 /"
            f"  ! {ovl['comment']}"
        )
    lines.append(" &OVERLAP /")
    lines.append("")

    # Coupling: finite-range transfer, POST interaction (ip1=0)
    lines.append(" &COUPLING icto=2 icfrom=1 kind=7 ip1=0 ip2=0 ip3=0 /")
    for cfp in r["cfps"]:
        lines.append(
            f" &CFP in={cfp['in_']} ib={cfp['ib']} ia={cfp['ia']}"
            f" kn={cfp['kn']} a={cfp['a']:.4f} /"
        )
    lines.append(" &CFP /")
    lines.append("")
    lines.append(" &COUPLING /")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate FRESCO NAMELIST inputs for Famiano 2002 Group I two-body reactions."
    )
    parser.add_argument(
        "--reaction",
        choices=list(REACTIONS.keys()),
        help="Reaction label.",
    )
    parser.add_argument(
        "--energy",
        type=float,
        help="Lab-frame energy in MeV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .in file path.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available reactions and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available two-body Group I reactions:")
        for label, rxn in REACTIONS.items():
            q = rxn["qval"]
            erange = rxn["energy_range"]
            print(f"  {label}")
            print(f"    {rxn['description']}")
            print(f"    Q = {q:.3f} MeV")
            print(f"    Default energy range: {erange[0]}-{erange[1]} MeV, step {erange[2]} MeV")
            print()
        return

    if not args.reaction or args.energy is None or not args.output:
        parser.error("--reaction, --energy, and --output are required (or use --list)")

    text = generate_input(args.reaction, args.energy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
