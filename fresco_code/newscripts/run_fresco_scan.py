#!/usr/bin/env python3
"""
run_fresco_scan.py
==================

Generate ONE single-energy FRESCO input deck per lab energy (5..400 MeV, 5 MeV
steps) for each two-body Group I reaction, run FRESCO on each, and write the
output to the directory layout that energy_bins_reaction_dir.py expects:

    <reaction_key>/
        inputs/   <reaction_key>_<E>MeV.in
        outputs/  <reaction_key>_<E>MeV.out      <-- post-processor reads these
        E_bins/   (created later by the post-processor)

The reaction_key names match the keys in energy_bins_reaction_dir.py, so after
running this you do, e.g.:

    python3 energy_bins_reaction_dir.py --reaction-dir he4_p_d_he3 \
            --reaction he4_p_d_he3 --bin-width 5 --max-energy 400

WHY ONE FILE PER ENERGY: the post-processor keeps only one lab energy per .out
file. A single multi-energy FRESCO run would blend all energies together.

WHY THESE NAMES: particle names written into the deck ('p','d','3He','4He',
'7Li','7Be','n') must equal the FRESCO_NAMES *values* in the post-processor, or
its outgoing-channel header match fails.

Usage
-----
    python3 run_fresco_scan.py --fresco /path/to/fresco          # generate + run
    python3 run_fresco_scan.py --gen-only                        # just write .in
    python3 run_fresco_scan.py --reactions he4_p_d_he3 --fresco fresco

Caveats unchanged from before: R1 is a complete single-neutron pickup; R2-R4
are CLUSTER scaffolds (verify the overlap QNs). Optical-model parameters are
placeholders -- swap in Menet(1971)/Perey&Perey(1976)/Schwandt(1982)/Lohr&
Haeberli(1974). FRESCO is used for the SHAPE; renormalise to measured sigma_tot.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Species: A, Z, mass excess (MeV), (spin, parity)
# Names are the strings FRESCO will echo AND what the post-processor matches.
# ---------------------------------------------------------------------------
SP = {
    "n":   dict(A=1, Z=0, dm=8.0713,  jpi=(0.5,  1)),
    "p":   dict(A=1, Z=1, dm=7.2890,  jpi=(0.5,  1)),
    "d":   dict(A=2, Z=1, dm=13.1357, jpi=(1.0,  1)),
    "t":   dict(A=3, Z=1, dm=14.9498, jpi=(0.5,  1)),
    "3He": dict(A=3, Z=2, dm=14.9312, jpi=(0.5,  1)),
    "4He": dict(A=4, Z=2, dm=2.4249,  jpi=(0.0,  1)),
    "7Li": dict(A=7, Z=3, dm=14.9071, jpi=(1.5, -1)),
    "7Be": dict(A=7, Z=4, dm=15.7690, jpi=(1.5, -1)),
}

ESTEP, EMAX, ELOW = 5.0, 400.0, 5.0

# Optical potentials keyed by "proj+targ" (placeholders -- EDIT).
#   vol  = (V, rV, aV, W, rW, aW)   type=1
#   surf = (Wd, rD, aD) or None     type=2 (imag surface)
#   so   = (Vso, rso, aso) or None  type=3
OMP = {
    "p+4He":   dict(rc=1.30, vol=(48.0,1.15,0.57, 0.0,1.15,0.57), surf=(6.0,1.30,0.50), so=(5.5,1.00,0.50)),
    "d+3He":   dict(rc=1.30, vol=(88.0,1.17,0.79, 0.0,1.17,0.79), surf=(12.0,1.33,0.74),so=(7.0,1.07,0.66)),
    "p+7Li":   dict(rc=1.30, vol=(50.0,1.20,0.60, 0.0,1.20,0.60), surf=(8.0,1.30,0.50), so=(5.5,1.00,0.50)),
    "4He+4He": dict(rc=1.40, vol=(120.,1.40,0.55, 15.,1.40,0.55), surf=None,            so=None),
    "n+7Be":   dict(rc=0.0,  vol=(48.0,1.20,0.65, 0.0,1.20,0.65), surf=(6.0,1.30,0.50), so=(5.5,1.00,0.50)),
}
BIND = dict(V=50.0, r=1.25, a=0.65, Vso=6.0, rso=1.25, aso=0.65)

# Reactions, keyed to match the post-processor. proj carries E_lab.
# overlaps: (composite, core, in[1=proj/2=targ], ic1, ic2, nn, l, sn, j, be)
REACTIONS = {
    "he4_p_d_he3": dict(
        proj="p", targ="4He", ejec="d", resid="3He",
        omp_in="p+4He", omp_out="d+3He", cluster=False, rnl=10.0,
        overlaps=[("4He","3He",2,1,2, 1,0,0.5,0.5, 20.578),
                  ("d","p",   1,2,1, 1,0,0.5,0.5,  2.2246)],
        note="single-neutron pickup -- fully specified"),
    "li7_p_he4_he4": dict(
        proj="p", targ="7Li", ejec="4He", resid="4He",
        omp_in="p+7Li", omp_out="4He+4He", cluster=True, rnl=100.0,
        overlaps=[("4He","p", 2,1,2, 1,0,0.5,0.5, 19.814),
                  ("7Li","4He",1,2,1, 2,1,0.5,1.5,  2.467)],
        note="triton pickup; ejectile-side overlap first, target-side second"),
    "he4_he4_p_li7": dict(
        proj="4He", targ="4He", ejec="p", resid="7Li",
        omp_in="4He+4He", omp_out="p+7Li", cluster=True, rnl=25.0, cutl=3.0,
        overlaps=[("4He","t",1,1,2, 2,1,0.5,1.5, 2.467),
                  ("7Li","4He",2,2,1, 1,0,0.5,0.5, 19.814)],
        note="triton stripping; CLUSTER QNs"),
    "he4_he4_n_be7": dict(
        proj="4He", targ="4He", ejec="n", resid="7Be",
        omp_in="4He+4He", omp_out="n+7Be", cluster=True, rnl=30.0, cutl=3.0,
        overlaps=[("4He","n",1,1,2, 1,0,0.5,0.5, 20.578),
                  ("7Be","4He",2,2,1, 2,1,0.5,1.5, 1.587)],
        note="3He stripping; CLUSTER QNs"),
}


def qvalue(r):
    mi = SP[r["proj"]]["dm"] + SP[r["targ"]]["dm"]
    mf = SP[r["ejec"]]["dm"] + SP[r["resid"]]["dm"]
    return mi - mf


def lab_threshold(r, Q):
    if Q >= 0:
        return 0.0
    return -Q * (SP[r["proj"]]["A"] + SP[r["targ"]]["A"]) / SP[r["targ"]]["A"]


def energy_list(r):
    Q = qvalue(r)
    Eth = lab_threshold(r, Q)
    start = ELOW
    n = int(round((EMAX - start) / ESTEP))
    return [start + i * ESTEP for i in range(n + 1)], Q, Eth


def fmt_e(E):
    return f"{int(round(E))}" if abs(E - round(E)) < 1e-9 else f"{E:g}".replace(".", "p")


def pot_lines(kp, label, ap, at):
    o = OMP[label]
    L = [f" &POT kp={kp} type=0 shape=0 p(1:3)={ap} {at} {o['rc']:.3f} /"]
    V, rV, aV, W, rW, aW = o["vol"]
    L.append(f" &POT kp={kp} type=1 shape=0 p(1:6)={V:.3f} {rV:.3f} {aV:.3f} {W:.3f} {rW:.3f} {aW:.3f} /")
    if o["surf"]:
        Wd, rD, aD = o["surf"]
        L.append(f" &POT kp={kp} type=2 shape=0 p(1:6)=0.0 0.0 0.0 {Wd:.3f} {rD:.3f} {aD:.3f} /")
    if o["so"]:
        Vso, rso, aso = o["so"]
        L.append(f" &POT kp={kp} type=3 shape=0 p(1:3)={Vso:.3f} {rso:.3f} {aso:.3f} /")
    return L


def bind_lines(kp):
    b = BIND
    return [f" &POT kp={kp} type=1 shape=0 p(1:3)={b['V']:.3f} {b['r']:.3f} {b['a']:.3f} /",
            f" &POT kp={kp} type=3 shape=0 p(1:3)={b['Vso']:.3f} {b['rso']:.3f} {b['aso']:.3f} /"]


def build_deck(key, r, E, Q):
    a, At, b, B = r["proj"], r["targ"], r["ejec"], r["resid"]
    ja, pa = SP[a]["jpi"]; jA, pA = SP[At]["jpi"]
    jb, pb = SP[b]["jpi"]; jB, pB = SP[B]["jpi"]
    L = []
    L.append(f"{At}({a},{b}){B}  E={E:g} MeV  Q={Q:+.3f}  [{key}]"[:80])
    L.append("NAMELIST")
    L.append(" &FRESCO")
    cutl = f" cutl={r['cutl']}" if r.get('cutl') else ""
    L.append(f"   hcm=0.05 rmatch=30.0 rintp=0.25 hnl=0.10 rnl={r['rnl']}{cutl}")
    L.append("   jtmin=0.0 jtmax=80 absend=-1.0")
    L.append("   thmin=0.0 thmax=180.0 thinc=1.0")
    L.append("   it0=1 iter=1")
    L.append("   rela='a'")
    L.append("   chans=1 smats=2 xstabl=1")
    L.append("   pel=1 exl=1 lab=1 lin=1")
    L.append(f"   elab={E:.1f}")            # single energy -> one .out per energy
    L.append("  /")
    L.append(f" &PARTITION namep='{a}' massp={SP[a]['A']} zp={SP[a]['Z']} "
             f"namet='{At}' masst={SP[At]['A']} zt={SP[At]['Z']} qval=0.000 nex=1 /")
    L.append(f"   &STATES jp={ja} ptyp={pa} ep=0.0 cpot=1 jt={jA} ptyt={pA} et=0.0 /")
    L.append(f" &PARTITION namep='{b}' massp={SP[b]['A']} zp={SP[b]['Z']} "
             f"namet='{B}' masst={SP[B]['A']} zt={SP[B]['Z']} qval={Q:.3f} nex=1 /")
    L.append(f"   &STATES jp={jb} ptyp={pb} ep=0.0 cpot=2 jt={jB} ptyt={pB} et=0.0 /")
    L.append(" &partition /")
    L += pot_lines(1, r["omp_in"],  SP[a]["A"], SP[At]["A"])
    L += pot_lines(2, r["omp_out"], SP[b]["A"], SP[B]["A"])
    L += bind_lines(3)
    L += bind_lines(4)
    L.append(" &pot /")
    overlap_ins = []
    for kn, (comp, core, in_, ic1, ic2, nn, l, sn, j, be) in enumerate(r["overlaps"], 1):
        kbpot = 3 if kn == 1 else 4
        L.append(f" &OVERLAP kn1={kn} ic1={ic1} ic2={ic2} in={in_} kind=0 "
                 f"nn={nn} l={l} sn={sn} j={j} kbpot={kbpot} be={be:.4f} isc=1 ipc=0 /")
        overlap_ins.append(in_)
    L.append(" &overlap /")
    L.append(" &COUPLING icto=2 icfrom=1 kind=7 ip1=0 ip2=-1 ip3=5 /")
    kns = [2, 1] if r.get("cluster") else [1, 2]
    for kn_cfp, in_ in zip(kns, overlap_ins):
        L.append(f"   &CFP in={in_} ib=1 ia=1 kn={kn_cfp} a=1.000 /")
    L.append(" &coupling /")
    L.append("")
    return "\n".join(L)


def run_fresco(fresco_exe, deck_text, out_path):
    """Run FRESCO in an isolated scratch dir (it litters fort.* files), capturing
    fort.6/stdout -- which is where the angular-distribution blocks the post-
    processor parses are printed -- to out_path."""
    scratch = tempfile.mkdtemp(prefix="fresco_")
    try:
        with open(out_path, "w") as out:
            subprocess.run([fresco_exe], input=deck_text, stdout=out,
                           stderr=subprocess.STDOUT, text=True, cwd=scratch,
                           check=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresco", default="fresco", help="FRESCO executable")
    ap.add_argument("--gen-only", action="store_true", help="write .in files but do not run")
    ap.add_argument("--reactions", nargs="*", default=list(REACTIONS),
                    choices=list(REACTIONS), help="subset of reaction keys")
    ap.add_argument("--root", default=".", help="root dir to create reaction folders in")
    args = ap.parse_args()

    if not args.gen_only and not shutil.which(args.fresco) and not os.path.exists(args.fresco):
        print(f"[warn] FRESCO executable '{args.fresco}' not found on PATH. "
              f"Use --gen-only to just write decks, or pass --fresco /path/to/fresco.")

    root = Path(args.root)
    for key in args.reactions:
        r = REACTIONS[key]
        energies, Q, Eth = energy_list(r)
        rdir = root / key
        (rdir / "inputs").mkdir(parents=True, exist_ok=True)
        (rdir / "outputs").mkdir(parents=True, exist_ok=True)
        tag = "CLUSTER-scaffold" if r["cluster"] else "complete"
        print(f"\n=== {key}  Q={Q:+.3f} MeV  Eth_lab={Eth:.2f} MeV  "
              f"{len(energies)} energies [{energies[0]:g}..{energies[-1]:g}]  ({tag}) ===")
        for E in energies:
            deck = build_deck(key, r, E, Q)
            in_path  = rdir / "inputs"  / f"{key}_{fmt_e(E)}MeV.in"
            out_path = rdir / "outputs" / f"{key}_{fmt_e(E)}MeV.out"
            in_path.write_text(deck)
            if args.gen_only:
                continue
            try:
                run_fresco(args.fresco, deck, out_path)
                print(f"  [run] {E:6.1f} MeV -> {out_path}")
            except FileNotFoundError:
                print(f"  [ERR] FRESCO not found ('{args.fresco}'). Stopping. "
                      f"Decks were written; rerun with a valid --fresco.")
                return
            except subprocess.CalledProcessError as e:
                print(f"  [ERR] {E:6.1f} MeV: FRESCO exited {e.returncode} (see {out_path})")
        print(f"  decks in {rdir/'inputs'}")
        if not args.gen_only:
            print(f"  next: python3 energy_bins_reaction_dir.py "
                  f"--reaction-dir {key} --reaction {key} --bin-width 5 --max-energy 400")


if __name__ == "__main__":
    main()
