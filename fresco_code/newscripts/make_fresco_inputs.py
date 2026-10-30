#!/usr/bin/env python3
"""
make_fresco_inputs.py
=====================

Generate FRESCO input decks reconstructing the *two-body* Group I reactions of
Famiano, Boyd & Kajino (2002, ApJ 576, 89), scanning the projectile LAB energy
from 5 to 400 MeV in 5 MeV steps.

WHAT THIS DOES (and does not) cover
-----------------------------------
Only the genuinely two-body Group I reactions can be done in DWBA/CRC. The many
three-body breakup channels in Table 2 [(p,pn), (p,2p), (p,2n), deuteron
breakup, ...] are NOT produced by a DWBA code; the paper itself handled those
with a crude phase-space-shape assumption normalised to data. Reconstruct those
separately. The four reactions below are the DWBA-amenable ones:

  R1  p + 4He -> d + 3He        (neutron pickup)        -- fully specified
  R2  p + 7Li -> 4He + 4He      (triton pickup; 7Li(p,a)a) -- CLUSTER, see note
  R3  4He + 4He -> p + 7Li      (triton stripping)      -- CLUSTER, see note
  R4  4He + 4He -> n + 7Be      (3He stripping)         -- CLUSTER, see note

R1 is a textbook single-nucleon transfer and is specified completely. R2-R4 are
cluster transfers that are also resonance-dominated (they proceed largely through
the 8Be compound system); the alpha-cluster overlap quantum numbers below are
reasonable cluster-model choices but YOU SHOULD VERIFY them against a cluster
reference. Per the paper's philosophy this matters less than usual: FRESCO is
used only for the SHAPE of dsigma/dOmega; the magnitude is renormalised to the
measured total cross section sigma_tot(E) afterwards.

OPTICAL POTENTIALS
------------------
The Woods-Saxon parameters in OMP below are sensible placeholders, NOT the exact
published sets. Replace them with the paper's references before production runs:
  protons   : Menet et al. (1971) global fit; Perey & Perey (1976); Schwandt+ (1982)
  alphas    : Perey & Perey (1976)
  deuterons : Perey & Perey (1976) volume/surface + Lohr & Haeberli (1974) spin-orbit
Validate by first reproducing p+4He elastic vs Rogers et al. (1969), as the paper did.

Run:  python3 make_fresco_inputs.py
Then: fresco < R1_p_4He_to_d_3He.in > R1.out   (etc.)
"""

# ---------------------------------------------------------------------------
# Nuclear data: mass excess (MeV) and ground-state spin/parity
# ---------------------------------------------------------------------------
MASS_EXCESS = {  # AME-ish values, MeV
    "n":   8.0713, "1H":  7.2890, "2H": 13.1357, "3H": 14.9498,
    "3He": 14.9312, "4He": 2.4249, "7Li": 14.9071, "7Be": 15.7690,
}
A = {"n": 1, "1H": 1, "2H": 2, "3H": 3, "3He": 3, "4He": 4, "7Li": 7, "7Be": 7}
Z = {"n": 0, "1H": 1, "2H": 1, "3H": 1, "3He": 2, "4He": 2, "7Li": 3, "7Be": 4}
# (spin, parity) ; parity +1/-1
JPI = {"n": (0.5, 1), "1H": (0.5, 1), "2H": (1.0, 1), "3H": (0.5, 1),
       "3He": (0.5, 1), "4He": (0.0, 1), "7Li": (1.5, -1), "7Be": (1.5, -1)}

ESTEP = 5.0          # MeV
EMAX  = 400.0        # MeV
EGRID_LOW = 5.0      # MeV (low end of requested scan)


def qvalue(a, A_t, b, B):
    return (MASS_EXCESS[a] + MASS_EXCESS[A_t]) - (MASS_EXCESS[b] + MASS_EXCESS[B])


def lab_threshold(a, A_t, Q):
    """Non-relativistic lab threshold for endothermic reaction (projectile a on
    target A_t at rest). Returns 0 for exothermic reactions."""
    if Q >= 0:
        return 0.0
    return -Q * (A[a] + A[A_t]) / A[A_t]


def energy_range(estart):
    """elab(1), elab(2), nlab(1) for a 5-MeV scan from estart to EMAX.
    nlab = number of 5-MeV steps; this build assumes that yields nlab+1
    energies including both endpoints. If your FRESCO counts ENERGIES instead
    of STEPS, add 1 to nlab."""
    # round start up to the next 5-MeV grid point
    e1 = EGRID_LOW
    nsteps = int(round((EMAX - e1) / ESTEP))
    return e1, EMAX, nsteps


# ---------------------------------------------------------------------------
# Optical-model potentials  (EDIT THESE -- placeholders, see header note)
# key -> dict of WS terms.  rc is the reduced Coulomb radius.
# vol  = (V, rV, aV, W, rW, aW)        type=1 volume
# surf = (0, 0, 0, Wd, rD, aD)         type=2 surface (imag only here)
# so   = (Vso, rso, aso)               type=3 spin-orbit (projectile)
# ---------------------------------------------------------------------------
OMP = {
    "p+4He":  dict(rc=1.30, vol=(48.0, 1.15, 0.57, 0.0, 1.15, 0.57),
                   surf=(0, 0, 0, 6.0, 1.30, 0.50), so=(5.5, 1.00, 0.50)),
    "d+3He":  dict(rc=1.30, vol=(88.0, 1.17, 0.79, 0.0, 1.17, 0.79),
                   surf=(0, 0, 0, 12.0, 1.33, 0.74), so=(7.0, 1.07, 0.66)),
    "p+7Li":  dict(rc=1.30, vol=(50.0, 1.20, 0.60, 0.0, 1.20, 0.60),
                   surf=(0, 0, 0, 8.0, 1.30, 0.50), so=(5.5, 1.00, 0.50)),
    "4He+4He":dict(rc=1.40, vol=(120.0, 1.40, 0.55, 15.0, 1.40, 0.55),
                   surf=None, so=None),
    "n+7Be":  dict(rc=0.0,  vol=(48.0, 1.20, 0.65, 0.0, 1.20, 0.65),
                   surf=(0, 0, 0, 6.0, 1.30, 0.50), so=(5.5, 1.00, 0.50)),
}

# Bound-state ("binding") potential geometry used to build the form factors.
# isc=1 makes FRESCO adjust the depth to reproduce the binding energy `be`.
BIND = dict(V=50.0, r=1.25, a=0.65, Vso=6.0, rso=1.25, aso=0.65)


def pot_blocks(kp, label):
    """Return &pot lines for optical channel `label` with index kp."""
    o = OMP[label]
    L = []
    # Coulomb (type=0): p1=ap, p2=at, p3=rc  -- ap,at filled by caller via mass nums
    L.append(f" &POT kp={kp} type=0 shape=0 p(1:3)={{ap}} {{at}} {o['rc']:.3f} /")
    V, rV, aV, W, rW, aW = o["vol"]
    L.append(f" &POT kp={kp} type=1 shape=0 "
             f"p(1:6)={V:.3f} {rV:.3f} {aV:.3f} {W:.3f} {rW:.3f} {aW:.3f} /")
    if o["surf"]:
        _, _, _, Wd, rD, aD = o["surf"]
        L.append(f" &POT kp={kp} type=2 shape=0 "
                 f"p(1:6)=0.0 0.0 0.0 {Wd:.3f} {rD:.3f} {aD:.3f} /")
    if o["so"]:
        Vso, rso, aso = o["so"]
        L.append(f" &POT kp={kp} type=3 shape=0 "
                 f"p(1:3)={Vso:.3f} {rso:.3f} {aso:.3f} /")
    return L


def bind_block(kp):
    b = BIND
    return [
        f" &POT kp={kp} type=1 shape=0 p(1:3)={b['V']:.3f} {b['r']:.3f} {b['a']:.3f} /",
        f" &POT kp={kp} type=3 shape=0 p(1:3)={b['Vso']:.3f} {b['rso']:.3f} {b['aso']:.3f} /",
    ]


# ---------------------------------------------------------------------------
# Reaction definitions
#   each: a (proj), A (targ) entrance ; b (proj), B (targ) exit
#   omp_in / omp_out : keys into OMP
#   overlaps: list of dicts describing the two transfer form factors
#       composite, core, in (1=proj side,2=targ side), ic1, ic2,
#       nn, l, sn, j, be
#   note: free-text caveat
# ---------------------------------------------------------------------------
REACTIONS = [
    dict(tag="R1_p_4He_to_d_3He",
         a="1H", At="4He", b="2H", B="3He",
         omp_in="p+4He", omp_out="d+3He",
         rnl=10.0,
         cluster=False,
         overlaps=[
             # <4He | 3He + n>: composite 4He(target,entrance ic1=1,in=2), core 3He(ic2=2)
             dict(comp="4He", core="3He", in_=2, ic1=1, ic2=2,
                  nn=1, l=0, sn=0.5, j=0.5, be=20.578),
             # <d | p + n>: composite d(proj,exit ic1=2,in=1), core p(ic2=1)
             dict(comp="2H", core="1H", in_=1, ic1=2, ic2=1,
                  nn=1, l=0, sn=0.5, j=0.5, be=2.2246),
         ],
         note="Standard single-neutron pickup. Fully specified."),

    dict(tag="R2_p_7Li_to_4He_4He",
         a="1H", At="7Li", b="4He", B="4He",
         omp_in="p+7Li", omp_out="4He+4He",
         rnl=100.0,
         cluster=True,
         overlaps=[
             # <7Li | 4He + t>  (alpha-triton cluster, L=1 -> 3/2-)
             dict(comp="7Li", core="4He", in_=2, ic1=1, ic2=2,
                  nn=2, l=1, sn=0.5, j=1.5, be=2.467),
             # <4He | p + t>  (ejectile alpha = p + t)
             dict(comp="4He", core="3H", in_=1, ic1=2, ic2=1,
                  nn=1, l=0, sn=0.5, j=0.5, be=19.814),
         ],
         note="Triton transfer; resonance-dominated (8Be). Cluster QNs are "
              "best-guess -- VERIFY. Identical 4He in exit: check symmetry factor."),

    dict(tag="R3_4He_4He_to_p_7Li",
         a="4He", At="4He", b="1H", B="7Li",
         omp_in="4He+4He", omp_out="p+7Li",
         rnl=25.0,
         cluster=True,
         overlaps=[
             # <4He | t + p> (projectile alpha = t + p, p is ejectile)
             dict(comp="4He", core="3H", in_=1, ic1=1, ic2=2,
                  nn=1, l=0, sn=0.5, j=0.5, be=19.814),
             # <7Li | 4He + t> (residual 7Li = alpha core + transferred t)
             dict(comp="7Li", core="4He", in_=2, ic1=2, ic2=1,
                  nn=2, l=1, sn=0.5, j=1.5, be=2.467),
         ],
         note="Triton stripping (inverse of R2). Cluster QNs best-guess -- VERIFY."),

    dict(tag="R4_4He_4He_to_n_7Be",
         a="4He", At="4He", b="n", B="7Be",
         omp_in="4He+4He", omp_out="n+7Be",
         rnl=30.0,
         cluster=True,
         overlaps=[
             # <4He | 3He + n> (projectile alpha = 3He + n, 3He transferred? )
             # Here the transferred cluster is 3He; ejectile is the neutron.
             dict(comp="4He", core="n", in_=1, ic1=1, ic2=2,
                  nn=1, l=0, sn=0.5, j=0.5, be=20.578),
             # <7Be | 4He + 3He>
             dict(comp="7Be", core="4He", in_=2, ic1=2, ic2=1,
                  nn=2, l=1, sn=0.5, j=1.5, be=1.587),
         ],
         note="3He stripping. Cluster QNs best-guess -- VERIFY. (Transferred "
              "cluster is 3He; this scaffold treats it schematically.)"),
]


def states_line(part, body, cpot=None, copy=None):
    """Build &states fields for one body. part: 'p' or 't'."""
    j, pi = JPI[body]
    if part == "p":
        s = f"jp={j} ptyp={pi} ep=0.0"
        if cpot is not None:
            s += f" cpot={cpot}"
    else:
        s = f"jt={j} ptyt={pi} et=0.0"
    return s


def build(rx):
    a, At, b, B = rx["a"], rx["At"], rx["b"], rx["B"]
    Q = qvalue(a, At, b, B)
    Eth = lab_threshold(a, At, Q)
    e1, e2, nlab = energy_range(Eth if Eth > 0 else EGRID_LOW)

    head = (f"{At}({a},{b}){B}  Q={Q:+.3f} MeV  Eth_lab={Eth:.2f} MeV  "
            f"[{rx['tag']}]")[:80]

    out = []
    out.append(head)
    out.append("NAMELIST")
    # ---- &fresco ----
    out.append(" &FRESCO")
    out.append(f"   hcm=0.05 rmatch=30.0 rintp=0.25 hnl=0.10 rnl={rx['rnl']} centre=0.0")
    out.append("   jtmin=0.0 jtmax=80 absend=-1.0")
    out.append("   thmin=0.0 thmax=180.0 thinc=1.0        ! 1-deg CM grid, as DWUCK in the paper")
    out.append("   it0=1 iter=1                            ! one-step DWBA")
    out.append("   rela='a'                                ! relativistic kinematics (remove if build rejects)")
    out.append("   chans=1 smats=2 xstabl=1")
    out.append("   pel=1 exl=1 lab=1 lin=1")
    out.append(f"   elab(1)={e1:.1f} elab(2)={e2:.1f} nlab(1)={nlab}   ! 5-MeV steps; see header note on nlab")
    out.append("  /")
    # ---- partitions ----
    # entrance: projectile a + target At  (partition 1)
    out.append(f" &PARTITION namep='{a}' massp={A[a]} zp={Z[a]} "
               f"namet='{At}' masst={A[At]} zt={Z[At]} qval=0.000 nex=1 /")
    out.append(f"   &STATES {states_line('p', a, cpot=1)} {states_line('t', At)} /")
    # exit: projectile b + target B  (partition 2)
    out.append(f" &PARTITION namep='{b}' massp={A[b]} zp={Z[b]} "
               f"namet='{B}' masst={A[B]} zt={Z[B]} qval={Q:.3f} nex=1 /")
    out.append(f"   &STATES {states_line('p', b, cpot=2)} {states_line('t', B)} /")
    out.append(" &partition /")
    # ---- potentials ----
    # kp=1 entrance optical, kp=2 exit optical, kp=3/4 binding pots
    for line in pot_blocks(1, rx["omp_in"]):
        out.append(line.format(ap=A[a], at=A[At]))
    for line in pot_blocks(2, rx["omp_out"]):
        out.append(line.format(ap=A[b], at=A[B]))
    out += bind_block(3)   # form factor 1
    out += bind_block(4)   # form factor 2
    out.append(" &pot /")
    # ---- overlaps ----
    for kn, ov in enumerate(rx["overlaps"], start=1):
        kbpot = 3 if kn == 1 else 4
        out.append(f" &OVERLAP kn1={kn} ic1={ov['ic1']} ic2={ov['ic2']} in={ov['in_']} "
                   f"kind=0 nn={ov['nn']} l={ov['l']} sn={ov['sn']} j={ov['j']} "
                   f"kbpot={kbpot} be={ov['be']:.4f} isc=1 ipc=0 /")
    out.append(" &overlap /")
    # ---- coupling: finite-range transfer (kind=7), entrance->exit ----
    out.append(" &COUPLING icto=2 icfrom=1 kind=7 ip1=0 ip2=-1 ip3=5 /")
    out.append("   &CFP in=1 ib=1 ia=1 kn=1 a=1.000 /")
    out.append("   &CFP in=2 ib=1 ia=1 kn=2 a=1.000 /")
    out.append(" &coupling /")
    out.append("")
    meta = dict(Q=Q, Eth=Eth, e1=e1, e2=e2, nlab=nlab,
                npts=nlab + 1, cluster=rx["cluster"], note=rx["note"])
    return "\n".join(out), meta


if __name__ == "__main__":
    from pathlib import Path

    # Place outputs under <project_root>/runs/newruns/<tag>/
    project_root = Path(__file__).resolve().parents[2]
    newruns_dir = project_root / "runs" / "newruns"

    print(f"{'reaction':<26}{'Q(MeV)':>9}{'Eth_lab':>9}"
          f"{'scan(MeV)':>16}{'pts':>5}")
    print("-" * 70)
    for rx in REACTIONS:
        text, m = build(rx)
        out_dir = newruns_dir / rx["tag"]
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / (rx["tag"] + ".in")
        fpath.write_text(text)
        scan = f"{m['e1']:.0f}..{m['e2']:.0f}/5"
        print(f"{rx['tag']:<26}{m['Q']:>+9.3f}{m['Eth']:>9.2f}{scan:>16}{m['npts']:>5}")
    print("-" * 70)
    print(f"Wrote 4 .in files under {newruns_dir}")
    print("R1 is fully specified; R2-R4 carry CLUSTER")
    print("placeholders flagged in each header -- verify the overlap QNs.")
