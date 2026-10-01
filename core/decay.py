"""
decay.py

Radioactive decay of unstable species in the cloud abundance vector.

This is applied as an operator-split step after the reaction-network Euler
update each timestep: cascade.py advances Y via nuclear reactions, then
apply_decay() analytically decays the result over the same dt. None of the
tracked daughters are themselves radioactive, so one closed-form pass per
parent species is exact -- no sub-stepping or matrix exponential is needed.

Explicit Euler (dY/dt = -lambda*Y) is deliberately NOT used: several tracked
species (8Li, 8B, 16N) have half-lives of order 1-10 s while the solver's dt
can range up to dt_max_s (~1e4 s per config/run.json), so lambda*dt >> 1 and
forward Euler would be wildly unstable. The analytic solution
Y(t+dt) = Y(t) * exp(-lambda*dt) is stable for any dt.

Half-lives are from standard nuclear data (NNDC/NUBASE). VERIFY against a
current table before trusting results at the percent level, particularly
10Be (long-lived, literature values vary between ~1.39-1.51 Myr).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

_YEAR_S = 3.15576e7  # Julian year, seconds
_MIN_S = 60.0
_DAY_S = 86400.0


@dataclass(frozen=True)
class DecayChannel:
    half_life_s: float
    daughters: Dict[str, int]  # species -> multiplicity; must conserve mass number A
    mode: str = ""


# Species -> DecayChannel. Species absent from this table (or from
# config/species.json) are treated as stable. Daughter multiplicities are
# chosen so that sum(multiplicity * A_daughter) == A_parent for every entry,
# which keeps sum(A_i * Y_i) invariant under apply_decay().
DECAY_TABLE: Dict[str, DecayChannel] = {
    "n":    DecayChannel(611.0,            {"p": 1},   "beta-"),
    "t":    DecayChannel(12.32 * _YEAR_S,   {"3He": 1}, "beta-"),
    "7Be":  DecayChannel(53.22 * _DAY_S,    {"7Li": 1}, "EC"),
    "8Li":  DecayChannel(0.838,             {"4He": 2}, "beta- (via unbound 8Be* -> 2a)"),
    "8B":   DecayChannel(0.770,             {"4He": 2}, "beta+ (via unbound 8Be* -> 2a)"),
    "10Be": DecayChannel(1.51e6 * _YEAR_S,  {"10B": 1}, "beta-"),
    "11C":  DecayChannel(20.334 * _MIN_S,   {"11B": 1}, "beta+/EC"),
    "13N":  DecayChannel(9.965 * _MIN_S,    {"13C": 1}, "beta+"),
    "14C":  DecayChannel(5730.0 * _YEAR_S,  {"14N": 1}, "beta-"),
    "14O":  DecayChannel(70.62,             {"14N": 1}, "beta+"),
    "15O":  DecayChannel(122.24,            {"15N": 1}, "beta+"),
    "16N":  DecayChannel(7.13,              {"16O": 1}, "beta-"),
    "17F":  DecayChannel(64.49,             {"17O": 1}, "beta+"),
    "18F":  DecayChannel(109.771 * _MIN_S,  {"18O": 1}, "beta+/EC"),
}


def decay_constant(species: str) -> float:
    """lambda = ln2 / half_life for a tracked species, else 0.0 (stable)."""
    channel = DECAY_TABLE.get(species)
    if channel is None:
        return 0.0
    return float(np.log(2.0) / channel.half_life_s)


def decay_dydt(Y: np.ndarray, species: List[str]) -> np.ndarray:
    """
    Linear decay RHS dY/dt = -lambda*Y_parent (+ mult*lambda*Y_parent to
    each daughter), aligned with `species`.

    For folding decay into an implicit solve alongside other rates (see
    scripts/run_famiano.py's thermonuclear step) rather than operator-
    splitting it via apply_decay(). A species with half-life << dt that is
    also being produced within the step (e.g. 8B from 7Be(p,g)) needs this:
    split "produce for dt, then decay for dt" undershoots its production/
    decay equilibrium Y = P*tau once dt >> tau, whereas backward Euler on
    the combined RHS converges to it for any dt.
    """
    Y = np.asarray(Y, dtype=float)
    dYdt = np.zeros_like(Y)
    index = {sp: i for i, sp in enumerate(species)}

    for parent, channel in DECAY_TABLE.items():
        p_idx = index.get(parent)
        if p_idx is None:
            continue
        rate = (np.log(2.0) / channel.half_life_s) * Y[p_idx]
        dYdt[p_idx] -= rate
        for daughter, mult in channel.daughters.items():
            d_idx = index.get(daughter)
            if d_idx is not None:
                dYdt[d_idx] += mult * rate

    return dYdt


def apply_decay(Y: np.ndarray, species: List[str], dt_s: float) -> np.ndarray:
    """
    Advance abundances Y by one exact radioactive-decay step of length dt_s.

    Species not present in `species` are silently skipped as decay parents
    (nothing to decay) and as daughters (decayed material falls out of the
    tracked budget -- add the daughter to config/species.json instead of
    relying on this).

    Returns a new array; does not mutate Y in place.
    """
    if dt_s <= 0.0:
        return np.asarray(Y, dtype=float)

    Y_new = np.asarray(Y, dtype=float).copy()
    index = {sp: i for i, sp in enumerate(species)}

    for parent, channel in DECAY_TABLE.items():
        p_idx = index.get(parent)
        if p_idx is None:
            continue

        Y_parent = Y_new[p_idx]
        if Y_parent <= 0.0:
            continue

        lam = np.log(2.0) / channel.half_life_s
        survive_frac = np.exp(-lam * dt_s)
        n_decayed = Y_parent * (1.0 - survive_frac)

        Y_new[p_idx] = Y_parent * survive_frac
        for daughter, mult in channel.daughters.items():
            d_idx = index.get(daughter)
            if d_idx is not None:
                Y_new[d_idx] += mult * n_decayed

    return Y_new
