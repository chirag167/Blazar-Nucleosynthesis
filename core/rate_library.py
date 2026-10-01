"""
rate_library.py

Parsers that turn the raw thermonuclear rate-data files in
data/ReactionRates/Thermal/ (and data/mass_table.json / config/species.json
for scope-filtering) into ThermonuclearReaction objects for
core/thermonuclear.py's ThermonuclearLibrary.

Sources implemented
--------------------
NACRE_1 (data/ReactionRates/Thermal/NACRE_1_coeff.csv)
    Angulo et al. 1999 analytic fits:
        ln(N_A<sv>) = a0 + a1/T9 + a2/T9^(1/3) + a3*T9^(1/3)
                         + a4*T9 + a5*T9^(5/3) + a6*ln(T9)
    The 'elements' column packs target/projectile/ejectile(s)/residual with
    a token order that differs by row length (see _NACRE1_TOKEN_ORDER below
    and its docstring for how this was determined -- empirically, by
    checking mass/charge conservation and matching known Q-values, not
    assumed). Multiple rows can share one 'elements' string (repeated
    resonant + non-resonant fit terms for one physical reaction); their
    rates are summed. Rows referencing species outside config/species.json
    (Na, Mg, Al, Si, P, ... -- this file is a general NACRE scrape, not
    curated to Famiano's network) are skipped.

NACRE_2 (data/ReactionRates/Thermal/NACRE_2/*.dat)
    Angulo et al. 1999 tabulated "adopted" rate vs T9 (plus low/high
    uncertainty bounds, unused here). One reaction per file; filenames are
    a compact target+projectile+ejectile+residual string (e.g.
    "6lipa3he.dat" = 6Li(p,a)3He). Decoded via an explicit, hand-verified
    lookup table (_NACRE2_REACTIONS) rather than a general tokenizer --
    with only 34 files, cross-checking each against known Q-values by hand
    is safer than trusting a greedy compact-string parser on an unlabeled
    format.

CF88 (data/ReactionRates/Thermal/CF88/)
    Caughlan & Fowler 1988 tabulated rate vs T9.
        - 3hdn4he.dat: single reaction, same compact-filename convention as
          NACRE_2 (added to the same hand-verified lookup table).
        - web_data.dat: a bundle of ~150 reactions, each introduced by an
          unambiguous "ELEMENT(REACT,EJECT)RESIDUAL" header line (e.g.
          "LI6(P,G)BE7"), parsed by regex. Reactions involving species
          outside config/species.json are skipped -- this file extends
          well past Famiano's Z<=10 (n..20Ne) network, up to P31.

Table 1 specials (data/ReactionRates/Thermal/14Cag18O.csv, 17Ona14C.csv)
    Two of Famiano's Table 1 "not in NACRE/CF88" reactions with an
    individual literature source, tabulated T9 vs rate directly.

Bao (data/ReactionRates/Thermal/Bao_rad_n_capture_rates.csv)
    Bao et al. 2000 radiative neutron-capture rates, tabulated vs. kT
    [keV] rather than T9; converted via kT[keV] = 86.17333 * T9. Rows
    outside config/species.json scope are skipped (this file also extends
    well past Ne, to the iron-peak region).

Deliberately NOT implemented: Wagoner_coeff.csv
    Wagoner (1969) / Malaney & Fowler (1989) special captures, using a
    multi-parameter fit (Q, R, G, g, C_r, D_r, F, rho_c) whose exact
    combination formula could not be confidently reconstructed from the
    data file alone or from a general web search -- guessing it risks a
    silently-wrong rate, which is worse than leaving the gap explicit.
    Famiano's text cites ~9-10 capture targets sourced this way (1H, 6Li,
    8Li, 7Be, 9Be, 10Be, 10B, 11B, 11C, 17O); of these, 7 appear as (n,g)
    rows in the repo's copy of Wagoner_coeff.csv (1H, 6Li, 9Be, 10Be, 10B,
    11B, 17O) and 3 are entirely absent from it (8Li->9Li, 7Be->8Be,
    11C->12C). See MISSING_SPECIAL_CAPTURES below. Needs the primary
    source (Wagoner 1969, ApJS 18, 247) before it can be safely coded.

Unit handling
-------------
NACRE, CF88, and Bao all publish the MOLAR rate N_A<sigma v> in
cm^3 mol^-1 s^-1. ThermonuclearReaction.rate_func's contract (see
core/thermonuclear.py) requires the PER-PAIR rate <sigma v> in cm^3 s^-1,
so every loader here divides by AVOGADRO at ingestion -- the same
"convert at the boundary" convention cascade.py uses for cross sections
(mb -> cm^2 in _reaction_sigma_matrix).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from thermonuclear import ThermonuclearLibrary, ThermonuclearReaction

Array = np.ndarray

AVOGADRO = 6.02214076e23
KEV_PER_T9 = 86.17333  # kT[keV] = KEV_PER_T9 * T9  (k_B = 8.617333e-5 eV/K)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_THERMAL_DIR = _REPO_ROOT / "data" / "ReactionRates" / "Thermal"
_DEFAULT_SPECIES_JSON = _REPO_ROOT / "config" / "species.json"

# Wagoner/Malaney-Fowler captures Famiano cites but this module does not yet
# implement (see module docstring). Exposed so callers/tests can assert the
# gap is known rather than silently absent.
MISSING_SPECIAL_CAPTURES: Tuple[str, ...] = (
    "p(n,g)d", "6Li(n,g)7Li", "9Be(n,g)10Be", "10Be(n,g)11Be",
    "10B(n,g)11B", "11B(n,g)12B", "17O(n,g)18O",   # in Wagoner_coeff.csv, formula not implemented
    "8Li(n,g)9Li", "7Be(n,g)8Be", "11C(n,g)12C",    # not even present in Wagoner_coeff.csv
)


class IngestReport:
    """Collects per-source load/skip counts and skip reasons for transparency."""

    def __init__(self) -> None:
        self.loaded: Dict[str, int] = {}
        self.skipped: Dict[str, List[str]] = {}

    def add_loaded(self, source: str, n: int = 1) -> None:
        self.loaded[source] = self.loaded.get(source, 0) + n

    def add_skipped(self, source: str, reason: str) -> None:
        self.skipped.setdefault(source, []).append(reason)

    def summary(self) -> str:
        lines = ["Rate ingestion report:"]
        for src in sorted(set(self.loaded) | set(self.skipped)):
            n = self.loaded.get(src, 0)
            n_skip = len(self.skipped.get(src, []))
            lines.append(f"  {src}: {n} reactions loaded, {n_skip} skipped")
        return "\n".join(lines)


# =============================================================================
# Species-name aliasing (NACRE/CF88 compact tokens <-> config/species.json keys)
# =============================================================================

def _load_species_table(species_json: Path = _DEFAULT_SPECIES_JSON) -> Dict[str, Tuple[int, int]]:
    """species name -> (A, Z), skipping the '_comment' key."""
    with open(species_json, "r") as f:
        raw = json.load(f)
    return {k: (int(v["A"]), int(v["Z"])) for k, v in raw.items() if not k.startswith("_")}


_ELEMENT_RE = re.compile(r"^(\d+)([A-Za-z]+)$")


def _build_alias_tables(species: Dict[str, Tuple[int, int]]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Build (nacre2_style, cf88_style, nacre1_style) alias maps: compact
    token -> canonical species.json key.

    nacre2_style : lowercase "{A}{symbol}", e.g. "7Be" -> "7be", "11B" -> "11b".
                   Used by NACRE_2/CF88 single-reaction FILENAMES (e.g.
                   "6lipa3he.dat"). p/d/t/n map to themselves; "a" is the
                   alpha alias, "2h"/"3h" are the d/t spellings those
                   filenames use.
    cf88_style   : uppercase "{SYMBOL}{A}", e.g. "7Be" -> "BE7", "11B" -> "B11".
                   Used by CF88's web_data.dat HEADER LINES (e.g. "LI6").
                   p/d/t map to "H1"/"H2"/"H3"; n maps to bare "N".
    nacre1_style : lowercase "{symbol}{A}", e.g. "7Be" -> "be7", "11B" -> "b11".
                   Used by NACRE_1_coeff.csv's 'elements' COLUMN (e.g.
                   "be7;p;li6") -- note this is symbol-then-digit, the
                   OPPOSITE order from nacre2_style, confirmed by checking
                   which ordering makes each row's tokens mass/charge-
                   balance (see load_nacre1's docstring comment). p/d/t/n/a
                   map the same bare way as nacre2_style.
    """
    nacre2_style: Dict[str, str] = {}
    cf88_style: Dict[str, str] = {}
    nacre1_style: Dict[str, str] = {}

    for canon in species:
        if canon in ("n", "p", "d", "t"):
            continue
        m = _ELEMENT_RE.match(canon)
        if not m:
            continue
        digits, letters = m.group(1), m.group(2)
        nacre2_style[(digits + letters).lower()] = canon
        cf88_style[(letters + digits).upper()] = canon
        nacre1_style[(letters + digits).lower()] = canon

    for style in (nacre2_style, nacre1_style):
        style["p"] = "p"
        style["d"] = "d"
        style["t"] = "t"
        style["n"] = "n"
        style["a"] = "4He"
    nacre2_style["2h"] = "d"
    nacre2_style["3h"] = "t"

    # Target/residual roles use the full element+mass form even for
    # hydrogen isotopes ("H1(P,E+NU)H2"), but the reactant/ejectile slot
    # uses bare light-particle symbols instead ("LI6(P,G)BE7" -- "P" here,
    # not "H1"). Both spellings must resolve since _decode_cf88_particle_list
    # and the target/residual lookup share this one alias table.
    cf88_style["H1"] = "p"
    cf88_style["H2"] = "d"
    cf88_style["H3"] = "t"
    cf88_style["P"] = "p"
    cf88_style["D"] = "d"
    cf88_style["T"] = "t"
    cf88_style["N"] = "n"
    cf88_style["A"] = "4He"

    return nacre2_style, cf88_style, nacre1_style


# =============================================================================
# Analytic-fit rate function (NACRE_1)
# =============================================================================

@dataclass
class _AnalyticFitTerm:
    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float

    def __call__(self, T9: float) -> float:
        if T9 <= 0.0:
            return 0.0
        ln_rate = (
            self.a0
            + self.a1 / T9
            + self.a2 / T9 ** (1.0 / 3.0)
            + self.a3 * T9 ** (1.0 / 3.0)
            + self.a4 * T9
            + self.a5 * T9 ** (5.0 / 3.0)
            + self.a6 * np.log(T9)
        )
        return float(np.exp(ln_rate))


class _SummedAnalyticRate:
    """Sum of several analytic fit terms (NACRE multi-term fits: e.g. a
    non-resonant term plus one or more resonance terms for the same
    reaction), converted from molar N_A<sv> to per-pair <sv> in cm^3/s.
    """

    def __init__(self, terms: List[_AnalyticFitTerm]) -> None:
        self.terms = terms

    def __call__(self, T9: float) -> float:
        total_molar = sum(term(T9) for term in self.terms)
        return total_molar / AVOGADRO


# =============================================================================
# Tabulated rate function (NACRE_2, CF88, Table-1 specials, Bao)
# =============================================================================

class _LogLinearTabulatedRate:
    """
    Log-linear interpolation of a molar rate N_A<sv> vs T9, converted to
    per-pair <sv> in cm^3/s. Flat extrapolation beyond the table's T9
    range (clamped to the nearest edge value) rather than blowing up.
    """

    def __init__(self, T9_grid: Array, molar_rate_grid: Array) -> None:
        order = np.argsort(T9_grid)
        self.T9 = np.asarray(T9_grid, dtype=float)[order]
        rate = np.asarray(molar_rate_grid, dtype=float)[order]
        rate = np.clip(rate, 1e-300, None)  # guard log(0)
        self.log_rate = np.log(rate)

    def __call__(self, T9: float) -> float:
        T9_clamped = float(np.clip(T9, self.T9[0], self.T9[-1]))
        log_r = np.interp(T9_clamped, self.T9, self.log_rate)
        return float(np.exp(log_r)) / AVOGADRO


class _KevTabulatedRate:
    """Bao-style rate tabulated vs kT [keV]; converts T9 -> kT before the
    same log-linear interpolation as _LogLinearTabulatedRate."""

    def __init__(self, kT_grid: Array, molar_rate_grid: Array) -> None:
        order = np.argsort(kT_grid)
        self.kT = np.asarray(kT_grid, dtype=float)[order]
        rate = np.clip(np.asarray(molar_rate_grid, dtype=float)[order], 1e-300, None)
        self.log_rate = np.log(rate)

    def __call__(self, T9: float) -> float:
        kT = T9 * KEV_PER_T9
        kT_clamped = float(np.clip(kT, self.kT[0], self.kT[-1]))
        log_r = np.interp(kT_clamped, self.kT, self.log_rate)
        return float(np.exp(log_r)) / AVOGADRO


# =============================================================================
# NACRE_1_coeff.csv
# =============================================================================
#
# The 'elements' column's token order is fixed per row length for 4- and
# 5-token rows (confirmed against ~20 known Q-values by hand -- see the
# ingestion writeup this module accompanies):
#   4 tokens -> [projectile, target, ejectile, residual]
#   5 tokens -> [target, projectile, ejectile1, ejectile2, residual]
#
# 3-token (capture) rows are NOT fixed-order: both a forward-capture row
# and a reverse-photodisintegration row appear for the same reaction, with
# the reverse row's tokens in the OPPOSITE order and Q negated, e.g.:
#     "d;d;he4"   Q=+23.847   (forward: d+d -> 4He)
#     "he4;d;d"   Q=-23.847   (reverse: same reaction, tokens reversed)
# Unlike the 4-/5-token case, this ambiguity IS resolvable by mass/charge
# conservation: a 3-token row's "target;projectile;residual" reading and
# its "residual;projectile;target" reading are two DIFFERENT candidate
# reactions (not a trivial time-reversal of one one), so at most one
# conserves A and Z. (For 4-/5-token rows, forward- and reverse-reading
# are always both conserving -- reversing a balanced equation keeps it
# balanced -- so conservation can't disambiguate there; that's why those
# arities use a single fixed template instead of a conservation-based
# choice between two.) A 3-token row where neither, or implausibly both,
# reading conserves is skipped rather than guessed.
#
# Rows sharing one 'elements' string (typically a non-resonant term plus
# one or more resonance terms) are additive fit terms -- NACRE reports
# multi-term fits -- and their rates are summed.

def _nacre1_species_from_tokens(
    tokens: List[str],
    nacre_alias: Dict[str, str],
    species: Dict[str, Tuple[int, int]],
) -> Optional[Tuple[str, str, List[str], str, bool]]:
    """
    Return (projectile, target, ejectiles, residual, q_sign_flip) or None
    if `tokens` doesn't resolve to a species.json-known, mass/charge-
    conserving reaction. See the block comment above for the per-length
    token-order convention.
    """

    def resolve(tok: str) -> Optional[str]:
        return nacre_alias.get(tok)

    resolved = [resolve(t) for t in tokens]
    if any(r is None for r in resolved):
        return None

    def conserves(proj: str, tgt: str, ejectiles: List[str], residual: str) -> bool:
        A_in = species[proj][0] + species[tgt][0]
        Z_in = species[proj][1] + species[tgt][1]
        A_out = sum(species[e][0] for e in ejectiles) + species[residual][0]
        Z_out = sum(species[e][1] for e in ejectiles) + species[residual][1]
        return A_in == A_out and Z_in == Z_out

    n = len(resolved)
    if n == 4:
        proj, tgt, ej, res = resolved[0], resolved[1], resolved[2], resolved[3]
        if not conserves(proj, tgt, [ej], res):
            return None
        return proj, tgt, [ej], res, False

    if n == 5:
        tgt, proj, ej1, ej2, res = resolved[0], resolved[1], resolved[2], resolved[3], resolved[4]
        if not conserves(proj, tgt, [ej1, ej2], res):
            return None
        return proj, tgt, [ej1, ej2], res, False

    if n == 3:
        # Candidate A: tokens = [target, projectile, residual]
        tgt_a, proj_a, res_a = resolved[0], resolved[1], resolved[2]
        a_ok = conserves(proj_a, tgt_a, [], res_a)
        # Candidate B: tokens = [residual, projectile, target] (reverse row)
        res_b, proj_b, tgt_b = resolved[0], resolved[1], resolved[2]
        b_ok = conserves(proj_b, tgt_b, [], res_b)

        if a_ok and not b_ok:
            return proj_a, tgt_a, [], res_a, False
        if b_ok and not a_ok:
            return proj_b, tgt_b, [], res_b, True
        return None  # neither or both conserve -- unparseable, skip

    return None


def load_nacre1(
    path: Path,
    species: Dict[str, Tuple[int, int]],
    nacre_alias: Dict[str, str],
    report: IngestReport,
) -> List[ThermonuclearReaction]:
    source = "NACRE_1"
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))

    # groups is keyed by the canonical-identity tuple built below (not the
    # raw elements string -- see canon_key's docstring comment).
    groups: Dict[Tuple, List[_AnalyticFitTerm]] = {}
    identity_cache: Dict[str, Optional[Tuple]] = {}

    # Key fit terms by the CANONICAL reaction identity (reactant/product
    # stoich), not the raw 'elements' string: NACRE_1 sometimes spells the
    # same physical reaction two different ways in different rows (e.g.
    # "be7;p;li6" and "p;li6;be7" both decode to 6Li+p->7Be, as two
    # separate empirical fit terms of one total rate) -- grouping by the
    # raw string would silently create two competing library entries for
    # one reaction instead of summing their terms as NACRE intends.
    def canon_key(proj: str, tgt: str, ejectiles: List[str], residual: str):
        reactants: Dict[str, int] = {}
        if proj == tgt:
            reactants[proj] = 2
        else:
            reactants[proj] = reactants.get(proj, 0) + 1
            reactants[tgt] = reactants.get(tgt, 0) + 1
        products: Dict[str, int] = {}
        for ej in ejectiles:
            products[ej] = products.get(ej, 0) + 1
        products[residual] = products.get(residual, 0) + 1
        return (frozenset(reactants.items()), frozenset(products.items())), reactants, products

    identities: Dict[str, Tuple] = {}   # canon_key -> (reactants_stoich, products_stoich, q_physical)

    for row in rows:
        elements = row["elements"].strip()
        tokens = [t for t in elements.split(";") if t]

        if elements not in identity_cache:
            parsed = _nacre1_species_from_tokens(tokens, nacre_alias, species)
            if parsed is None:
                identity_cache[elements] = None
            else:
                proj, tgt, ejectiles, residual, was_reversed = parsed
                q_file = float(row["Q"])
                q_physical = -q_file if was_reversed else q_file
                key, reactants_stoich, products_stoich = canon_key(proj, tgt, ejectiles, residual)
                identity_cache[elements] = (key, reactants_stoich, products_stoich, q_physical)
                identities.setdefault(key, (reactants_stoich, products_stoich, q_physical))

        identity = identity_cache[elements]
        if identity is None:
            report.add_skipped(source, f"out-of-scope species in '{elements}'")
            continue

        try:
            term = _AnalyticFitTerm(
                a0=float(row["a0"]), a1=float(row["a1"]), a2=float(row["a2"]),
                a3=float(row["a3"]), a4=float(row["a4"]), a5=float(row["a5"]),
                a6=float(row["a6"]),
            )
        except (KeyError, ValueError):
            report.add_skipped(source, f"malformed fit coefficients for '{elements}'")
            continue

        groups.setdefault(identity[0], []).append(term)

    reactions: List[ThermonuclearReaction] = []
    for key, terms in groups.items():
        reactants_stoich, products_stoich, q_physical = identities[key]

        reactions.append(ThermonuclearReaction(
            reactants_stoich=dict(reactants_stoich),
            products_stoich=dict(products_stoich),
            rate_func=_SummedAnalyticRate(terms),
            q_value_mev=q_physical,
            source="NACRE 1999 (Angulo et al.)",
        ))
        report.add_loaded(source)

    return reactions


# =============================================================================
# NACRE_2 and CF88 single-reaction tabulated files
# =============================================================================
#
# Hand-verified (target, projectile, ejectile_mult, residual) for every file
# in data/ReactionRates/Thermal/NACRE_2/ and the one single-reaction CF88
# file, cross-checked against known reaction Q-values -- see the ingestion
# writeup this module accompanies. Using an explicit table instead of a
# general compact-string tokenizer avoids silently mis-splitting an
# unfamiliar filename.

_NACRE2_REACTIONS: Dict[str, Tuple[str, str, Dict[str, int], str]] = {
    "10bpa7be.dat":     ("10B", "p", {"4He": 1}, "7Be"),
    "10bpg11c.dat":     ("10B", "p", {}, "11C"),
    "11ban14n.dat":     ("11B", "4He", {"n": 1}, "14N"),
    "11bpa8be.dat":     ("11B", "p", {"4He": 1}, "8Be"),
    "11bpg12c.dat":     ("11B", "p", {}, "12C"),
    "12cag16o.dat":     ("12C", "4He", {}, "16O"),
    "12cpg13n.dat":     ("12C", "p", {}, "13N"),
    "13can16o.dat":     ("13C", "4He", {"n": 1}, "16O"),
    "13cpg14n.dat":     ("13C", "p", {}, "14N"),
    "13npg14o.dat":     ("13N", "p", {}, "14O"),
    "14npg15o.dat":     ("14N", "p", {}, "15O"),
    "15npa12c.dat":     ("15N", "p", {"4He": 1}, "12C"),
    "15npg16o.dat":     ("15N", "p", {}, "16O"),
    "2hag6li.dat":      ("d", "4He", {}, "6Li"),
    "2hdg4he.dat":      ("d", "d", {}, "4He"),
    "2hdn3he.dat":      ("d", "d", {"n": 1}, "3He"),
    "2hdp3h.dat":       ("d", "d", {"p": 1}, "t"),
    "2hpg3he.dat":      ("d", "p", {}, "3He"),
    "3hag7li.dat":      ("t", "4He", {}, "7Li"),
    "3hdn4he.dat":      ("t", "d", {"n": 1}, "4He"),
    "3he3he2p4he.dat":  ("3He", "3He", {"p": 2}, "4He"),
    "3heag7be.dat":     ("3He", "4He", {}, "7Be"),
    "3hedp4he.dat":     ("3He", "d", {"p": 1}, "4He"),
    "6lipa3he.dat":     ("6Li", "p", {"4He": 1}, "3He"),
    "6lipg7be.dat":     ("6Li", "p", {}, "7Be"),
    "7beag11c.dat":     ("7Be", "4He", {}, "11C"),
    "7bepg8b.dat":      ("7Be", "p", {}, "8B"),
    "7liag11b.dat":     ("7Li", "4He", {}, "11B"),
    "7lipa4he.dat":     ("7Li", "p", {"4He": 1}, "4He"),
    "7lipg8be.dat":     ("7Li", "p", {}, "8Be"),
    "9bean12c.dat":     ("9Be", "4He", {"n": 1}, "12C"),
    "9bepa6li.dat":     ("9Be", "p", {"4He": 1}, "6Li"),
    "9bepd8be.dat":     ("9Be", "p", {"d": 1}, "8Be"),
    "9bepg10b.dat":     ("9Be", "p", {}, "10B"),
}

# CF88's own copy of 3H(d,n)4He, same convention/table as above -- kept
# separate from NACRE_2's version rather than merged, so both sources
# remain independently inspectable (they should roughly agree).
_CF88_SINGLE_REACTIONS: Dict[str, Tuple[str, str, Dict[str, int], str]] = {
    "3hdn4he.dat": ("t", "d", {"n": 1}, "4He"),
}


def _load_tabulated_dat(path: Path) -> Tuple[Array, Array]:
    T9s: List[float] = []
    rates: List[float] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                T9s.append(float(parts[0]))
                rates.append(float(parts[1]))
            except (ValueError, IndexError):
                continue
    return np.asarray(T9s), np.asarray(rates)


def _make_reaction_from_2body(
    target: str, projectile: str, ejectiles: Dict[str, int], residual: str,
    rate_func, source: str,
) -> ThermonuclearReaction:
    reactants_stoich: Dict[str, int] = {}
    if target == projectile:
        reactants_stoich[target] = 2
    else:
        reactants_stoich[target] = reactants_stoich.get(target, 0) + 1
        reactants_stoich[projectile] = reactants_stoich.get(projectile, 0) + 1

    products_stoich: Dict[str, int] = dict(ejectiles)
    products_stoich[residual] = products_stoich.get(residual, 0) + 1

    return ThermonuclearReaction(
        reactants_stoich=reactants_stoich,
        products_stoich=products_stoich,
        rate_func=rate_func,
        source=source,
    )


def load_nacre2(thermal_dir: Path, report: IngestReport) -> List[ThermonuclearReaction]:
    source = "NACRE_2"
    nacre2_dir = thermal_dir / "NACRE_2"
    reactions: List[ThermonuclearReaction] = []

    for filename, (target, projectile, ejectiles, residual) in _NACRE2_REACTIONS.items():
        filepath = nacre2_dir / filename
        if not filepath.exists():
            report.add_skipped(source, f"missing file {filename}")
            continue
        T9, rate = _load_tabulated_dat(filepath)
        if len(T9) < 2:
            report.add_skipped(source, f"insufficient data in {filename}")
            continue
        reactions.append(_make_reaction_from_2body(
            target, projectile, ejectiles, residual,
            _LogLinearTabulatedRate(T9, rate),
            source="NACRE 1999 (Angulo et al.), tabulated",
        ))
        report.add_loaded(source)

    return reactions


def load_cf88_single(thermal_dir: Path, report: IngestReport) -> List[ThermonuclearReaction]:
    source = "CF88_single"
    cf88_dir = thermal_dir / "CF88"
    reactions: List[ThermonuclearReaction] = []

    for filename, (target, projectile, ejectiles, residual) in _CF88_SINGLE_REACTIONS.items():
        filepath = cf88_dir / filename
        if not filepath.exists():
            report.add_skipped(source, f"missing file {filename}")
            continue
        T9, rate = _load_tabulated_dat(filepath)
        if len(T9) < 2:
            report.add_skipped(source, f"insufficient data in {filename}")
            continue
        reactions.append(_make_reaction_from_2body(
            target, projectile, ejectiles, residual,
            _LogLinearTabulatedRate(T9, rate),
            source="Caughlan & Fowler 1988, tabulated",
        ))
        report.add_loaded(source)

    return reactions


# =============================================================================
# CF88 web_data.dat (bundled multi-reaction file)
# =============================================================================

_CF88_HEADER_RE = re.compile(
    r"^([A-Z]+\d+)\(([A-Z0-9+]+),([A-Z0-9+]+)\)([A-Z]+\d+)$"
)
_CF88_MULT_TOKEN_RE = re.compile(r"^(\d*)([A-Z]+)$")


def _decode_cf88_particle_list(token: str, cf88_alias: Dict[str, str]) -> Optional[Dict[str, int]]:
    """
    Decode a CF88 reactant/ejectile token like 'P', 'A', '2A', 'NN', 'E+NU'
    into a stoich dict of canonical species, or None if any piece is
    unrecognized (weak-interaction tokens like 'E+NU' are deliberately not
    in cf88_alias, so those reactions are skipped rather than guessed).
    """
    out: Dict[str, int] = {}
    for piece in token.split("+"):
        if piece == "G":
            continue  # radiative capture/decay: no ejectile particle
        m = _CF88_MULT_TOKEN_RE.match(piece)
        if not m:
            return None
        mult = int(m.group(1)) if m.group(1) else 1
        symbol = m.group(2)
        # 'NN' (two neutrons as one token) is a special case, not mult+symbol.
        if symbol == "NN":
            symbol, mult = "N", 2 * mult
        canon = cf88_alias.get(symbol)
        if canon is None:
            return None
        out[canon] = out.get(canon, 0) + mult
    return out


def load_cf88_bundle(
    thermal_dir: Path,
    species: Dict[str, Tuple[int, int]],
    cf88_alias: Dict[str, str],
    report: IngestReport,
) -> List[ThermonuclearReaction]:
    source = "CF88_web_data"
    path = thermal_dir / "CF88" / "web_data.dat"
    if not path.exists():
        report.add_skipped(source, "web_data.dat not found")
        return []

    with open(path, "r", encoding="ascii", errors="replace") as f:
        content = f.read()

    # Split into (header, block-of-numeric-lines) segments. EVERY header
    # line must delimit a block, including ones _CF88_HEADER_RE can't
    # decode (e.g. "B11(P,A)2HE4", "HE4(A)BE8", "HE3(E-,NU)H3"). Previously
    # only decodable headers were kept as delimiters, so an undecodable
    # header's table was silently appended to the PREVIOUS reaction's
    # block and interpolated together with it -- e.g. 11B(p,n)11C picked
    # up B11(P,A)2HE4's values (2.98e6 instead of 9.86e-16 at T9=0.6).
    header_positions = [
        (m.start(), m.group(1))
        for m in re.finditer(r"^\s{2,}(\S+\(\S*\)\S*)\s*$", content, re.MULTILINE)
    ]

    reactions: List[ThermonuclearReaction] = []
    for i, (pos, header) in enumerate(header_positions):
        end = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(content)
        block = content[pos:end]

        m = _CF88_HEADER_RE.match(header)
        if m is None:
            report.add_skipped(source, f"unparsed header format '{header}'")
            continue
        target_tok, react_tok, eject_tok, residual_tok = m.groups()

        target = cf88_alias.get(target_tok)
        residual = cf88_alias.get(residual_tok)
        reactants = _decode_cf88_particle_list(react_tok, cf88_alias)
        ejectiles = _decode_cf88_particle_list(eject_tok, cf88_alias)

        if target is None or residual is None or reactants is None or ejectiles is None:
            report.add_skipped(source, f"out-of-scope or unrecognized species in '{header}'")
            continue
        if len(reactants) != 1 or sum(reactants.values()) != 1:
            report.add_skipped(source, f"unsupported multi-particle projectile in '{header}'")
            continue
        projectile = next(iter(reactants))

        if target not in species or residual not in species or projectile not in species:
            report.add_skipped(source, f"species not in config/species.json for '{header}'")
            continue
        if not all(sp in species for sp in ejectiles):
            report.add_skipped(source, f"ejectile species not in config/species.json for '{header}'")
            continue

        T9s: List[float] = []
        rates: List[float] = []
        for line in block.splitlines()[1:]:
            line = line.strip()
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                T9s.append(float(parts[0]))
                rates.append(float(parts[1]))
            except ValueError:
                continue

        if len(T9s) < 2:
            report.add_skipped(source, f"insufficient tabulated data for '{header}'")
            continue
        if len(set(T9s)) != len(T9s):
            # Repeated T9 values mean two tables were merged into one block.
            report.add_skipped(source, f"repeated T9 grid points (merged tables?) in '{header}'")
            continue

        reactions.append(_make_reaction_from_2body(
            target, projectile, ejectiles, residual,
            _LogLinearTabulatedRate(np.asarray(T9s), np.asarray(rates)),
            source="Caughlan & Fowler 1988, tabulated (web_data.dat)",
        ))
        report.add_loaded(source)

    return reactions


# =============================================================================
# Table 1 special-case reactions (14Cag18O.csv, 17Ona14C.csv)
# =============================================================================

_TABLE1_SPECIALS: Dict[str, Tuple[str, str, Dict[str, int], str]] = {
    "14Cag18O.csv":  ("14C", "4He", {}, "18O"),     # 14C(a,g)18O, Famiano Table 1 ref. 7 (Funck & Langanke 1989)
    "17Ona14C.csv":  ("17O", "n", {"4He": 1}, "14C"),  # 17O(n,a)14C, Famiano Table 1 ref. 8 (Schatz et al. 1993)
}


def load_table1_specials(thermal_dir: Path, report: IngestReport) -> List[ThermonuclearReaction]:
    source = "Table1_specials"
    reactions: List[ThermonuclearReaction] = []

    for filename, (target, projectile, ejectiles, residual) in _TABLE1_SPECIALS.items():
        path = thermal_dir / filename
        if not path.exists():
            report.add_skipped(source, f"missing file {filename}")
            continue

        T9s: List[float] = []
        rates: List[float] = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#") or row[0].strip().lower() == "t9":
                    continue
                try:
                    T9s.append(float(row[0]))
                    rates.append(float(row[1]))
                except (ValueError, IndexError):
                    continue

        if len(T9s) < 2:
            report.add_skipped(source, f"insufficient data in {filename}")
            continue

        reactions.append(_make_reaction_from_2body(
            target, projectile, ejectiles, residual,
            _LogLinearTabulatedRate(np.asarray(T9s), np.asarray(rates)),
            source=f"Famiano Table 1 special case ({filename})",
        ))
        report.add_loaded(source)

    return reactions


# =============================================================================
# Bao et al. 2000 radiative neutron-capture rates
# =============================================================================

_ELEMENT_SYMBOL_TO_Z: Dict[str, int] = {
    "n": 0, "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
}
_KEV_COLUMNS = ("5", "10", "15", "20", "25", "30", "40", "50", "60", "80", "100")


def load_bao(
    thermal_dir: Path,
    species: Dict[str, Tuple[int, int]],
    report: IngestReport,
) -> List[ThermonuclearReaction]:
    source = "Bao"
    path = thermal_dir / "Bao_rad_n_capture_rates.csv"
    if not path.exists():
        report.add_skipped(source, "file not found")
        return []

    z_to_name = {z: sym for sym, z in _ELEMENT_SYMBOL_TO_Z.items()}
    az_to_canon = {az: name for name, az in species.items()}

    reactions: List[ThermonuclearReaction] = []
    with open(path, "r") as f:
        lines = [l for l in f if not l.startswith("#")]
    reader = csv.DictReader(lines)

    for row in reader:
        symbol = row["Element"].strip()
        try:
            A = int(row["A"].strip())
        except (KeyError, ValueError):
            continue
        Z = _ELEMENT_SYMBOL_TO_Z.get(symbol)
        if Z is None:
            continue  # out of Famiano's n..20Ne scope

        target = az_to_canon.get((A, Z))
        residual = az_to_canon.get((A + 1, Z))
        if target is None or residual is None:
            report.add_skipped(source, f"{symbol}{A}(n,g) -- target or product not in config/species.json")
            continue

        kTs: List[float] = []
        rates: List[float] = []
        for col in _KEV_COLUMNS:
            val = row.get(col, "").strip()
            if not val:
                continue
            try:
                rates.append(float(val))
                kTs.append(float(col))
            except ValueError:
                continue

        if len(kTs) < 2:
            report.add_skipped(source, f"{symbol}{A}(n,g) -- insufficient kT coverage")
            continue

        reactions.append(_make_reaction_from_2body(
            target, "n", {}, residual,
            _KevTabulatedRate(np.asarray(kTs), np.asarray(rates)),
            source="Bao et al. 2000, tabulated vs kT",
        ))
        report.add_loaded(source)

    return reactions


# =============================================================================
# Top-level assembly
# =============================================================================

_UNBOUND_8BE = "8Be"


def _rewrite_unbound_8be(rxn: ThermonuclearReaction) -> None:
    """
    Replace an 8Be product with 2 4He in place.

    8Be breaks up into two alphas in ~1e-16 s and has no entry in
    decay.DECAY_TABLE (a decay constant that large would make the network
    needlessly stiff), so leaving it as a product let it accumulate as a
    spurious long-lived reservoir -- which 4He+8Be->12C then burned into
    every species from 12C upward. Rewriting at ingestion also makes
    11B(p,a)8Be (NACRE_2) and 11B(p,3a) (NACRE_1) share one identity, so
    the cross-source dedup in build_thermonuclear_library() keeps only one
    of these two representations of the same rate.
    """
    if _UNBOUND_8BE not in rxn.products_stoich:
        return
    original_name = rxn.name()
    mult = rxn.products_stoich.pop(_UNBOUND_8BE)
    rxn.products_stoich["4He"] = rxn.products_stoich.get("4He", 0) + 2 * mult
    rxn.label = f"{original_name} [8Be->2a]"


def build_thermonuclear_library(
    thermal_dir: Path = _DEFAULT_THERMAL_DIR,
    species_json: Path = _DEFAULT_SPECIES_JSON,
) -> Tuple[ThermonuclearLibrary, IngestReport]:
    """
    Parse every implemented rate-data source and assemble a populated
    ThermonuclearLibrary. See module docstring for what's covered and what
    (Wagoner_coeff.csv) is deliberately left out.
    """
    species = _load_species_table(species_json)
    nacre2_alias, cf88_alias, nacre1_alias = _build_alias_tables(species)
    report = IngestReport()

    # Priority order for cross-source deduplication below: when the same
    # physical reaction is covered by more than one source, keep only the
    # highest-priority one rather than summing them (they're alternative
    # representations of one published rate, not independent physical
    # contributions -- summing would double the rate). Direct tabulated
    # sources are preferred over NACRE_1's analytic fits: cross-checking
    # NACRE_1 against NACRE_2 for 6Li(p,g)7Be during development turned up
    # a ~1e38 discrepancy at T9=0.6 between two differently-worded NACRE_1
    # rows for that SAME reaction, traced to one row's fit evidently only
    # being valid over a temperature range this file doesn't record --
    # extrapolating it outside that range silently produced a wildly wrong
    # rate. Tabulated sources carry no such extrapolation risk within
    # their stated T9 grid (0.001-10, covering Famiano's full range).
    per_source = [
        ("NACRE_2", load_nacre2(thermal_dir, report)),
        ("CF88_single", load_cf88_single(thermal_dir, report)),
        ("CF88_web_data", load_cf88_bundle(thermal_dir, species, cf88_alias, report)),
        ("Table1_specials", load_table1_specials(thermal_dir, report)),
        ("Bao", load_bao(thermal_dir, species, report)),
        ("NACRE_1", load_nacre1(thermal_dir / "NACRE_1_coeff.csv", species, nacre1_alias, report)),
    ]

    library = ThermonuclearLibrary()
    seen_identities: Dict[Tuple, str] = {}
    for source_name, reactions in per_source:
        for rxn in reactions:
            if _UNBOUND_8BE in rxn.reactants_stoich:
                report.add_skipped(
                    source_name,
                    f"{rxn.name()}: 8Be is unbound (tau ~ 1e-16 s) and never "
                    f"accumulates as a reactant -- its products are rewritten to 2a",
                )
                report.loaded[source_name] = report.loaded.get(source_name, 0) - 1
                continue

            # Radiative capture = a single product nucleus (e.g. 7Li(p,g)8Be).
            # Recorded BEFORE the 8Be rewrite so the dedup key below keeps
            # 7Li(p,g)8Be->2a distinct from 7Li(p,a)a, which has the same
            # final particles but is a separate physical channel.
            is_radiative = (
                len(rxn.products_stoich) == 1
                and next(iter(rxn.products_stoich.values())) == 1
            )
            _rewrite_unbound_8be(rxn)

            key = (
                frozenset(rxn.reactants_stoich.items()),
                frozenset(rxn.products_stoich.items()),
                is_radiative,
            )
            if key in seen_identities:
                report.add_skipped(
                    source_name,
                    f"duplicate of {rxn.name()} already loaded from {seen_identities[key]}",
                )
                report.loaded[source_name] = report.loaded.get(source_name, 0) - 1
                continue
            seen_identities[key] = source_name
            library.add_reaction(rxn)

    return library, report


if __name__ == "__main__":
    lib, rep = build_thermonuclear_library()
    print(rep.summary())
    print(f"Total reactions in library: {len(lib)}")
