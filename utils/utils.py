from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union
import json
import re

c = 3*10**8 # speed of light -- m/s
AMU_TO_MEV = 931.49410242 # conversion from mass number to MeV/c^2
m_e = 0.511 # mass of electron -- MeV/c^2
m_p = 938.272 # mass of proton -- MeV/c^2

r_e = 2.818e-15 # electron radius in m

def beta(E, A):

    m = A * AMU_TO_MEV

    v = c * np.sqrt(1 - 1 / (1 + E / m)**2 )

    beta = v/c

    return beta


def lorentz_factor(beta):

    gamma = 1 / np.sqrt(1 - beta**2)

    return gamma

"""
utils/qvalue.py

Q-value utilities for the non-thermal reaction network.

This module is intentionally designed to match reactions.py, which expects
one of the following callables to exist:

    q_value_mev(reactants_stoich, products_stoich)
    compute_q_value_mev(reactants_stoich, products_stoich)

Both should accept stoichiometric dictionaries and return a float in MeV.

"""


# -------------------------------------------------------------------------
# Species canonicalization
# -------------------------------------------------------------------------

_ALIAS_TO_CANONICAL = {
    "a": "4He",
    "alpha": "4He",
    "he4": "4He",
    "He4": "4He",
    "he3": "3He",
    "He3": "3He",
    "proton": "p",
    "neutron": "n",
    "deuteron": "d",
    "triton": "t",
    "1H": "p",
    "H1": "p",
    "2H": "d",
    "H2": "d",
    "3H": "t",
    "H3": "t",
}


def canonical_species_name(species: str) -> str:
    s = str(species).strip()
    if s in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[s]

    # li7 -> 7Li
    m1 = re.fullmatch(r"([A-Za-z]+)(\d+)", s)
    if m1:
        sym, A = m1.groups()
        sym = sym[0].upper() + sym[1:].lower()
        trial = f"{A}{sym}"
        return _ALIAS_TO_CANONICAL.get(trial, trial)

    # 7li -> 7Li
    m2 = re.fullmatch(r"(\d+)([A-Za-z]+)", s)
    if m2:
        A, sym = m2.groups()
        sym = sym[0].upper() + sym[1:].lower()
        trial = f"{A}{sym}"
        return _ALIAS_TO_CANONICAL.get(trial, trial)

    return s


# -------------------------------------------------------------------------
# Minimal built-in masses for light particles
# Atomic masses in u for neutral atoms where applicable.
# Using atomic masses is fine as long as electrons cancel consistently.
# -------------------------------------------------------------------------

_BUILTIN_MASS_TABLE_U: Dict[str, float] = {
    # Light particles (AME 2020)
    "n":    1.00866491588,
    "p":    1.00782503223,
    "d":    2.01410177812,
    "t":    3.0160492779,
    "3He":  3.0160293201,
    "4He":  4.00260325413,
    # Li / Be / B
    "6Li":  6.0151228874,
    "7Li":  7.0160034366,
    "7Be":  7.0169292400,
    "8Li":  8.0224668,
    "9Be":  9.0121831,
    "10Be": 10.0135338,
    "10B":  10.0129369,
    "11B":  11.0093054,
    # Carbon
    "11C":  11.0114336,
    "12C":  12.0000000,
    "13C":  13.0033548,
    "14C":  14.0032420,
    # Nitrogen
    "13N":  13.0057386,
    "14N":  14.0030740,
    "15N":  15.0001089,
    "16N":  16.0061017,
    # Oxygen
    "15O":  15.0030656,
    "16O":  15.9949146,
    "17O":  16.9991315,
    "18O":  17.9991610,
    # Fluorine / Neon
    "17F":  17.0020952,
    "18F":  18.0009377,
    "20Ne": 19.9924402,
    "21Ne": 20.9938853,
}


def load_mass_table(path: Optional[str | Path] = None) -> Dict[str, float]:
    """
    Load masses from JSON if provided; otherwise return built-in light masses.

    Expected JSON format:
        {
          "p": 1.00782503223,
          "4He": 4.00260325413,
          "12C": 12.0,
          ...
        }
    """
    table = dict(_BUILTIN_MASS_TABLE_U)

    if path is None:
        return table

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mass table file not found: {p}")

    with p.open("r") as f:
        data = json.load(f)

    for species, mass in data.items():
        table[canonical_species_name(species)] = float(mass)

    return table


def get_species_mass_u(
    species: str,
    mass_table: Optional[Mapping[str, float]] = None,
) -> float:
    key = canonical_species_name(species)
    table = dict(_BUILTIN_MASS_TABLE_U) if mass_table is None else {
        canonical_species_name(k): float(v) for k, v in mass_table.items()
    }

    if key not in table:
        raise KeyError(
            f"Mass for species '{key}' not found. "
            f"Add it to the external mass table."
        )
    return float(table[key])


def total_mass_u(
    stoich: Mapping[str, int],
    mass_table: Optional[Mapping[str, float]] = None,
) -> float:
    total = 0.0
    for species, count in stoich.items():
        if int(count) == 0:
            continue
        total += int(count) * get_species_mass_u(species, mass_table=mass_table)
    return total


def q_value_mev(
    reactants_stoich: Mapping[str, int],
    products_stoich: Mapping[str, int],
    mass_table: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Compute Q [MeV] from stoichiometric dictionaries.

    Positive Q  -> exothermic
    Negative Q  -> endothermic
    """
    m_initial = total_mass_u(reactants_stoich, mass_table=mass_table)
    m_final = total_mass_u(products_stoich, mass_table=mass_table)
    return (m_initial - m_final) * AMU_TO_MEV


def compute_q_value_mev(
    reactants_stoich: Mapping[str, int],
    products_stoich: Mapping[str, int],
    mass_table: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Alias kept for compatibility with reactions.py.
    """
    return q_value_mev(
        reactants_stoich=reactants_stoich,
        products_stoich=products_stoich,
        mass_table=mass_table,
    )


def reaction_threshold_lab_mev(
    projectile: str,
    target: str,
    q_mev: float,
    mass_table: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Approximate nonrelativistic lab-frame threshold for a + A -> products:

        E_thr ~= -Q * (1 + m_a / m_A),   for Q < 0

    Returns 0 for exothermic reactions.
    """
    if q_mev >= 0.0:
        return 0.0

    m_a = get_species_mass_u(projectile, mass_table=mass_table)
    m_A = get_species_mass_u(target, mass_table=mass_table)
    return (-q_mev) * (1.0 + m_a / m_A)


# -------------------------------------------------------------------------
# Relativistic CM-frame → lab-frame energy conversion
# -------------------------------------------------------------------------

def cm_energy_to_lab_mev(
    t_cm_mev: Union[float, np.ndarray],
    m_projectile_mev: float,
    m_target_mev: float,
) -> Union[float, np.ndarray]:
    """
    Convert total CM-frame kinetic energy to lab-frame projectile kinetic energy
    using the exact relativistic invariant-mass relation.

    Parameters
    ----------
    t_cm_mev : float or array
        Total kinetic energy in the CM frame [MeV].
        This is T_cm = sqrt(s) - m_a - m_A, the sum of both particles'
        kinetic energies in the CM frame (the energy available for the reaction).
    m_projectile_mev : float
        Rest-mass energy of the projectile [MeV/c^2].
    m_target_mev : float
        Rest-mass energy of the target [MeV/c^2].

    Returns
    -------
    float or array
        Lab-frame kinetic energy of the projectile [MeV], with the target at rest.

    Derivation
    ----------
    The Lorentz invariant s (Mandelstam variable) in the lab frame
    (target at rest, projectile with KE = T_lab):

        s = m_a^2 + m_A^2 + 2 m_A (T_lab + m_a)

    In the CM frame, the total CM energy is:

        sqrt(s) = T_cm + m_a + m_A

    Inverting for T_lab:

        T_lab = [s - m_a^2 - m_A^2] / (2 m_A)  -  m_a

    where s = (T_cm + m_a + m_A)^2.

    Non-relativistic limit check (T_cm << m_a, m_A):

        T_lab ≈ T_cm * (m_a + m_A) / m_A   (reduced-mass formula) ✓
    """
    scalar = np.isscalar(t_cm_mev)
    t_cm = np.atleast_1d(np.asarray(t_cm_mev, dtype=float))

    sqrt_s = t_cm + m_projectile_mev + m_target_mev
    s = sqrt_s ** 2

    t_lab = (s - m_projectile_mev**2 - m_target_mev**2) / (2.0 * m_target_mev) - m_projectile_mev

    return float(t_lab[0]) if scalar else t_lab


def cm_to_lab_energy_mev(
    t_cm_mev: Union[float, np.ndarray],
    projectile: str,
    target: str,
    mass_table: Optional[Mapping[str, float]] = None,
) -> Union[float, np.ndarray]:
    """
    Convenience wrapper for :func:`cm_energy_to_lab_mev` that looks up
    rest masses by species name.

    Parameters
    ----------
    t_cm_mev : float or array
        Total CM-frame kinetic energy [MeV].
    projectile : str
        Projectile species name (e.g. ``"p"``, ``"4He"``).
    target : str
        Target species name (e.g. ``"15N"``, ``"12C"``).
    mass_table : mapping, optional
        External mass table {species: mass_u}. Falls back to the built-in
        table which covers all species in the blazar reaction network.

    Returns
    -------
    float or array
        Lab-frame projectile kinetic energy [MeV].

    Note
    ----
    Total cross sections are Lorentz invariant (sigma_total is the same
    in all frames), so the sigma column values do not need to be transformed —
    only the energy axis changes.
    """
    m_a_mev = get_species_mass_u(projectile, mass_table=mass_table) * AMU_TO_MEV
    m_A_mev = get_species_mass_u(target,     mass_table=mass_table) * AMU_TO_MEV
    return cm_energy_to_lab_mev(t_cm_mev, m_a_mev, m_A_mev)