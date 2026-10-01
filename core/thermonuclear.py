"""
thermonuclear.py

Thermonuclear reaction network (Famiano et al. 2002, Section 3, eq. [2]-[3]):
two-body reactions among cloud species at the cloud's bulk temperature,
as opposed to the non-thermal jet-cloud cascade handled in cascade.py.

This module replaces the earlier core/operators/ scaffolding
(ThermonuclearOperator, SimpleThermonuclearOperator, DummyDecayOperator),
which was written against a state shape (state.Y, state.dY,
network.change_in_abund()) that no longer exists. The real state is
NetworkState / CloudState (state.py): abundances live at state.cloud.Y,
species metadata at state.species_data, and every RHS contribution in this
codebase is returned as a plain dYdt_cloud array aligned with
state.cloud.species (see cascade.compute_cascade_step), so that cascade and
thermonuclear contributions can simply be summed by the caller.

Unit contract for rate_func
----------------------------
rate_func(T9) must return the PER-PAIR rate coefficient <sigma*v> in cm^3 s^-1
(i.e. already divided by Avogadro's number), NOT the molar rate N_A<sigma*v>
in cm^3 mol^-1 s^-1 that NACRE/Caughlan-Fowler tables publish directly. When
ingesting those tables, divide their fitted N_A<sigma*v> by N_A = 6.02214076e23
before storing the fit here. This mirrors how cascade.py's
_reaction_sigma_matrix() converts cross sections from mb to cm^2 at the
ingestion boundary rather than leaving the conversion implicit downstream.

No reaction data is populated yet -- ThermonuclearLibrary starts empty.
compute_thermonuclear_dydt() on an empty library is a safe no-op (returns
zeros), so wiring this module in ahead of data ingestion does not change any
existing simulation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np

from state import NetworkState


Array = np.ndarray

AVOGADRO = 6.02214076e23


@dataclass
class ThermonuclearReaction:
    """
    One two-body thermonuclear reaction, e.g. 6Li(p,alpha)3He.

    Attributes
    ----------
    reactants_stoich : Dict[str, int]
        Species consumed, with stoichiometric multiplicity. Must contain
        exactly two entries for two distinct species (mult 1 each) or one
        entry with multiplicity 2 for an identical-particle reaction
        (e.g. 4He+4He).
    products_stoich : Dict[str, int]
        Species produced, with stoichiometric multiplicity.
    rate_func : Callable[[float], float]
        rate_func(T9) -> per-pair <sigma*v> in cm^3 s^-1. See module
        docstring for the NACRE/Caughlan-Fowler unit conversion this
        requires at ingestion time.
    q_value_mev : float, optional
        Reaction Q-value in MeV, for energetics bookkeeping. Not consumed
        by compute_thermonuclear_dydt() yet (cloud temperature is held
        fixed per Famiano Table 3; no energy equation is solved).
    label : str, optional
        Human-readable reaction name, e.g. "6Li(p,a)3He". Auto-generated
        from stoichiometry if omitted.
    source : str
        Citation for the rate (e.g. "NACRE 1999", "Caughlan & Fowler 1988",
        "Famiano Table 1"), for provenance tracking.
    """

    reactants_stoich: Dict[str, int]
    products_stoich: Dict[str, int]
    rate_func: Callable[[float], float]
    q_value_mev: Optional[float] = None
    label: Optional[str] = None
    source: str = ""

    def __post_init__(self) -> None:
        n_reactant_particles = sum(self.reactants_stoich.values())
        if n_reactant_particles != 2:
            raise ValueError(
                "ThermonuclearReaction currently supports only two-body "
                f"reactants; got reactants_stoich={self.reactants_stoich!r} "
                f"summing to {n_reactant_particles} particles."
            )
        if len(self.reactants_stoich) not in (1, 2):
            raise ValueError(
                f"reactants_stoich must have 1 or 2 distinct species; "
                f"got {self.reactants_stoich!r}."
            )

    def name(self) -> str:
        if self.label:
            return self.label

        def fmt(stoich: Mapping[str, int]) -> str:
            parts = []
            for sp, mult in sorted(stoich.items()):
                parts.append(sp if mult == 1 else f"{mult}{sp}")
            return "+".join(parts)

        return f"{fmt(self.reactants_stoich)}->{fmt(self.products_stoich)}"

    @property
    def is_identical_particle(self) -> bool:
        """True for reactions like 4He+4He where both reactants are the same species."""
        return len(self.reactants_stoich) == 1


@dataclass
class ThermonuclearLibrary:
    """
    Collection of thermonuclear reactions.

    Starts empty. Populate via add_reaction() once rate data has been
    ingested (NACRE / Caughlan & Fowler / Bao et al. / Famiano Table 1).
    """

    reactions: List[ThermonuclearReaction] = field(default_factory=list)

    def add_reaction(self, reaction: ThermonuclearReaction) -> None:
        self.reactions.append(reaction)

    def by_reactant(self, species: str) -> List[ThermonuclearReaction]:
        return [r for r in self.reactions if species in r.reactants_stoich]

    def __len__(self) -> int:
        return len(self.reactions)


def T9_from_state(state: NetworkState) -> float:
    """
    Cloud temperature in units of 1e9 K, from state.cloud.temperature_K.

    Famiano's models hold cloud temperature fixed per run (Table 3 lists a
    constant T8/T9 per model), so this is a direct unit conversion, not an
    evolved quantity -- there is no energy equation in this network.
    """
    return float(state.cloud.temperature_K) / 1.0e9


def compute_thermonuclear_dydt(
    state: NetworkState,
    library: ThermonuclearLibrary,
    T9: Optional[float] = None,
    Y: Optional[Array] = None,
) -> Array:
    """
    Compute dY/dt contributions from thermonuclear reactions, aligned with
    state.cloud.species (same convention as cascade.compute_cascade_step's
    dYdt_cloud, so the two can be summed directly by the caller).

    For a two-body reaction i+j -> products with per-pair rate coefficient
    <sigma*v>(T9) [cm^3 s^-1]:

        n_i = density_cm3 * Y_i    (CloudState's number-density convention;
                                     see cascade._cloud_target_number_densities)
        R = n_i * n_j * <sigma*v> / (1 + delta_ij)   [reactions cm^-3 s^-1]
        dY_i/dt -= R / density_cm3   (per unit reactant stoichiometry)
        dY_k/dt += R / density_cm3 * mult_k

    Species not present in state.cloud.species are silently skipped (same
    "not yet tracked" convention cascade.py uses for cloud species lookups),
    since this module ships with an empty reaction list until step 2 (rate
    ingestion) populates it and step 3 (species audit) confirms coverage.

    Parameters
    ----------
    state : NetworkState
    library : ThermonuclearLibrary
    T9 : float, optional
        Cloud temperature in 1e9 K. Defaults to T9_from_state(state).
    Y : np.ndarray, optional
        Abundance vector to evaluate at, aligned with state.cloud.species.
        Defaults to state.cloud.Y. Lets a caller evaluate f(Y) at a
        perturbed or trial abundance vector -- e.g. jacobian.py's finite-
        difference Jacobian, or an implicit-Euler solve -- without having
        to mutate state.cloud.Y and restore it afterward.

    Returns
    -------
    np.ndarray
        dY/dt contribution, shape (len(state.cloud.species),).
    """
    if T9 is None:
        T9 = T9_from_state(state)

    Y_vec = state.cloud.Y if Y is None else np.asarray(Y, dtype=float)
    dYdt = np.zeros_like(Y_vec, dtype=float)

    if not library.reactions:
        return dYdt

    density = float(state.cloud.density_cm3)
    if density <= 0.0:
        return dYdt

    species_index = {sp: i for i, sp in enumerate(state.cloud.species)}

    for rxn in library.reactions:
        reactant_items = list(rxn.reactants_stoich.items())
        if rxn.is_identical_particle:
            (sp1, _mult), = reactant_items
            sp2 = sp1
            delta_ij = 1
        else:
            (sp1, _m1), (sp2, _m2) = reactant_items
            delta_ij = 0

        idx1 = species_index.get(sp1)
        idx2 = species_index.get(sp2)
        if idx1 is None or idx2 is None:
            continue

        sigma_v = float(rxn.rate_func(T9))
        if sigma_v <= 0.0:
            continue

        n1 = density * float(Y_vec[idx1])
        n2 = density * float(Y_vec[idx2])

        rate = n1 * n2 * sigma_v / (1 + delta_ij)  # reactions cm^-3 s^-1
        if rate <= 0.0:
            continue

        dn_dt = rate  # per reaction event consuming/producing at stoich mult below

        for sp, mult in rxn.reactants_stoich.items():
            idx = species_index.get(sp)
            if idx is not None:
                dYdt[idx] -= mult * dn_dt / density

        for sp, mult in rxn.products_stoich.items():
            idx = species_index.get(sp)
            if idx is not None:
                dYdt[idx] += mult * dn_dt / density

    return dYdt
