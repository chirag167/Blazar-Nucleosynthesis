"""
cascade.py

Orchestrates the non-thermal cascade using the already-defined modules:

- state.py      : NetworkState / CascadeState / ProjectileSpectrum
- stopping.py   : stopping_power_bin_average
- survival.py   : build_survival_and_yield
- reactions.py  : ReactionLibrary / ReactionChannel helpers

This file does NOT reimplement stopping power, survival fractions, or
cross-section interpolation. It only coordinates those modules.

Current design notes
--------------------
1. This module treats each stored projectile spectrum as an edge-defined-bin
   spectrum whose `energy_MeV` field contains the bin centers. The actual
   energy-bin edges are supplied explicitly via the `energy_edges_mev` argument.

2. Cloud target number densities are currently constructed as:
       n_target_i = cloud.density_cm3 * Y_i
   because CloudState stores abundances Y_i and one scalar density_cm3.
   If you later decide on a different abundance-to-number-density conversion,
   only `_cloud_target_number_densities()` needs to change.

3. Product-energy distributions are taken from each reaction channel's
   `product_distribution_model`. If that model is still a stub
   (e.g. DWBAStubModel), this module raises NotImplementedError rather than
   silently inventing a distribution.

4. Heavy / thermalized products are added directly to the cloud abundance RHS.
   Non-thermal descendants are injected into cascade spectra.

5. This module computes one cascade bookkeeping pass and stores the results in
   `state.cascade.rhs_cache`. A helper is provided to apply one explicit Euler
   abundance update if desired, but timestep selection remains in timestep.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

import numpy as np

from state import NetworkState, ProjectileSpectrum
from stopping import stopping_power_bin_average
from survival import build_survival_and_yield
from reactions import ReactionLibrary, canonical_species_name


Array = np.ndarray


@dataclass
class CascadeStepResult:
    """
    Bookkeeping produced by one cascade pass.
    """
    dYdt_cloud: np.ndarray
    injected_spectra: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any]


def _require_species_data(state: NetworkState, species: str) -> Tuple[int, int]:
    """
    Return (A, Z) for a species from state.species_data.
    """
    try:
        sp = state.species_data[species]
    except KeyError as exc:
        raise KeyError(
            f"Species '{species}' is missing from state.species_data."
        ) from exc
    return int(sp.A), int(sp.Z)


def _cloud_mass_fraction_arrays(state: NetworkState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return cloud arrays aligned with state.cloud.species:
        A_cl, Z_cl, X_cl
    where X_cl are mass fractions.
    """
    X_dict = state.get_mass_fractions()

    species = state.cloud.species
    A_cl = np.asarray([state.species_data[s].A for s in species], dtype=float)
    Z_cl = np.asarray([state.species_data[s].Z for s in species], dtype=float)
    X_cl = np.asarray([X_dict[s] for s in species], dtype=float)
    return A_cl, Z_cl, X_cl


def _cloud_target_number_densities(state: NetworkState) -> Dict[str, float]:
    """
    Construct target number densities from the current cloud state.

    Current convention:
        n_target_i = density_cm3 * Y_i

    This is the only place that assumption lives.
    """
    return {
        sp: float(state.cloud.density_cm3 * y)
        for sp, y in zip(state.cloud.species, state.cloud.Y)
    }


def _electron_number_density(state: NetworkState) -> float:
    """
    Crude electron-density helper from the currently available state variables.

    Current convention:
        n_e = density_cm3 * ionization_fraction * sum_i(Z_i Y_i)

    If you later adopt a different cloud-density normalization, update this
    helper only.
    """
    total = 0.0
    for sp, y in zip(state.cloud.species, state.cloud.Y):
        Z = float(state.species_data[sp].Z)
        total += Z * float(y)

    return float(state.cloud.density_cm3 * state.cloud.ionization_fraction * total)


def _ensure_cloud_species_index(state: NetworkState, species: str) -> Optional[int]:
    """
    Return the cloud-species index if the species is tracked in CloudState.
    Otherwise return None.
    """
    try:
        return state.cloud.species_index(species)
    except KeyError:
        return None


def _all_product_species_for_reactions(reactions_for_projectile) -> List[str]:
    """
    Stable union of all product species appearing in the provided reactions.
    """
    seen = set()
    ordered: List[str] = []

    for rxn in reactions_for_projectile:
        for prod in rxn.products_as_objects():
            s = canonical_species_name(prod.species)
            if s not in seen:
                seen.add(s)
                ordered.append(s)

    return ordered


def _build_tau_tensor(
    reactions_for_projectile,
    product_species_order: Sequence[str],
    energy_edges_mev: np.ndarray,
) -> np.ndarray:
    """
    Build the tau tensor expected by survival.build_survival_and_yield():

        tau.shape = (n_rxn, n_proj_bins, n_products, n_prod_bins)

    Each reaction channel provides a product_distribution_model. The returned
    distribution is multiplied by the product multiplicity, so tau carries the
    expected number of particles of each product species per reaction event.
    """
    n_rxn = len(reactions_for_projectile)
    n_proj_bins = len(energy_edges_mev) - 1
    n_products = len(product_species_order)
    n_prod_bins = n_proj_bins

    tau = np.zeros((n_rxn, n_proj_bins, n_products, n_prod_bins), dtype=float)
    proj_bin_centers = 0.5 * (energy_edges_mev[:-1] + energy_edges_mev[1:])

    prod_index = {s: i for i, s in enumerate(product_species_order)}

    for i_rxn, rxn in enumerate(reactions_for_projectile):
        if rxn.product_distribution_model is None:
            raise NotImplementedError(
                f"Reaction '{rxn.name()}' does not have a product_distribution_model."
            )

        for k, eproj in enumerate(proj_bin_centers):
            for prod in rxn.products_as_objects():
                p_species = canonical_species_name(prod.species)
                p_idx = prod_index[p_species]

                try:
                    dist = rxn.product_distribution_model.distribution(
                        projectile_energy_mev=float(eproj),
                        product_species=p_species,
                        product_energy_edges_mev=energy_edges_mev,
                    )
                except NotImplementedError as exc:
                    raise NotImplementedError(
                        f"Reaction '{rxn.name()}' still uses an unimplemented "
                        f"product-energy distribution for product '{p_species}'."
                    ) from exc

                dist = np.asarray(dist, dtype=float)
                if dist.shape != (n_prod_bins,):
                    raise ValueError(
                        f"Distribution for reaction '{rxn.name()}', product '{p_species}' "
                        f"has shape {dist.shape}; expected {(n_prod_bins,)}."
                    )

                tau[i_rxn, k, p_idx, :] += float(prod.multiplicity) * dist

    return tau


def _reaction_target_density_vector(
    reactions_for_projectile,
    cloud_target_number_densities: Mapping[str, float],
) -> np.ndarray:
    """
    Return target densities aligned with the reaction ordering.
    """
    return np.asarray(
        [float(cloud_target_number_densities.get(rxn.target, 0.0)) for rxn in reactions_for_projectile],
        dtype=float,
    )


_MB_TO_CM2 = 1.0e-27  # 1 millibarns = 1e-27 cm^2


def _reaction_sigma_matrix(
    reactions_for_projectile,
    energy_edges_mev: np.ndarray,
) -> np.ndarray:
    """
    Return sigma_bin with shape (n_rxn, n_bins) in cm², aligned with reaction ordering.

    Cross sections are fetched in mb from each ReactionChannel and converted to cm²
    here because survival.py expects sigma in cm² so that Lambda = N * sigma has
    units of cm^{-1} and the exponent Lambda * dE / epsilon is dimensionless.
    """
    if not reactions_for_projectile:
        return np.zeros((0, len(energy_edges_mev) - 1), dtype=float)

    rows = [np.asarray(rxn.sigma_bin_average_mb(energy_edges_mev), dtype=float) * _MB_TO_CM2
            for rxn in reactions_for_projectile]
    return np.vstack(rows)


def _accumulate_products(
    state: NetworkState,
    reactions_for_projectile,
    product_species_order: Sequence[str],
    yield_tensor: np.ndarray,
    injected_amount: float,
    injected_spectra: Dict[str, np.ndarray],
    dYdt_cloud: np.ndarray,
    dt_s: float,
) -> None:
    """
    Accumulate product source terms from one injected projectile bin.

    Non-thermal descendants are injected into spectra.
    Thermal / heavy descendants are added directly to the cloud abundance RHS.
    """
    n_rxn, _, n_products, n_prod_bins = yield_tensor.shape

    for i_rxn, rxn in enumerate(reactions_for_projectile):
        # Target destruction contribution to cloud abundances.
        # Total number of reaction events from this injected projectile is the
        # integrated yield over all products/bins divided in practice by the
        # channel multiplicity structure. The robust quantity available here is
        # the per-reaction destruction probability:
        #   sum_k beta_i(k) DeltaS(k)
        #
        # We reconstruct it from the yield tensor by summing and dividing by the
        # total multiplicity carried by tau for species that appear in this rxn.
        rxn_events = 0.0
        for p_idx in range(n_products):
            total_for_product = float(np.sum(yield_tensor[i_rxn, :, p_idx, :]))
            if total_for_product > 0.0:
                rxn_events = max(rxn_events, total_for_product)

        rxn_events *= injected_amount

        t_idx = _ensure_cloud_species_index(state, rxn.target)
        if t_idx is not None and dt_s > 0.0:
            dYdt_cloud[t_idx] -= rxn_events / dt_s

        # Product creation contribution.
        for p_idx, p_species in enumerate(product_species_order):
            produced = injected_amount * np.sum(yield_tensor[i_rxn, :, p_idx, :])
            if produced <= 0.0:
                continue

            prod_obj = None
            for obj in rxn.products_as_objects():
                if canonical_species_name(obj.species) == p_species:
                    prod_obj = obj
                    break

            if prod_obj is None:
                continue

            if prod_obj.can_continue_nonthermal:
                injected_spectra.setdefault(p_species, np.zeros(n_prod_bins, dtype=float))
                injected_spectra[p_species] += injected_amount * np.sum(
                    yield_tensor[i_rxn, :, p_idx, :],
                    axis=0,
                )
            else:
                c_idx = _ensure_cloud_species_index(state, p_species)
                if c_idx is not None and dt_s > 0.0:
                    dYdt_cloud[c_idx] += produced / dt_s


def _projectile_survival_reinjection(
    spectrum_values: np.ndarray,
    survival_table: np.ndarray,
    initial_bin: int,
    destination: np.ndarray,
) -> None:
    """
    Reinject surviving projectiles into lower bins.

    For an amount N injected into bin n, cumulative survival S[k] gives the
    fraction surviving down to bin k. We distribute surviving particles across
    bins 1 through initial_bin.

    Bin 0 is NOT populated: particles that thermalize (reach below-grid energies)
    are handled separately via s_thermalized in the calling cascade loop.
    """
    N0 = float(spectrum_values[initial_bin])
    if N0 <= 0.0:
        return

    n_bins = len(survival_table)
    into_bin = np.zeros(n_bins, dtype=float)

    # Bins 1..n-1: fraction stopped in each bin
    for k in range(1, n_bins):
        into_bin[k] = max(survival_table[k] - survival_table[k - 1], 0.0)
    # Bin 0 is left at 0: thermalizing particles are added to cloud by the
    # caller using s_thermalized (from build_survival_and_yield).

    destination += N0 * into_bin


def compute_cascade_step(
    state: NetworkState,
    reaction_library: ReactionLibrary,
    energy_edges_mev: np.ndarray,
    dt_s: Optional[float] = None,
) -> CascadeStepResult:
    """
    Compute one cascade bookkeeping pass.

    Parameters
    ----------
    state
        Current full network state.
    reaction_library
        Reaction library containing the non-thermal channels.
    energy_edges_mev
        Common projectile/product energy bin edges.
    dt_s
        Timestep used only to convert event counts into cloud-abundance rates.
        If omitted, state.solver.dt_s is used.

    Returns
    -------
    CascadeStepResult
        Contains:
        - dYdt_cloud
        - injected_spectra
        - diagnostics

    Notes
    -----
    This function does not itself choose dt and does not advance state.solver.
    """
    state.validate()

    if dt_s is None:
        dt_s = state.solver.dt_s

    if dt_s is None or dt_s <= 0.0:
        raise ValueError("compute_cascade_step requires a positive dt_s.")

    energy_edges_mev = np.asarray(energy_edges_mev, dtype=float)
    n_bins = len(energy_edges_mev) - 1
    if n_bins <= 0:
        raise ValueError("energy_edges_mev must define at least one bin.")

    # Cloud composition / medium inputs
    A_cl, Z_cl, X_cl = _cloud_mass_fraction_arrays(state)
    n_e = _electron_number_density(state)
    target_number_densities = _cloud_target_number_densities(state)

    dYdt_cloud = np.zeros_like(state.cloud.Y, dtype=float)
    injected_spectra: Dict[str, np.ndarray] = {}
    diag: Dict[str, Any] = {
        "projectiles": {},
    }

    for projectile in state.cascade.projectile_species():
        spectrum = state.cascade.get_spectrum(projectile)

        if spectrum.n_bins != n_bins:
            raise ValueError(
                f"Spectrum for projectile '{projectile}' has {spectrum.n_bins} bins, "
                f"but energy_edges_mev defines {n_bins} bins."
            )

        A_proj, Z_proj = _require_species_data(state, projectile)

        reactions_for_projectile = reaction_library.by_projectile(projectile)
        if not reactions_for_projectile:
            # No reactions: just carry the spectrum forward unchanged.
            injected_spectra.setdefault(projectile, np.zeros(n_bins, dtype=float))
            injected_spectra[projectile] += spectrum.values
            diag["projectiles"][projectile] = {
                "n_reactions": 0,
                "status": "no_reactions",
            }
            continue

        sigma_bin = _reaction_sigma_matrix(reactions_for_projectile, energy_edges_mev)
        target_dens_vec = _reaction_target_density_vector(
            reactions_for_projectile,
            target_number_densities,
        )

        epsilon_bin = stopping_power_bin_average(
            A_cl=A_cl,
            X_cl=X_cl,
            X_ion=float(state.cloud.ionization_fraction),
            Z_proj=Z_proj,
            E_edges=energy_edges_mev,
            A_proj=A_proj,
            n_e=n_e,
            T_e=float(state.cloud.temperature_K),
        )

        product_species_order = _all_product_species_for_reactions(reactions_for_projectile)
        tau = _build_tau_tensor(
            reactions_for_projectile=reactions_for_projectile,
            product_species_order=product_species_order,
            energy_edges_mev=energy_edges_mev,
        )

        projectile_out = np.zeros(n_bins, dtype=float)

        for initial_bin in range(n_bins - 1, -1, -1):
            amount = float(spectrum.values[initial_bin])
            if amount <= 0.0:
                continue

            sy = build_survival_and_yield(
                epsilon_bin=epsilon_bin,
                sigma_bin=sigma_bin,
                target_densities=target_dens_vec,
                tau=tau,
                E_edges=energy_edges_mev,
                initial_bin=initial_bin,
            )

            # Surviving projectile bookkeeping
            _projectile_survival_reinjection(
                spectrum_values=spectrum.values,
                survival_table=np.asarray(sy["S"], dtype=float),
                initial_bin=initial_bin,
                destination=projectile_out,
            )

            # Product bookkeeping
            _accumulate_products(
                state=state,
                reactions_for_projectile=reactions_for_projectile,
                product_species_order=product_species_order,
                yield_tensor=np.asarray(sy["yield"], dtype=float),
                injected_amount=amount,
                injected_spectra=injected_spectra,
                dYdt_cloud=dYdt_cloud,
                dt_s=float(dt_s),
            )

        injected_spectra.setdefault(projectile, np.zeros(n_bins, dtype=float))
        injected_spectra[projectile] += projectile_out

        diag["projectiles"][projectile] = {
            "n_reactions": len(reactions_for_projectile),
            "reaction_names": [rxn.name() for rxn in reactions_for_projectile],
            "product_species": list(product_species_order),
            "epsilon_bin": epsilon_bin.copy(),
            "sigma_bin": sigma_bin.copy(),
            "target_densities": target_dens_vec.copy(),
        }

    return CascadeStepResult(
        dYdt_cloud=dYdt_cloud,
        injected_spectra=injected_spectra,
        diagnostics=diag,
    )


def store_cascade_step_in_state(
    state: NetworkState,
    step_result: CascadeStepResult,
) -> None:
    """
    Store a computed cascade step into state.cascade.rhs_cache without yet
    mutating cloud abundances or replacing spectra.
    """
    state.cascade.rhs_cache["dYdt_cloud"] = np.asarray(step_result.dYdt_cloud, dtype=float)
    state.cascade.rhs_cache["injected_spectra"] = {
        sp: np.asarray(vals, dtype=float)
        for sp, vals in step_result.injected_spectra.items()
    }
    state.cascade.rhs_cache["diagnostics"] = step_result.diagnostics


def replace_cascade_spectra_from_step(
    state: NetworkState,
    step_result: CascadeStepResult,
) -> None:
    """
    Replace the cascade spectra using the injected_spectra bookkeeping.
    """
    new_spectra: Dict[str, ProjectileSpectrum] = {}

    for species, values in step_result.injected_spectra.items():
        if species in state.cascade.spectra:
            energy = state.cascade.spectra[species].energy_MeV.copy()
        else:
            # Reuse the first existing spectrum grid as the common grid.
            if not state.cascade.spectra:
                raise ValueError("No existing cascade spectra found to infer the common energy grid.")
            any_spec = next(iter(state.cascade.spectra.values()))
            energy = any_spec.energy_MeV.copy()

        new_spectra[species] = ProjectileSpectrum(
            species=species,
            energy_MeV=energy,
            values=np.asarray(values, dtype=float),
        )

    state.cascade.spectra = new_spectra


def apply_cloud_euler_step_from_cache(state: NetworkState, dt_s: Optional[float] = None) -> None:
    """
    Apply an explicit Euler abundance update using the cached dY/dt produced by
    compute_cascade_step() / store_cascade_step_in_state().
    """
    if dt_s is None:
        dt_s = state.solver.dt_s

    if dt_s is None or dt_s <= 0.0:
        raise ValueError("apply_cloud_euler_step_from_cache requires a positive dt_s.")

    if "dYdt_cloud" not in state.cascade.rhs_cache:
        raise KeyError("state.cascade.rhs_cache does not contain 'dYdt_cloud'.")

    dydt = np.asarray(state.cascade.rhs_cache["dYdt_cloud"], dtype=float)
    if dydt.shape != state.cloud.Y.shape:
        raise ValueError(
            f"Cached dYdt_cloud has shape {dydt.shape}, expected {state.cloud.Y.shape}."
        )

    state.cloud.Y = state.cloud.Y + float(dt_s) * dydt
    state.cloud.clip_negative()


def run_cascade_step(
    state: NetworkState,
    reaction_library: ReactionLibrary,
    energy_edges_mev: np.ndarray,
    dt_s: Optional[float] = None,
    update_cloud: bool = True,
    update_spectra: bool = True,
) -> CascadeStepResult:
    """
    High-level convenience wrapper:
    1. compute cascade bookkeeping
    2. store in rhs_cache
    3. optionally update cloud abundances
    4. optionally replace cascade spectra
    """
    result = compute_cascade_step(
        state=state,
        reaction_library=reaction_library,
        energy_edges_mev=energy_edges_mev,
        dt_s=dt_s,
    )

    store_cascade_step_in_state(state, result)

    if update_cloud:
        apply_cloud_euler_step_from_cache(state, dt_s=dt_s)

    if update_spectra:
        replace_cascade_spectra_from_step(state, result)

    return result