"""
run_famiano.py

End-to-end driver script for reproducing Famiano (2002) blazar jet
nucleosynthesis results.

Usage
-----
    cd /path/to/Blazar-Nucleosynthesis
    python scripts/run_famiano.py [--config config/run.json]

The script reads cloud, jet, species, and run configuration from the
config/ directory, builds the initial NetworkState, loads the reaction
library from data/CrossSections/, and advances the cascade using
explicit Euler steps with Famiano's adaptive timestep.

Outputs are written to the outputs/ directory as configured in run.json.

Notes
-----
- This is the Famiano (2002) reproduction mode: the cloud composition,
  jet spectrum, and stopping power formula all follow the simplest
  assumptions in that paper. See config/cloud.json and config/jet.json
  for the parameters — VERIFY them against the paper before trusting
  numerical results.
- For Group 1 reactions, the product energy distribution uses the
  DWBAStubModel, which raises NotImplementedError. Until real DWBA
  output is provided, set use_group1=False or the cascade step will
  raise an error when Group 1 reactions are encountered.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on the path regardless of working directory.
# _ROOT must come before _ROOT/core so that `from utils.utils import ...`
# inside core/ resolves utils as the package at _ROOT/utils/, not a bare module.
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "core"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.tau_loader import assign_tau_models_to_library, build_tau_models
from state import CloudState, CascadeState, SolverState, NetworkState, SpeciesData, ProjectileSpectrum
from grids import make_energy_grid
from reactions import ReactionLibrary, DWBAStubModel, DeltaAtThermalModel
from tau_loader import build_tau_models, assign_tau_models_to_library
from cascade import run_cascade_step, _cloud_mass_fraction_arrays, _electron_number_density
from stopping import stopping_power
from timestep import compute_next_dt, estimate_initial_dt, euler_increment
from decay import apply_decay, decay_dydt, DECAY_TABLE
from thermonuclear import compute_thermonuclear_dydt
from rate_library import build_thermonuclear_library
from jacobian import robust_implicit_euler_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_famiano")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _jet_mass_rate_msun_per_yr(jet_cfg: dict) -> float:
    """
    Jet mass input rate M_dot [M_sun/yr] for the model named by
    jet.json's mass_rate.use (e.g. "model_A" -> model_A_Msun_per_yr).
    Falls back to Model B's rate when 'use' is absent, which was the
    hard-coded behaviour before the selector existed.
    """
    mass_rate_cfg = jet_cfg.get("mass_rate", {})
    model = mass_rate_cfg.get("use", "model_B")
    key = f"{model}_Msun_per_yr"
    if key not in mass_rate_cfg:
        raise KeyError(f"jet.json mass_rate.use = '{model}' but '{key}' is not defined.")
    return float(mass_rate_cfg[key])


# Lower bound on the n-p elastic cross section over ~1-400 MeV (~30 mb near
# a few hundred MeV), and the number of collisions taken to thermalize a
# neutron. Used only to decide whether jet-produced neutrons escape.
_SIGMA_NP_MIN_CM2 = 3.0e-26
_N_COLLISIONS_TO_THERMALIZE = 20.0


def light_product_stopped_mask(
    state: NetworkState, species: str, energy_edges: np.ndarray, L_cm: float,
) -> np.ndarray:
    """
    Per-energy-bin 0/1 mask: 1 where a nonthermal product of `species`
    created at that bin's energy stops inside a cloud of length L_cm.

    Charged particles: CSDA range R(E) = sum over bins up to E of
    dE / stopping_power(E), using the same medium model as the cascade.
    Neutrons: no Coulomb stopping, so they are counted as stopped (in every
    bin) only if even the smallest n-p elastic cross section gives at least
    _N_COLLISIONS_TO_THERMALIZE collisions across the cloud; otherwise they
    all escape. The two regimes are far apart for Famiano's models (Model A:
    ~1e3 collisions; Model B: < 1).
    """
    sd = state.species_data[species]
    n_bins = len(energy_edges) - 1
    if sd.Z == 0:
        i_p = state.cloud.species.index("p") if "p" in state.cloud.species else None
        n_H = state.cloud.density_cm3 * (state.cloud.Y[i_p] if i_p is not None else 0.0)
        stops = L_cm * n_H * _SIGMA_NP_MIN_CM2 >= _N_COLLISIONS_TO_THERMALIZE
        return np.full(n_bins, 1.0 if stops else 0.0)

    A_cl, _, X_cl = _cloud_mass_fraction_arrays(state)
    E_c = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    eps = stopping_power(
        A_cl=A_cl, X_cl=X_cl, X_ion=float(state.cloud.ionization_fraction),
        Z_proj=int(sd.Z), E=E_c, A_proj=int(sd.A),
        n_e=_electron_number_density(state), T_e=float(state.cloud.temperature_K),
    )
    range_cm = np.cumsum(np.diff(energy_edges) / np.maximum(eps, 1e-300))
    return (range_cm <= L_cm).astype(float)


def _load_configs(config_dir: Path, run_cfg_path: Path) -> tuple[dict, dict, dict, dict]:
    cloud_cfg  = _load_json(config_dir / "cloud.json")
    jet_cfg    = _load_json(config_dir / "jet.json")
    species_cfg = _load_json(config_dir / "species.json")
    run_cfg    = _load_json(run_cfg_path)
    return cloud_cfg, jet_cfg, species_cfg, run_cfg


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def build_species_data(species_cfg: dict) -> dict[str, SpeciesData]:
    return {
        name: SpeciesData(name=name, A=int(v["A"]), Z=int(v["Z"]))
        for name, v in species_cfg.items()
        if not name.startswith("_")
    }


def build_cloud_state(cloud_cfg: dict) -> CloudState:
    abundances = {k: v for k, v in cloud_cfg["abundances"].items() if not k.startswith("_")}
    species = list(abundances.keys())
    Y = np.array([abundances[s] for s in species], dtype=float)
    return CloudState(
        species=species,
        Y=Y,
        density_cm3=float(cloud_cfg["density_cm3"]),
        temperature_K=float(cloud_cfg["temperature_K"]),
        ionization_fraction=float(cloud_cfg["ionization_fraction"]),
    )


def build_initial_spectra(
    jet_cfg: dict,
    energy_edges: np.ndarray,
    species_data: dict,
) -> dict[str, ProjectileSpectrum]:
    """
    Build initial non-thermal projectile spectra from jet config.

    Supports:
        spectrum.type = "monoenergetic"  (Famiano 2002 default)
            Particles are injected at a single energy per nucleon.
            E_i = E_per_nucleon_MeV * A_i.
            The particle count is placed into the nearest energy grid bin.

        spectrum.type = "power_law"
            dN/dE proportional to E^{-s}  (for future sensitivity studies)
    """
    spec_cfg = jet_cfg["spectrum"]
    species_list = [s for s in jet_cfg["species"] if not s.startswith("_")]
    comp = {k: float(v) for k, v in jet_cfg.get("composition_number_fractions", {}).items()
            if not k.startswith("_")}

    bin_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    n_bins = len(bin_centers)
    norm = float(spec_cfg.get("normalization", 1.0))
    spectra = {}

    if spec_cfg["type"] == "monoenergetic":
        # Famiano (2002): monoenergetic injection at E = E_per_nucleon * A
        E_per_nuc = float(spec_cfg["E_per_nucleon_MeV"])

        for sp in species_list:
            A = species_data[sp].A if sp in species_data else 1
            E_inj = E_per_nuc * A

            values = np.zeros(n_bins, dtype=float)
            idx = int(np.argmin(np.abs(bin_centers - E_inj)))
            frac = comp.get(sp, 1.0)
            values[idx] = norm * frac
            if abs(bin_centers[idx] - E_inj) > 0.5 * (energy_edges[1] - energy_edges[0]):
                log.warning(
                    "Species %s: injection energy %.1f MeV not covered by grid "
                    "(nearest bin centre is %.1f MeV).", sp, E_inj, bin_centers[idx]
                )

            spectra[sp] = ProjectileSpectrum(
                species=sp,
                energy_MeV=bin_centers.copy(),
                values=values,
            )

    elif spec_cfg["type"] == "power_law":
        s_idx = float(spec_cfg["spectral_index"])
        E_min = float(spec_cfg["E_min_MeV"])
        E_max = float(spec_cfg["E_max_MeV"])

        mask = (bin_centers >= E_min) & (bin_centers <= E_max)
        base_spectrum = np.where(mask, norm * bin_centers ** (-s_idx), 0.0)

        for sp in species_list:
            frac = comp.get(sp, 1.0)
            spectra[sp] = ProjectileSpectrum(
                species=sp,
                energy_MeV=bin_centers.copy(),
                values=(frac * base_spectrum).copy(),
            )

    else:
        raise ValueError(f"Unsupported jet spectrum type '{spec_cfg['type']}'.")

    return spectra


def build_network_state(
    cloud_cfg: dict,
    jet_cfg: dict,
    species_cfg: dict,
    run_cfg: dict,
    energy_edges: np.ndarray,
) -> NetworkState:
    species_data = build_species_data(species_cfg)
    cloud = build_cloud_state(cloud_cfg)
    spectra = build_initial_spectra(jet_cfg, energy_edges, species_data)

    cascade = CascadeState(spectra=spectra)

    time_cfg = run_cfg["time"]
    solver = SolverState(
        t_s=float(time_cfg["t0_s"]),
        t_end_s=float(time_cfg["tmax_s"]),
        dt_s=float(time_cfg["dt_s"]),
    )

    state = NetworkState(
        cloud=cloud,
        cascade=cascade,
        solver=solver,
        species_data=species_data,
    )
    state.validate()
    return state


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_history_row(
    fh,
    writer,
    state: NetworkState,
    t_s: float,
    step: int,
    delta_m_over_m0: float = 0.0,
) -> None:
    row = {"step": step, "t_s": t_s, "delta_m_over_m0": delta_m_over_m0}
    row.update({sp: float(y) for sp, y in zip(state.cloud.species, state.cloud.Y)})
    writer.writerow(row)
    fh.flush()


# ---------------------------------------------------------------------------
# Cloud species expansion
# ---------------------------------------------------------------------------

def expand_cloud_with_reaction_products(
    state: NetworkState,
    lib,
) -> None:
    """
    Pre-populate cloud.species with every species any reaction can produce,
    seeded at Y = 0.  Without this, cascade.py silently discards dY/dt
    contributions for species that have no cloud slot.
    """
    all_products: set[str] = set()
    for rxn in lib.reactions:
        all_products.update(rxn.products_stoich.keys())

    added = []
    for sp in sorted(all_products):
        if sp in state.cloud.species:
            continue
        if sp not in state.species_data:
            log.warning(
                "Product species '%s' not in species_data — skipping cloud expansion for it.", sp
            )
            continue
        state.cloud.species.append(sp)
        state.cloud.Y = np.append(state.cloud.Y, 0.0)
        added.append(sp)

    if added:
        log.info("Expanded cloud species with %d reaction products (Y=0): %s", len(added), added)


# ---------------------------------------------------------------------------
# Jet normalization
# ---------------------------------------------------------------------------

_MSUN_G    = 1.989e33          # g
_AMU_G     = 1.66053906660e-24 # g
_C_CM_S    = 2.99792458e10     # cm/s
_S_PER_YR  = 3.15576e7         # s
_M_P_MEV   = 938.272           # MeV/c²


def compute_jet_normalization_factor(
    jet_cfg: dict,
    cloud_cfg: dict,
    species_cfg: dict,
) -> float:
    """
    Compute the dimensionless ratio  n_jet_total / n_baryon_cloud.

    This is the physical normalization for ProjectileSpectrum.values so that
    cascade reaction rates have correct units (abundance change per baryon per
    second).

    Derivation
    ----------
    The steady-state number density of jet particles in the cloud is

        n_jet = Φ / v_jet

    where Φ [cm⁻² s⁻¹] is the particle flux through the cloud face and
    v_jet = β c is the jet particle speed (same β for all species at the same
    MeV/nucleon).

        Φ = (Ṁ / m_avg) / A_cloud

    with  Ṁ  the jet mass rate [g/s],  m_avg  the mean jet particle mass [g],
    and  A_cloud = π (d/2)²  the cloud cross-section area [cm²].

    The normalization factor is then

        f_norm = n_jet / n_cloud

    and each species' injection bin value is set to  f_norm × number_fraction.
    """
    import math

    # --- Jet composition and mean particle mass ---
    species_list = [s for s in jet_cfg["species"] if not s.startswith("_")]
    comp = {k: float(v) for k, v in jet_cfg.get("composition_number_fractions", {}).items()
            if not k.startswith("_")}
    total_frac = sum(comp.get(sp, 1.0) for sp in species_list)

    m_avg_u = sum(
        (comp.get(sp, 1.0) / total_frac) * int(species_cfg.get(sp, {}).get("A", 1))
        for sp in species_list
    )
    m_avg_g = m_avg_u * _AMU_G

    # --- Mass rate [g/s] ---
    mass_rate_Msun_yr = _jet_mass_rate_msun_per_yr(jet_cfg)
    mass_rate_g_s = mass_rate_Msun_yr * _MSUN_G / _S_PER_YR

    # --- Cloud cross-section area [cm²] ---
    cloud_diameter_cm = float(cloud_cfg.get("cloud_diameter_cm", 5.0e11))
    A_cloud_cm2 = math.pi * (cloud_diameter_cm / 2.0) ** 2

    # --- Particle flux [cm⁻² s⁻¹] ---
    phi = (mass_rate_g_s / m_avg_g) / A_cloud_cm2

    # --- Jet speed: same β for all species at same MeV/nucleon ---
    E_per_nuc = float(jet_cfg["spectrum"]["E_per_nucleon_MeV"])
    gamma_rel = 1.0 + E_per_nuc / _M_P_MEV
    beta = math.sqrt(1.0 - 1.0 / gamma_rel ** 2)
    v_jet = beta * _C_CM_S

    # --- Steady-state jet number density [cm⁻³] ---
    n_jet = phi / v_jet

    # --- Cloud baryon number density [cm⁻³] ---
    n_cloud = float(cloud_cfg["density_cm3"])

    f_norm = n_jet / n_cloud

    log.info(
        "Jet normalization: Ṁ = %.2e g/s | Φ = %.2e cm⁻²s⁻¹ | β = %.4f"
        " | n_jet = %.2e cm⁻³ | n_cloud = %.2e cm⁻³ | f_norm = %.3e",
        mass_rate_g_s, phi, beta, n_jet, n_cloud, f_norm,
    )
    return f_norm


# ---------------------------------------------------------------------------
# Cloud initial mass (for ΔM/M₀ tracking)
# ---------------------------------------------------------------------------

def compute_cloud_initial_mass_g(cloud_cfg: dict, state: NetworkState) -> float:
    """
    Compute the initial cloud mass [g] from geometry and baryon inventory.

        M₀ = n_cloud × V_cloud × m_u × <A>

    where <A> = Σ Y_i A_i is the mean baryon mass number at t = 0.
    """
    import math

    n_cloud = float(cloud_cfg["density_cm3"])
    cloud_length_cm   = float(cloud_cfg.get("cloud_length_cm",   5.0e11))
    cloud_diameter_cm = float(cloud_cfg.get("cloud_diameter_cm", 5.0e11))
    V_cloud = math.pi * (cloud_diameter_cm / 2.0) ** 2 * cloud_length_cm

    mean_A = sum(
        float(y) * float(state.species_data[sp].A)
        for sp, y in zip(state.cloud.species, state.cloud.Y)
        if sp in state.species_data
    )

    return n_cloud * V_cloud * mean_A * _AMU_G


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    config_dir = Path(args.config).resolve().parent
    run_cfg_path = Path(args.config).resolve()

    log.info("Loading configuration from %s", config_dir)
    cloud_cfg, jet_cfg, species_cfg, run_cfg = _load_configs(config_dir, run_cfg_path)

    # Energy grid
    grid = make_energy_grid(run_cfg["energy_grid"])
    energy_edges = grid.edges
    log.info("Energy grid: %d bins, %.2f–%.2f MeV", grid.n_bins, grid.e_min, grid.e_max)

    # Reaction library
    # Non-thermal projectiles include the initial jet species PLUS all A<8 secondary
    # products (d, t, n, 3He, 6Li, 7Li). Famiano Section 3: "only particles with A<8
    # are treated as energetic projectiles."
    _nonthermal_projectiles = ["p", "n", "d", "t", "3He", "4He", "6Li", "7Li"]

    log.info("Loading reaction library from %s/data/CrossSections/", _ROOT)
    lib = ReactionLibrary.from_directories(
        base_dir=_ROOT,
        projectile_species=_nonthermal_projectiles,
        target_species=None,    # accept all targets found in the data files
        include_group1=True,   # Group1 requires DWBA product distributions — not yet implemented
        include_group2=True,
    )
    log.info("Loaded %d reaction channels.", len(lib.reactions))
    # load and assign tau models
    # Q-values (MeV) from atomic mass differences (NNDC).
    # Reactions with only FRESCO ejectile data need kinematic inference for
    # the heavy recoil: T_recoil = T_proj_center + Q - T_ejectile_center.
    _infer_recoil = {
        "he4_he4_n_be7": ("n",  "be7", -18.99),   # 4He+4He -> n+7Be
        "he4_he4_p_li7": ("p",  "li7", -17.35),   # 4He+4He -> p+7Li
        "he4_p_d_he3":   ("d",  "he3", -18.35),   # 4He+p   -> d+3He
    }
    models = build_tau_models(
        e_bins_root=_ROOT / "fresco_code/newruns",
        reactions=["he4_he4_n_be7", "he4_he4_p_li7", "he4_p_d_he3", "li7_p_he4_he4"],
        energy_edges_mev=grid.edges,
        product_species=["n", "p", "d", "t", "3He", "4He", "7Li", "7Be"],
        infer_recoil=_infer_recoil,
    )
    n_assigned = assign_tau_models_to_library(models, lib)
    log.info("TauProductModel assigned to %d Group 1 channels.", n_assigned)

    # Remaining Group 1 channels have no FRESCO kinematic output yet.
    # Fall back to DeltaAtThermalModel (all products deposit at the lowest
    # energy bin) so they don't raise NotImplementedError at runtime.
    n_fallback = 0
    for rxn in lib.by_group("group1"):
        if isinstance(rxn.product_distribution_model, DWBAStubModel):
            rxn.product_distribution_model = DeltaAtThermalModel()
            n_fallback += 1
    if n_fallback:
        log.info(
            "%d Group 1 channel(s) using DeltaAtThermalModel fallback "
            "(no FRESCO kinematic data).", n_fallback
        )

    # Thermonuclear reaction library (Famiano Section 3: relevant when the
    # cloud is dense/hot enough for pair reactions among cloud species
    # themselves, e.g. Model A/C -- negligible for the cold, low-density
    # Model B BLR cloud). Off by default so existing Model B configs are
    # unaffected; enable via run.json's "thermonuclear.enabled".
    thermo_cfg = run_cfg.get("thermonuclear", {})
    thermonuclear_enabled = bool(thermo_cfg.get("enabled", False))
    thermo_lib = None
    if thermonuclear_enabled:
        thermo_lib, thermo_report = build_thermonuclear_library()
        log.info(
            "Loaded thermonuclear reaction library: %d reactions.\n%s",
            len(thermo_lib), thermo_report.summary(),
        )
    else:
        log.info("Thermonuclear reactions disabled (run.json 'thermonuclear.enabled').")

    # Initial state
    state = build_network_state(cloud_cfg, jet_cfg, species_cfg, run_cfg, energy_edges)

    # Expand cloud with all possible reaction products (Y=0) so that
    # dYdt_cloud has a valid slot for every species a reaction can produce.
    expand_cloud_with_reaction_products(state, lib)
    if thermonuclear_enabled:
        expand_cloud_with_reaction_products(state, thermo_lib)
    log.info("Cloud species after expansion: %s", state.cloud.species)

    # Rescale injection spectra to physical per-second injection rates.
    # spectrum.values[k] = (n_jet/n_cloud) × (v_jet/L_cloud)  [s⁻¹]
    # The cascade computes  dYdt = amount × yield / dt_s, so setting
    # amount = rate × dt_s gives the correct  dYdt = rate × yield  [s⁻¹].
    f_norm = compute_jet_normalization_factor(jet_cfg, cloud_cfg, species_cfg)
    _E_per_nuc_mev = float(jet_cfg["spectrum"]["E_per_nucleon_MeV"])
    _gamma_jet = 1.0 + _E_per_nuc_mev / _M_P_MEV
    _v_jet_cms = math.sqrt(1.0 - 1.0 / _gamma_jet ** 2) * _C_CM_S
    _L_cloud_cm = float(cloud_cfg.get("cloud_length_cm",
                        cloud_cfg.get("cloud_diameter_cm", 5.0e11)))
    _v_over_L = _v_jet_cms / _L_cloud_cm   # s⁻¹
    for spec in state.cascade.spectra.values():
        spec.values[:] *= f_norm * _v_over_L
    log.info(
        "Jet injection: f_norm=%.3e  v=%.3e cm/s  L=%.2e cm  "
        "v/L=%.3e s⁻¹  combined rate factor=%.3e s⁻¹",
        f_norm, _v_jet_cms, _L_cloud_cm, _v_over_L, f_norm * _v_over_L,
    )

    # Cloud mass and jet mass rate for ΔM/M₀ tracking.
    # M₀ is the initial cloud mass from geometry + density (not the total BLR mass).
    _M0_g = compute_cloud_initial_mass_g(cloud_cfg, state)
    _mass_rate_g_s = _jet_mass_rate_msun_per_yr(jet_cfg) * _MSUN_G / _S_PER_YR
    log.info(
        "Cloud mass M₀ = %.3e g (%.3e M_sun) | jet mass rate = %.3e g/s | "
        "ΔM/M₀ rate = %.3e s⁻¹",
        _M0_g, _M0_g / _MSUN_G, _mass_rate_g_s,
        _mass_rate_g_s / _M0_g,
    )

    # Store per-second injection rates (already scaled above).
    # At each step we set spectrum.values = rate × dt_s so the cascade formula
    # dYdt = (amount × yield) / dt_s reduces to the dt-independent rate × yield.
    _injection_rates: dict[str, np.ndarray] = {
        sp: spec.values.copy()
        for sp, spec in state.cascade.spectra.items()
    }

    # Cloud volume and jet mean-A for constant-volume density evolution (Famiano ρ̇ = K).
    _cloud_diameter_cm = float(cloud_cfg.get("cloud_diameter_cm", 5.0e11))
    _V_cloud_cm3 = math.pi * (_cloud_diameter_cm / 2.0) ** 2 * _L_cloud_cm
    _jet_species_list = [s for s in jet_cfg["species"] if not s.startswith("_")]
    _jet_comp = {k: float(v) for k, v in jet_cfg.get("composition_number_fractions", {}).items()
                 if not k.startswith("_")}
    _jet_total_frac = sum(_jet_comp.get(sp, 1.0) for sp in _jet_species_list)
    _mean_A_jet = sum(
        (_jet_comp.get(sp, 1.0) / _jet_total_frac) * float(species_cfg.get(sp, {}).get("A", 1))
        for sp in _jet_species_list
    )
    # Jet BARYON fractions for density-evolution mixing.
    # density_cm3 tracks baryon number density (n_baryons/cm³); Y_i = n_i/n_baryons,
    # so jet baryons must be added at baryon fractions: Y_i_jet = number_frac_i / mean_A_jet.
    _jet_baryon_fractions = {
        sp: (_jet_comp.get(sp, 1.0) / _jet_total_frac) / _mean_A_jet
        for sp in _jet_species_list
    }
    log.info(
        "Cloud volume = %.3e cm³ | mean jet A = %.3f | "
        "baryon density growth rate = %.3e cm⁻³ s⁻¹",
        _V_cloud_cm3, _mean_A_jet,
        _mass_rate_g_s / (_V_cloud_cm3 * _AMU_G),
    )

    # Output setup
    output_cfg = run_cfg.get("output", {})
    out_csv = _ROOT / output_cfg.get("history_csv", "outputs/abundance_history.csv")
    out_json = _ROOT / output_cfg.get("final_state_json", "outputs/final_state.json")
    write_every = int(output_cfg.get("write_every_n_steps", 100))
    # A jet on/off transition resets dt down to dt_min (see the transition
    # handling below), and the fast ramp that follows -- production
    # resuming, or a short-lived species climbing back to equilibrium --
    # typically completes within a couple hundred solver steps, long before
    # dt has regrown back to a "normal" size. At the coarse write_every_n_steps
    # cadence used for the rest of a run, that entire ramp can start and
    # finish inside the gap between two written rows, so the CSV shows a
    # multi-order-of-magnitude change between adjacent rows even though the
    # solver stepped through it smoothly -- the recording missed it, not the
    # physics. For this many steps after each transition, write every step
    # instead, then fall back to the normal cadence once things have settled.
    fine_write_steps = int(output_cfg.get("fine_write_steps_after_transition", 300))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    solver_cfg = run_cfg.get("solver", {})
    gamma    = float(solver_cfg.get("gamma", 0.01))
    dt_min   = float(solver_cfg.get("dt_min_s", 1e4))
    dt_max   = float(solver_cfg.get("dt_max_s", 1e11))
    # Species with Y below this are ignored by the timestep controller, so
    # physically irrelevant trace species (e.g. 8Li at ~1e-25) can't throttle
    # dt for the whole network.
    dt_rel_floor = float(solver_cfg.get("dt_rel_floor", 1e-30))
    max_steps = int(solver_cfg.get("max_steps", 100000))

    # Jet on/off intervals: injection (and jet mass loading) is only active
    # while t_s falls within one of these [on_s, off_s] pairs, letting the
    # jet be switched on and off multiple times over the run. The cascade
    # keeps running (secondary reactions + decay) at all times, until tmax_s.
    # Defaults to always-on (a single interval covering the whole run) if
    # not configured. A legacy scalar 'jet_stop_s' is still honored as
    # [[t0_s, jet_stop_s]] for backward compatibility.
    if "jet_intervals" in run_cfg["time"]:
        _jet_intervals = [
            (float(on_s), float(off_s)) for on_s, off_s in run_cfg["time"]["jet_intervals"]
        ]
    elif "jet_stop_s" in run_cfg["time"]:
        _jet_intervals = [(float(run_cfg["time"]["t0_s"]), float(run_cfg["time"]["jet_stop_s"]))]
    else:
        _jet_intervals = [(float(run_cfg["time"]["t0_s"]), state.solver.t_end_s)]

    def _is_jet_on(t_s: float) -> bool:
        return any(on_s <= t_s < off_s for on_s, off_s in _jet_intervals)

    # Sorted on/off edges, used to (a) stop a step from straddling a
    # transition and (b) detect when a step lands exactly on one.
    _jet_boundaries = sorted({b for on_s, off_s in _jet_intervals for b in (on_s, off_s)})

    def _next_jet_boundary(t_s: float) -> float | None:
        for b in _jet_boundaries:
            if b > t_s:
                return b
        return None

    _jet_was_on = _is_jet_on(state.solver.t_s)
    _fine_write_until_step = 0  # write every step through this step number

    decay_cfg = run_cfg.get("decay", {})
    decay_enabled = bool(decay_cfg.get("enabled", True))
    if decay_enabled:
        log.info(
            "Radioactive decay enabled for %d species: %s",
            len(DECAY_TABLE), sorted(DECAY_TABLE.keys()),
        )
    else:
        log.info("Radioactive decay disabled via config/run.json 'decay.enabled'.")

    # How jet-cloud products lighter than A=6 (n, d, t, 3He) are treated:
    # "escape" (default; the original thin-target assumption) or "range"
    # (stop in the cloud where their range is shorter than the cloud).
    _light_product_mode = run_cfg.get("cascade", {}).get("light_products", "escape")
    if _light_product_mode not in ("escape", "range"):
        raise ValueError(f"run.json cascade.light_products must be 'escape' or 'range', got {_light_product_mode!r}")
    if _light_product_mode == "range":
        for _sp in ("n", "d", "t", "3He"):
            if _sp in state.species_data:
                _frac = light_product_stopped_mask(state, _sp, energy_edges, _L_cloud_cm).mean()
                log.info("Light product %-3s: stops in cloud in %.0f%% of energy bins (L = %.2e cm).",
                         _sp, 100 * _frac, _L_cloud_cm)
    else:
        log.info("Light jet-cloud products (n, d, t, 3He) escape the cloud (cascade.light_products = 'escape').")

    import csv
    fieldnames = ["step", "t_s", "delta_m_over_m0"] + list(state.cloud.species)

    delta_m_over_m0 = 0.0   # cumulative ΔM/M₀
    _initial_density_cm3 = state.cloud.density_cm3  # for per-step injection correction
    _species_A = {
        sp: int(cfg["A"])
        for sp, cfg in species_cfg.items()
        if isinstance(cfg, dict) and "A" in cfg
    }
    # Mass numbers aligned with state.cloud.species, for crediting the
    # baryons the cascade moves around each step (see baryon-conservation
    # reconciliation in the main loop below).
    _A_cloud = np.array(
        [state.species_data[sp].A if sp in state.species_data else 1.0
         for sp in state.cloud.species],
        dtype=float,
    )
    _jet_sp_set = set(_injection_rates.keys())
    _thermo_path_counts: dict[str, int] = {}  # implicit-solve path tally between log lines

    t_start_wall = time.time()
    log.info("Starting evolution: t_end = %.3e s, max_steps = %d", state.solver.t_end_s, max_steps)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        _write_history_row(fh, writer, state, state.solver.t_s, 0, delta_m_over_m0)

        step = 0
        while not state.solver.done and step < max_steps:
            dt = state.solver.dt_s

            # --- Clamp dt so a single step never straddles a jet on/off
            # transition. Without this, a step whose dt was grown large
            # during a quiet (jet-off) stretch could run right past an
            # "on" boundary and apply the OLD jet state for part of a step
            # that should have seen the NEW one (or vice versa on an "off"
            # boundary) -- i.e. injection starting/stopping late or early.
            # Clamping forces every transition to fall exactly on a step
            # edge, so each step's _jet_on flag is correct for its full
            # duration. ---
            _next_boundary = _next_jet_boundary(state.solver.t_s)
            if _next_boundary is not None:
                _gap = _next_boundary - state.solver.t_s
                if _gap > 0.0:
                    dt = min(dt, _gap)

            # --- Jet on/off: only inject new jet particles while t_s falls in
            # one of _jet_intervals, while letting the cascade keep evolving
            # (secondary reactions + decay) at all times until t_end_s. ---
            _jet_on = _is_jet_on(state.solver.t_s)
            if _jet_on != _jet_was_on:
                log.info(
                    "Jet injection %s at t = %.4e s.",
                    "started" if _jet_on else "stopped", state.solver.t_s,
                )
                _jet_was_on = _jet_on
                # Sample every step for a while after ANY transition (both
                # directions) -- the fast ramp up when injection resumes
                # and the fast decay-down when it stops are both easy to
                # miss between two write_every_n_steps-spaced rows.
                _fine_write_until_step = step + fine_write_steps
                if _jet_on:
                    # Only reset on OFF->ON: reusing a stale dt (however
                    # large it happens to be) for the first step where
                    # injection switches on would apply the new source term
                    # over a step sized for the old, quieter dynamics -- an
                    # explicit-Euler overshoot that shows up as an unphysical
                    # jump. Forcing this one step back down to dt_min lets
                    # the controller re-measure the new dYdt honestly; it
                    # grows back to an efficient step size within a handful
                    # of steps once real activity resumes.
                    #
                    # Do NOT do this on ON->OFF: there is no overshoot risk
                    # in going quiet (decay uses the exact exponential
                    # solution regardless of dt, and the boundary clamp
                    # above already stops this step exactly at the edge).
                    # Worse, forcing dt down here is actively harmful: once
                    # injection stops, delta_y is ~0 every step, so
                    # compute_next_dt's "nothing changed" branch just holds
                    # dt wherever it last was -- nothing grows it back up.
                    # Crushing it to dt_min right as the quiet stretch
                    # begins would pin the solver at the smallest possible
                    # step for the entire OFF interval (potentially tens of
                    # thousands of steps to cross a gap that only needed a
                    # handful), rather than coasting at whatever dt the
                    # controller had already earned.
                    dt = min(dt, dt_min)

            # --- Set per-step injection: rate [s⁻¹] × dt ---
            # Famiano's model: jet continuously transits the cloud.
            # amount = rate × dt gives the correct dt-independent dYdt = rate × yield.
            # As cloud density grows the jet flux per baryon decreases:
            # f_norm = n_jet/n_cloud ∝ 1/density, so rescale by ρ₀/ρ each step.
            _rho_frac = _initial_density_cm3 / state.cloud.density_cm3
            for sp, rate_vals in _injection_rates.items():
                _amount = rate_vals * dt * _rho_frac if _jet_on else np.zeros_like(rate_vals)
                if sp in state.cascade.spectra:
                    state.cascade.spectra[sp].values[:] = _amount
                else:
                    from state import ProjectileSpectrum
                    state.cascade.spectra[sp] = ProjectileSpectrum(
                        species=sp,
                        energy_MeV=energy_edges[:-1] + 0.5 * np.diff(energy_edges),
                        values=_amount,
                    )

            # --- Cascade step ---
            result = run_cascade_step(
                state=state,
                reaction_library=lib,
                energy_edges_mev=energy_edges,
                dt_s=dt,
                update_cloud=False,   # we apply the Euler step manually below
                update_spectra=False, # steady-state: spectrum is reset each step
                cloud_length_cm=_L_cloud_cm,
            )

            # Thermalize non-thermal reaction products into Y_cloud.
            # Heavy recoils (A≥6: 6Li, 7Li, 7Be) always stop. Light products
            # (n, d, t, 3He) either all escape ("escape": the original thin-
            # target Model B assumption) or, with "range", stop in the energy
            # bins whose range is shorter than the cloud -- in Model A's
            # ~1e5 g/cm^2 cloud that is all of them. Products are deposited
            # directly; their own secondary reactions while slowing are not
            # followed (Famiano eq. 12 terms 3+), in either mode.
            # Jet species (p, 4He) are handled by the density-evolution step below.
            for _sp_nt, _nt_vals in result.injected_spectra.items():
                if _sp_nt in _jet_sp_set:
                    continue
                if _sp_nt not in state.cloud.species or dt <= 0.0:
                    continue
                if _species_A.get(_sp_nt, 0) < 6:
                    if _light_product_mode == "escape":
                        continue
                    _stopped = light_product_stopped_mask(state, _sp_nt, energy_edges, _L_cloud_cm)
                    _amount_th = float(np.dot(_nt_vals, _stopped))
                else:
                    _amount_th = float(np.sum(_nt_vals))
                _c_idx_nt = state.cloud.species.index(_sp_nt)
                result.dYdt_cloud[_c_idx_nt] += _amount_th / dt

            _Y_step_start = state.cloud.Y.copy()

            if not (thermonuclear_enabled and dt > 0.0):
                # --- Explicit Euler abundance update (jet-cloud reactions) ---
                # Used when the thermonuclear network is off (Model B): the
                # jet-cloud rates are slow compared with any dt the
                # controller picks, so explicit Euler is adequate.
                y_new, delta_y = euler_increment(
                    y=state.cloud.Y,
                    dydt=result.dYdt_cloud,
                    dt=dt,
                    enforce_nonnegative=True,
                )
                state.cloud.Y = y_new
            else:
                # --- Jet-cloud + thermonuclear + decay in ONE implicit solve ---
                # The jet-cloud changes enter the backward-Euler RHS instead of
                # being added explicitly beforehand: net production as a
                # constant source, net destruction of a cloud species as a
                # rate proportional to its abundance (so it can never drive it
                # negative). Adding them first (the old order) linearized the
                # thermonuclear solve around a state inflated by a whole
                # step's worth of jet products -- 34% more d and 30% more n
                # at dt = 2 s in the Model A run -- and the d+d / d+t terms
                # (quadratic in d) then gave dt-dependent balance values for
                # n, t and 8Li. That made large steps knock them down by up
                # to 40%, and the timestep controller kept throwing dt back to
                # ~0.02 s. With the source inside the solve, a steady state is
                # steady at any dt.
                #
                # Thermonuclear reactions and decay are solved implicitly
                # because they are stiff at Model A conditions (explicit
                # Euler clipped nearly every species at dt_min). Decay is in
                # the same RHS so short-lived species (8B, 8Li) reach their
                # production/decay balance for any dt.
                _S_casc = result.dYdt_cloud
                _src = np.where(_S_casc > 0.0, _S_casc, 0.0)
                _Y0 = state.cloud.Y
                _kappa = np.where((_S_casc < 0.0) & (_Y0 > 0.0), -_S_casc / np.where(_Y0 > 0.0, _Y0, 1.0), 0.0)

                def _thermo_rhs(Y_trial: np.ndarray) -> np.ndarray:
                    rhs = compute_thermonuclear_dydt(state, thermo_lib, Y=Y_trial)
                    if decay_enabled:
                        rhs = rhs + decay_dydt(Y_trial, state.cloud.species)
                    return rhs + _src - _kappa * Y_trial

                # One linearized backward-Euler solve when that stays
                # non-negative; otherwise Newton-iterated backward Euler, then
                # sub-steps (see jacobian.robust_implicit_euler_step).
                y_new, delta_y, _clipped, _J_diag, _thermo_info = robust_implicit_euler_step(
                    y=state.cloud.Y,
                    rhs_func=_thermo_rhs,
                    dt=dt,
                    significant_floor=dt_rel_floor,
                )
                _thermo_path_counts[_thermo_info["path"]] = _thermo_path_counts.get(_thermo_info["path"], 0) + 1
                # Destruction lifetime of each species (-1/J_ii), for the
                # timestep controller.
                _tau_destroy = 1.0 / np.maximum(-_J_diag, 1e-300)
                if np.any(_clipped & (np.abs(state.cloud.Y) > dt_rel_floor)):
                    log.warning(
                        "step %d: implicit update still clipped to zero for %s at dt=%.3e s.",
                        step, [state.cloud.species[i] for i in np.flatnonzero(_clipped)], dt,
                    )
                state.cloud.Y = y_new

            # --- Credit the cascade's baryon injection to density_cm3 ---
            # result.dYdt_cloud already moved mass between species (targets
            # destroyed, products created, jet self-thermalization via
            # s_thermalized). Since Y_i = n_i/n_baryons must always satisfy
            # Sum_i A_i Y_i = 1, the baryon-density denominator has to grow by
            # exactly the same fractional amount the numerator just grew by:
            #     n_new = n_old * (1 + dt * Sum_i A_i dYdt_cloud_i)
            # Previously density_cm3 was left untouched here and only grown
            # later by the unrelated bulk jet-mass rate below, so every
            # reacted/thermalized baryon was added to Y without ever being
            # credited to the baryon budget -- manufacturing mass for free.
            _n_before_cascade = state.cloud.density_cm3
            if thermonuclear_enabled and dt > 0.0:
                # Thermonuclear reactions and decay conserve Sum_i A_i Y_i
                # exactly, so the whole of this step's baryon change came from
                # the jet-cloud terms inside the implicit solve.
                _frac_cascade = float(np.dot(_A_cloud, state.cloud.Y - _Y_step_start))
            else:
                _frac_cascade = dt * float(np.dot(_A_cloud, result.dYdt_cloud))
            _frac_cascade = max(_frac_cascade, -0.999)  # guard against a pathological negative rate
            state.cloud.density_cm3 = _n_before_cascade * (1.0 + _frac_cascade)
            state.cloud.Y = state.cloud.Y / (1.0 + _frac_cascade)

            # --- Constant-volume density evolution (Famiano ρ̇ = K) ---
            # density_cm3 = baryon number density (n_baryons/cm³); Y_i = n_i/n_baryons.
            # Ṁ is the TOTAL jet mass entering the cloud this step, reacted and
            # unreacted alike. The reacted/thermalized share was already
            # credited above via the cascade, so only the remainder is mixed
            # in as raw jet composition here -- otherwise that mass would be
            # double-counted (once as reaction products, once again as raw
            # p/⁴He from the bulk rate).
            #   1. Dilute existing composition: Y_i *= n_baryons_old / n_baryons_new
            #   2. Add remaining jet baryons at jet baryon fractions (Y_i_jet = number_frac_i / mean_A_jet)
            # This preserves Sum_i A_i Y_i = 1 at all times.
            _m_cascade_g = _frac_cascade * _n_before_cascade * _V_cloud_cm3 * _AMU_G
            _mass_rate_g_s_eff = _mass_rate_g_s if _jet_on else 0.0
            _m_bulk_g = max(0.0, _mass_rate_g_s_eff * dt - _m_cascade_g)
            _old_density = state.cloud.density_cm3
            state.cloud.density_cm3 += _m_bulk_g / (_V_cloud_cm3 * _AMU_G)
            _f_old = _old_density / state.cloud.density_cm3   # old baryon fraction
            _f_new = 1.0 - _f_old                              # new jet baryon fraction
            state.cloud.Y *= _f_old
            for _sp, _Y_jet in _jet_baryon_fractions.items():
                if _sp in state.cloud.species:
                    state.cloud.Y[state.cloud.species.index(_sp)] += _Y_jet * _f_new

            # --- Radioactive decay (operator split) ---
            # Applied to the thermalized cloud reservoir only (not the
            # in-flight non-thermal spectra, which are reset each step -- see
            # update_spectra=False above). Uses the exact exponential solution
            # rather than explicit Euler because several tracked species
            # (8Li, 8B, 16N) have half-lives far shorter than dt_max_s, which
            # would make forward Euler numerically unstable.
            # With thermonuclear reactions enabled, decay was already applied
            # inside the implicit solve above.
            #
            # The decay change is added to delta_y (and the post-decay Y is
            # what the controller sees) so compute_next_dt judges each
            # species by its net change over the step. Excluding it made a
            # species in production/decay balance look like it changed by
            # Y*dt/tau every step, forcing dt down to gamma*tau.
            if decay_enabled and dt > 0.0 and not thermonuclear_enabled:
                _Y_pre_decay = state.cloud.Y
                state.cloud.Y = apply_decay(_Y_pre_decay, state.cloud.species, dt)
                delta_y = delta_y + (state.cloud.Y - _Y_pre_decay)

            # --- Adaptive timestep (Famiano eq. 4) ---
            # Species whose destruction lifetime is shorter than the step the
            # rest of the network allows are left out of the ratio test (see
            # compute_next_dt): the implicit solve already puts them at their
            # production/destruction balance. Without this, ⁸Li (t½ = 0.84 s,
            # Y ~ 1e-17) pinned dt at ~0.03-0.15 s in the Model A run.
            # Lifetimes come from the thermonuclear Jacobian (which includes
            # decay); with thermonuclear reactions off, from decay alone.
            if thermonuclear_enabled and dt > 0.0:
                _tau_for_dt = _tau_destroy
            elif decay_enabled:
                _tau_for_dt = np.array(
                    [DECAY_TABLE[sp].half_life_s / np.log(2.0) if sp in DECAY_TABLE else np.inf
                     for sp in state.cloud.species],
                    dtype=float,
                )
            else:
                _tau_for_dt = None
            dt_next, _dt_limiter, _n_dt_excluded = compute_next_dt(
                y_new=state.cloud.Y,
                delta_y=delta_y,
                dt_current=dt,
                gamma=gamma,
                rel_floor=dt_rel_floor,
                dt_min=dt_min,
                dt_max=dt_max,
                destruction_lifetime_s=_tau_for_dt,
                return_limiter=True,
            )

            # --- Advance time and track ΔM/M₀ ---
            # advance_time() moves t_s forward by state.solver.dt_s, so that
            # field must hold the dt actually used above for this step's
            # update (the possibly boundary-clamped `dt`), NOT dt_next --
            # dt_next is the size proposed for the step AFTER this one.
            # Setting it first and advancing second, as before, moved time
            # by next step's size instead of this step's, silently
            # decoupling t_s from the dt used to compute y_new/delta_y/decay
            # on every step. dt_next is staged into state.solver.dt_s only
            # once advance_time() has consumed the correct value.
            delta_m_over_m0 += _mass_rate_g_s_eff * dt / _M0_g
            state.solver.set_dt(dt)
            state.solver.advance_time()
            state.solver.set_dt(dt_next)
            step += 1

            if step % write_every == 0:
                log.info(
                    "step %6d | t = %.4e s | dt = %.4e s | ΔM/M₀ = %.3e | Y_p = %.4e"
                    " | dt set by %s (%d fast species skipped) | thermo solves %s",
                    step, state.solver.t_s, dt_next, delta_m_over_m0,
                    state.cloud.get_abundance("p") if "p" in state.cloud.species else float("nan"),
                    state.cloud.species[_dt_limiter] if _dt_limiter is not None else "none",
                    _n_dt_excluded,
                    dict(_thermo_path_counts) or "-",
                )
                _thermo_path_counts.clear()
            # Write every step through _fine_write_until_step (recent jet
            # transition) even when it falls between write_every_n_steps
            # rows, on top of the normal coarse cadence -- see
            # fine_write_steps_after_transition above.
            if step % write_every == 0 or step <= _fine_write_until_step:
                _write_history_row(fh, writer, state, state.solver.t_s, step, delta_m_over_m0)

        # Write final row
        _write_history_row(fh, writer, state, state.solver.t_s, step, delta_m_over_m0)

    wall_time = time.time() - t_start_wall
    log.info(
        "Evolution finished: %d steps, t_final = %.4e s, wall time = %.1f s",
        step, state.solver.t_s, wall_time,
    )

    # Write final state JSON
    final = {
        "t_s": state.solver.t_s,
        "step": step,
        "delta_m_over_m0": delta_m_over_m0,
        "stop_reason": state.solver.stop_reason,
        "abundances": {sp: float(y) for sp, y in zip(state.cloud.species, state.cloud.Y)},
        "mass_fractions": state.get_mass_fractions(),
    }
    with open(out_json, "w") as f:
        json.dump(final, f, indent=2)
    log.info("Final state written to %s", out_json)
    log.info("Abundance history written to %s", out_csv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Famiano (2002) blazar nucleosynthesis runner")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "config" / "run.json"),
        help="Path to run.json (default: config/run.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
