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

from state import CloudState, CascadeState, SolverState, NetworkState, SpeciesData, ProjectileSpectrum
from grids import make_energy_grid
from reactions import ReactionLibrary
from cascade import run_cascade_step
from timestep import compute_next_dt, estimate_initial_dt, euler_increment

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


def _load_configs(config_dir: Path) -> tuple[dict, dict, dict, dict]:
    cloud_cfg  = _load_json(config_dir / "cloud.json")
    jet_cfg    = _load_json(config_dir / "jet.json")
    species_cfg = _load_json(config_dir / "species.json")
    run_cfg    = _load_json(config_dir / "run.json")
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
    mass_rate_cfg = jet_cfg.get("mass_rate", {})
    mass_rate_Msun_yr = float(mass_rate_cfg.get("model_B_Msun_per_yr", 1e-6))
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
    cloud_cfg, jet_cfg, species_cfg, run_cfg = _load_configs(config_dir)

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
        include_group1=False,   # Group1 requires DWBA product distributions — not yet implemented
        include_group2=True,
    )
    log.info("Loaded %d reaction channels.", len(lib.reactions))

    # Initial state
    state = build_network_state(cloud_cfg, jet_cfg, species_cfg, run_cfg, energy_edges)

    # Expand cloud with all possible reaction products (Y=0) so that
    # dYdt_cloud has a valid slot for every species a reaction can produce.
    expand_cloud_with_reaction_products(state, lib)
    log.info("Cloud species after expansion: %s", state.cloud.species)

    # Rescale injection spectra to physical units.
    # spectrum.values[k] = n_jet_species(k) / n_baryon_cloud so that the
    # cascade reaction rates come out in [abundance per baryon per second].
    f_norm = compute_jet_normalization_factor(jet_cfg, cloud_cfg, species_cfg)
    for spec in state.cascade.spectra.values():
        spec.values[:] *= f_norm

    # BLR total mass and jet mass rate for ΔM/M₀ tracking.
    # ΔM/M₀ = cumulative jet mass transiting BLR / total BLR mass (Famiano's convention).
    blr_mass_Msun = float(cloud_cfg.get("blr_mass_Msun", 1.0e6))
    _M0_g = blr_mass_Msun * _MSUN_G
    _mass_rate_g_s = (
        float(jet_cfg.get("mass_rate", {}).get("model_B_Msun_per_yr", 1e-6))
        * _MSUN_G / _S_PER_YR
    )
    log.info(
        "BLR mass M₀ = %.3e M_sun | jet mass rate = %.3e g/s | "
        "ΔM/M₀ rate = %.3e yr⁻¹",
        blr_mass_Msun, _mass_rate_g_s,
        _mass_rate_g_s * _S_PER_YR / _M0_g,
    )

    # Snapshot of the steady-state jet injection spectrum.
    # In Famiano's model the jet continuously transits the cloud; unreacted
    # particles exit rather than thermalize. We maintain a constant spectrum
    # (the steady-state jet density) by resetting to the injection values at
    # the start of every cascade step.  Only nuclear-reaction products modify
    # the cloud abundances.
    _injection_values: dict[str, np.ndarray] = {
        sp: spec.values.copy()
        for sp, spec in state.cascade.spectra.items()
    }

    # Output setup
    output_cfg = run_cfg.get("output", {})
    out_csv = _ROOT / output_cfg.get("history_csv", "outputs/abundance_history.csv")
    out_json = _ROOT / output_cfg.get("final_state_json", "outputs/final_state.json")
    write_every = int(output_cfg.get("write_every_n_steps", 100))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    solver_cfg = run_cfg.get("solver", {})
    gamma    = float(solver_cfg.get("gamma", 0.01))
    dt_min   = float(solver_cfg.get("dt_min_s", 1e4))
    dt_max   = float(solver_cfg.get("dt_max_s", 1e11))
    max_steps = int(solver_cfg.get("max_steps", 100000))

    import csv
    fieldnames = ["step", "t_s", "delta_m_over_m0"] + list(state.cloud.species)

    delta_m_over_m0 = 0.0   # cumulative ΔM/M₀

    t_start_wall = time.time()
    log.info("Starting evolution: t_end = %.3e s, max_steps = %d", state.solver.t_end_s, max_steps)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        _write_history_row(fh, writer, state, state.solver.t_s, 0, delta_m_over_m0)

        step = 0
        while not state.solver.done and step < max_steps:
            dt = state.solver.dt_s

            # --- Reset jet spectrum to steady-state injection values ---
            # Famiano's model: jet continuously transits cloud at constant flux.
            # Unreacted particles exit; we maintain the constant injection
            # profile each step so reaction rates depend only on cloud composition.
            for sp, inj_vals in _injection_values.items():
                if sp in state.cascade.spectra:
                    state.cascade.spectra[sp].values[:] = inj_vals
                else:
                    from state import ProjectileSpectrum
                    state.cascade.spectra[sp] = ProjectileSpectrum(
                        species=sp,
                        energy_MeV=energy_edges[:-1] + 0.5 * np.diff(energy_edges),
                        values=inj_vals.copy(),
                    )

            # --- Cascade step ---
            result = run_cascade_step(
                state=state,
                reaction_library=lib,
                energy_edges_mev=energy_edges,
                dt_s=dt,
                update_cloud=False,   # we apply the Euler step manually below
                update_spectra=False, # steady-state: spectrum is reset each step
            )

            # --- Explicit Euler abundance update ---
            y_new, delta_y = euler_increment(
                y=state.cloud.Y,
                dydt=result.dYdt_cloud,
                dt=dt,
                enforce_nonnegative=True,
            )
            state.cloud.Y = y_new

            # --- Adaptive timestep (Famiano eq. 4) ---
            dt_next = compute_next_dt(
                y_new=y_new,
                delta_y=delta_y,
                dt_current=dt,
                gamma=gamma,
                dt_min=dt_min,
                dt_max=dt_max,
            )
            state.solver.set_dt(dt_next)

            # --- Advance time and track ΔM/M₀ ---
            delta_m_over_m0 += _mass_rate_g_s * dt / _M0_g
            state.solver.advance_time()
            step += 1

            if step % write_every == 0:
                log.info(
                    "step %6d | t = %.4e s | dt = %.4e s | ΔM/M₀ = %.3e | Y_p = %.4e",
                    step, state.solver.t_s, dt_next, delta_m_over_m0,
                    state.cloud.get_abundance("p") if "p" in state.cloud.species else float("nan"),
                )
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
