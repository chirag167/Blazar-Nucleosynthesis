from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

"""
state.py

This module defines the core data containers for the non-thermal nucleosynthesis
simulation. It stores the full physical and numerical state of the system as it
evolves in time, including the cloud composition, non-thermal particle spectra,
species metadata, and solver runtime information.

The purpose of this file is to keep all simulation state organized in a single,
consistent structure that can be passed between the different parts of the code.
In particular, it separates the problem into three main components:

1. Cloud state:
   Stores the target cloud properties and the evolving abundances of all nuclear
   species. The abundances are stored internally as Y_i and are updated during
   the cascade evolution.

2. Cascade state:
   Stores the non-thermal particle distributions, spectra, and any intermediate
   quantities produced by the cascade solver, such as cached source terms or
   reaction-rate information.

3. Solver state:
   Stores the runtime evolution variables, including the current simulation time,
   the final end time, the current timestep, iteration count, and stopping flags.
   The timestep is not user-specified; it is computed internally during the
   evolution according to the timestep prescription adopted from the Famiano
   formalism.

These state objects are grouped into a top-level NetworkState object, which
represents the complete state of a simulation at a given instant. This design
makes it easier to:
- pass the simulation state cleanly between modules,
- update abundances self-consistently during the cascade,
- validate inputs and runtime quantities,
- load initial conditions from JSON configuration files,
- extend the code later without changing the overall architecture.

This file is intended only for data structures and validation logic. It should
not contain the main physics solvers themselves, such as stopping power,
survival fractions, cascade transport, timestep calculation, or abundance-rate
evaluation. Those belong in separate modules.

"""

ArrayLike = np.ndarray


@dataclass
class SpeciesData:
    """
    Nuclear/species metadata.

    Attributes
    ----------
    name : str
        Species name, e.g. 'H1', 'He4', 'Li7', 'p', 'n', 'alpha'
    A : int
        Mass number
    Z : int
        Charge number
    """
    name: str
    A: int
    Z: int


@dataclass
class CloudState:
    """
    State of the cloud/target medium.

    Internal convention:
    - abundances are stored as Y_i (number abundance per baryon-like normalization)
    - species ordering is fixed by `species`
    - Y must have same length as species
    """
    species: List[str]
    Y: ArrayLike
    density_cm3: float
    temperature_K: float
    ionization_fraction: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.Y = np.asarray(self.Y, dtype=float)
        self.validate()

    def validate(self) -> None:
        if len(self.species) != len(self.Y):
            raise ValueError(
                f"CloudState mismatch: len(species)={len(self.species)} "
                f"but len(Y)={len(self.Y)}"
            )
        if np.any(self.Y < 0.0):
            raise ValueError("CloudState.Y contains negative abundances.")
        if self.density_cm3 < 0.0:
            raise ValueError("density_cm3 must be non-negative.")
        if self.temperature_K < 0.0:
            raise ValueError("temperature_K must be non-negative.")
        if not (0.0 <= self.ionization_fraction <= 1.0):
            raise ValueError("ionization_fraction must lie in [0, 1].")

    @property
    def n_species(self) -> int:
        return len(self.species)

    def species_index(self, name: str) -> int:
        try:
            return self.species.index(name)
        except ValueError as exc:
            raise KeyError(f"Species '{name}' not found in cloud state.") from exc

    def get_abundance(self, name: str) -> float:
        return float(self.Y[self.species_index(name)])

    def set_abundance(self, name: str, value: float) -> None:
        if value < 0.0:
            raise ValueError("Abundance cannot be negative.")
        self.Y[self.species_index(name)] = value

    def as_dict(self) -> Dict[str, float]:
        return {sp: float(y) for sp, y in zip(self.species, self.Y)}

    def clip_negative(self) -> None:
        self.Y = np.maximum(self.Y, 0.0)

    def renormalize(self) -> None:
        """
        Renormalize abundances so sum(Y)=1, if possible.

        Use this only if your chosen abundance convention requires it.
        """
        total = np.sum(self.Y)
        if total > 0.0:
            self.Y = self.Y / total

    def copy(self) -> "CloudState":
        return CloudState(
            species=self.species.copy(),
            Y=self.Y.copy(),
            density_cm3=self.density_cm3,
            temperature_K=self.temperature_K,
            ionization_fraction=self.ionization_fraction,
            metadata=self.metadata.copy(),
        )


@dataclass
class ProjectileSpectrum:
    """
    Non-thermal projectile spectrum on an energy grid.
    """
    species: str
    energy_MeV: ArrayLike
    values: ArrayLike
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.energy_MeV = np.asarray(self.energy_MeV, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.validate()

    def validate(self) -> None:
        if len(self.energy_MeV) != len(self.values):
            raise ValueError(
                f"ProjectileSpectrum mismatch: len(energy_MeV)={len(self.energy_MeV)} "
                f"but len(values)={len(self.values)}"
            )
        if np.any(self.energy_MeV < 0.0):
            raise ValueError("Energy grid contains negative values.")
        if np.any(self.values < 0.0):
            raise ValueError("Spectrum values must be non-negative.")

    @property
    def n_bins(self) -> int:
        return len(self.energy_MeV)

    def copy(self) -> "ProjectileSpectrum":
        return ProjectileSpectrum(
            species=self.species,
            energy_MeV=self.energy_MeV.copy(),
            values=self.values.copy(),
            metadata=self.metadata.copy(),
        )


@dataclass
class CascadeState:
    """
    State of the non-thermal cascade.

    spectra :
        Dictionary keyed by projectile species, e.g.
        {
            "p": ProjectileSpectrum(...),
            "alpha": ProjectileSpectrum(...),
        }

    rhs_cache :
        Optional storage for intermediate rates/source terms computed during
        cascade solving and reused by abundance RHS evaluation.
    """
    spectra: Dict[str, ProjectileSpectrum] = field(default_factory=dict)
    rhs_cache: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for key, spectrum in self.spectra.items():
            if key != spectrum.species:
                raise ValueError(
                    f"CascadeState key '{key}' does not match "
                    f"ProjectileSpectrum.species '{spectrum.species}'."
                )
            spectrum.validate()

    def projectile_species(self) -> List[str]:
        return list(self.spectra.keys())

    def get_spectrum(self, species: str) -> ProjectileSpectrum:
        try:
            return self.spectra[species]
        except KeyError as exc:
            raise KeyError(f"No spectrum stored for projectile '{species}'.") from exc

    def set_spectrum(self, spectrum: ProjectileSpectrum) -> None:
        self.spectra[spectrum.species] = spectrum

    def copy(self) -> "CascadeState":
        return CascadeState(
            spectra={k: v.copy() for k, v in self.spectra.items()},
            rhs_cache=self.rhs_cache.copy(),
            metadata=self.metadata.copy(),
        )


@dataclass
class SolverState:
    """
    Numerical controls and current integration status.

    Notes
    -----
    - t_s starts at 0 by construction
    - t_end_s is user-specified
    - dt_s is computed internally during evolution (using equation 4 from Famiano et al. 2002).
    """
    t_s: float
    t_end_s: float
    step: int = 0
    dt_s: Optional[float] = None
    done: bool = False
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.t_s < 0.0:
            raise ValueError("Current time t_s must be non-negative.")
        if self.t_end_s < self.t_s:
            raise ValueError("t_end_s must be >= t_s.")
        if self.dt_s is not None and self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive if set.")

    def set_dt(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            raise ValueError("Computed dt_s must be positive.")
        self.dt_s = dt_s

    def advance_time(self) -> None:
        if self.dt_s is None:
            raise ValueError("Cannot advance time because dt_s has not been set.")
        self.t_s += self.dt_s
        self.step += 1
        if self.t_s >= self.t_end_s:
            self.t_s = self.t_end_s
            self.done = True
            if self.stop_reason is None:
                self.stop_reason = "reached_t_end"

    def copy(self) -> "SolverState":
        return SolverState(
            t_s=self.t_s,
            t_end_s=self.t_end_s,
            step=self.step,
            dt_s=self.dt_s,
            done=self.done,
            stop_reason=self.stop_reason,
            metadata=self.metadata.copy(),
        )
    

@dataclass
class NetworkState:
    """
    Full simulation state.

    species_data :
        Optional metadata for all known species.
    cloud :
        Evolving cloud abundances and medium properties.
    cascade :
        Evolving non-thermal cascade state.
    solver :
        Time integration state.
    diagnostics :
        Arbitrary run-time diagnostics.
    """
    cloud: CloudState
    cascade: CascadeState
    solver: SolverState
    species_data: Dict[str, SpeciesData] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.cloud.validate()
        self.cascade.validate()
        self.solver.validate()

        for name, sp in self.species_data.items():
            if name != sp.name:
                raise ValueError(
                    f"species_data key '{name}' does not match SpeciesData.name '{sp.name}'."
                )

        for sp in self.cloud.species:
            if self.species_data and sp not in self.species_data:
                raise ValueError(
                    f"Cloud species '{sp}' missing from species_data."
                )

    @property
    def time_s(self) -> float:
        return self.solver.t_s

    @property
    def dt_s(self) -> float:
        return self.solver.dt_s

    def copy(self) -> "NetworkState":
        return NetworkState(
            cloud=self.cloud.copy(),
            cascade=self.cascade.copy(),
            solver=self.solver.copy(),
            species_data={
                k: SpeciesData(name=v.name, A=v.A, Z=v.Z)
                for k, v in self.species_data.items()
            },
            diagnostics=self.diagnostics.copy(),
        )

    def get_mass_fractions(self) -> Dict[str, float]:
        """
        Convert internal Y_i abundances to mass fractions X_i using

            X_i = A_i Y_i / sum_j(A_j Y_j)

        Requires species_data.
        """
        if not self.species_data:
            raise ValueError("species_data is required to compute mass fractions.")

        A = np.array([self.species_data[sp].A for sp in self.cloud.species], dtype=float)
        numerator = A * self.cloud.Y
        denom = np.sum(numerator)

        if denom <= 0.0:
            return {sp: 0.0 for sp in self.cloud.species}

        X = numerator / denom
        return {sp: float(x) for sp, x in zip(self.cloud.species, X)}

    @classmethod
    def from_dicts(
        cls,
        cloud_dict: Dict[str, Any],
        solver_dict: Dict[str, Any],
        species_dict: Optional[Dict[str, Any]] = None,
        cascade_dict: Optional[Dict[str, Any]] = None,
    ) -> "NetworkState":
        """
        Minimal constructor from plain dictionaries.
        Useful before writing a dedicated io.py loader.
        """

        species_data: Dict[str, SpeciesData] = {}
        if species_dict is not None:
            for name, vals in species_dict.items():
                species_data[name] = SpeciesData(
                    name=name,
                    A=int(vals["A"]),
                    Z=int(vals["Z"]),
                )

        abundance_type = cloud_dict.get("abundance_type", "Y")
        abundances = cloud_dict["abundances"]
        species = list(abundances.keys())

        if abundance_type == "Y":
            Y = np.array([abundances[sp] for sp in species], dtype=float)

        elif abundance_type == "X":
            if not species_data:
                raise ValueError(
                    "species_data is required to convert mass fractions X to abundances Y."
                )
            X = np.array([abundances[sp] for sp in species], dtype=float)
            A = np.array([species_data[sp].A for sp in species], dtype=float)
            denom = np.sum(X / A)
            if denom <= 0.0:
                raise ValueError("Cannot convert X to Y because normalization is invalid.")
            Y = (X / A) / denom

        else:
            raise ValueError(f"Unsupported abundance_type '{abundance_type}'.")

        cloud = CloudState(
            species=species,
            Y=Y,
            density_cm3=float(cloud_dict["density_cm3"]),
            temperature_K=float(cloud_dict["temperature_K"]),
            ionization_fraction=float(cloud_dict["ionization_fraction"]),
            metadata=cloud_dict.get("metadata", {}),
        )

        solver = SolverState(
            t_s=float(solver_dict["time"]["t0_s"]),
            dt_s=float(solver_dict["time"]["dt_s"]),
            t_end_s=float(solver_dict["time"]["tmax_s"]),
            metadata=solver_dict.get("metadata", {}),
        )

        cascade = CascadeState(metadata=(cascade_dict or {}).get("metadata", {}))

        state = cls(
            cloud=cloud,
            cascade=cascade,
            solver=solver,
            species_data=species_data,
        )
        state.validate()
        return state