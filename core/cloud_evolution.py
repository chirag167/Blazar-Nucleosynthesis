"""
cloud_evolution.py

Updates cloud geometry (density_cm3 or volume) after each cascade step,
implementing the three evolution modes described in Famiano et al. (2002).

Modes
-----
static        Cloud parameters are held constant.  Use this when cloud
              geometry evolution is not needed.

const_volume  Cloud volume V is fixed.  Net new nuclei from the cascade
              increase the number density:
                  n_new = n_old + r_nuclei * dt

const_density Cloud number density n is held constant.  Net new nuclei
              cause the cloud to expand:
                  V_new = V_old * (1 + sum_i dY_i/dt * dt)

Baryon accounting
-----------------
Two sources of baryon injection are distinguished:

  1. Nuclear-reaction products deposited into the cloud — captured by
     dYdt_cloud from compute_cascade_step().  The net nuclei injection rate
     is n * sum_i (dY_i/dt), which is positive whenever a cascade projectile
     contributes its baryons to the cloud.

  2. Thermalised (below-grid) projectiles — NOT yet automatically tracked
     in CascadeStepResult.  These are available as `s_thermalized` inside
     build_survival_and_yield() but are not currently accumulated in the
     cascade loop.  Use inject_thermalized() to add them explicitly once
     that accounting is wired up.

Usage
-----
    from cloud_evolution import CloudEvolution, CloudMode

    evo = CloudEvolution(mode=CloudMode.CONST_VOLUME, V_cm3=1e30)

    # inside the main time loop, after compute_cascade_step():
    evo.step(state, result.dYdt_cloud, dt_s)
    apply_cloud_euler_step_from_cache(state, dt_s=dt_s)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from state import NetworkState


class CloudMode(str, Enum):
    STATIC = "static"
    CONST_VOLUME = "const_volume"
    CONST_DENSITY = "const_density"


@dataclass
class CloudEvolution:
    """
    Tracks and updates cloud geometry after each cascade step.

    Parameters
    ----------
    mode : CloudMode | str
        One of "static", "const_volume", or "const_density".
    V_cm3 : float
        Initial (and for const_volume, fixed) cloud volume [cm^3].
    """

    mode: CloudMode
    V_cm3: float

    def __post_init__(self) -> None:
        self.mode = CloudMode(self.mode)
        self.V_cm3 = float(self.V_cm3)
        if self.V_cm3 <= 0.0:
            raise ValueError("V_cm3 must be positive.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def nuclei_injection_rate(
        self,
        state: NetworkState,
        dYdt_cloud: np.ndarray,
    ) -> float:
        """
        Net nuclei density injection rate [cm^-3 s^-1] from the cascade.

        Computed as n_cloud × Σ_i (dY_i/dt).

        This is positive whenever a cascade projectile's baryons are
        incorporated into the cloud via nuclear reactions (the projectile
        is not counted in the cloud abundances, but the extra products are).
        Zero for purely thermonuclear reactions (both reactants in cloud).
        """
        return float(state.cloud.density_cm3 * np.sum(dYdt_cloud))

    def baryon_injection_rate(
        self,
        state: NetworkState,
        dYdt_cloud: np.ndarray,
    ) -> float:
        """
        Net baryon density injection rate [cm^-3 s^-1] from the cascade.

        Computed as n_cloud × Σ_i A_i (dY_i/dt).  Use this when tracking
        baryon number rather than nuclei count.
        """
        if state.species_data:
            A = np.array(
                [state.species_data[sp].A for sp in state.cloud.species],
                dtype=float,
            )
        else:
            A = np.ones(len(state.cloud.species), dtype=float)
        return float(state.cloud.density_cm3 * np.dot(A, dYdt_cloud))

    # ------------------------------------------------------------------
    # Evolution step
    # ------------------------------------------------------------------

    def step(
        self,
        state: NetworkState,
        dYdt_cloud: np.ndarray,
        dt_s: float,
    ) -> float:
        """
        Update cloud geometry by one timestep.

        Call this BEFORE or AFTER apply_cloud_euler_step_from_cache — the
        two operations are independent (one updates Y_i, the other updates
        density_cm3 or V_cm3).  Calling before is recommended so that the
        density used in the nuclei-rate calculation is the pre-step value.

        Parameters
        ----------
        state : NetworkState
            Current state.  density_cm3 is updated in-place for
            const_volume mode.
        dYdt_cloud : np.ndarray, shape (n_species,)
            Cloud abundance rate from the current cascade step
            (CascadeStepResult.dYdt_cloud).
        dt_s : float
            Timestep [s].

        Returns
        -------
        float
            Net nuclei injection rate [cm^-3 s^-1] used in this step.
        """
        if self.mode == CloudMode.STATIC:
            return 0.0

        r = self.nuclei_injection_rate(state, dYdt_cloud)

        if self.mode == CloudMode.CONST_VOLUME:
            state.cloud.density_cm3 = max(
                0.0, state.cloud.density_cm3 + r * float(dt_s)
            )

        elif self.mode == CloudMode.CONST_DENSITY:
            # Fractional volume change = sum_i dY_i × dt
            # (new nuclei / total nuclei = ΔV / V at fixed n)
            frac = float(np.sum(dYdt_cloud)) * float(dt_s)
            self.V_cm3 = max(0.0, self.V_cm3 * (1.0 + frac))

        return r

    # ------------------------------------------------------------------
    # Thermalization hook (future use)
    # ------------------------------------------------------------------

    def inject_thermalized(
        self,
        state: NetworkState,
        n_nuclei_cm3: float,
        species_fractions: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add thermalised (below-grid) nuclei to the cloud.

        Call this once per timestep after accumulating
            sum_bins  s_thermalized(k) × amount(k)
        for each projectile species and converting to a nuclei density.

        This is separate from step() because the s_thermalized bookkeeping
        is not yet wired into CascadeStepResult.  Once it is, these two
        methods can be merged.

        Parameters
        ----------
        state : NetworkState
        n_nuclei_cm3 : float
            Number of thermalised nuclei per cm^3 deposited this timestep.
        species_fractions : np.ndarray, shape (n_cloud_species,), optional
            Fractional composition of the thermalised material.
            If provided, the cloud Y_i are updated accordingly.
            If omitted, only the bulk density/volume is updated.
        """
        if self.mode == CloudMode.STATIC or n_nuclei_cm3 <= 0.0:
            return

        if self.mode == CloudMode.CONST_VOLUME:
            n_old = state.cloud.density_cm3
            n_new = max(0.0, n_old + n_nuclei_cm3)
            if species_fractions is not None and n_new > 0.0:
                # Blend thermalised composition into cloud Y_i
                frac = np.asarray(species_fractions, dtype=float)
                if frac.shape == state.cloud.Y.shape:
                    frac = frac / (frac.sum() or 1.0)
                    state.cloud.Y = (
                        n_old * state.cloud.Y + n_nuclei_cm3 * frac
                    ) / n_new
            state.cloud.density_cm3 = n_new

        elif self.mode == CloudMode.CONST_DENSITY:
            n = state.cloud.density_cm3
            if n > 0.0:
                dV = n_nuclei_cm3 * self.V_cm3 / n
                self.V_cm3 = max(0.0, self.V_cm3 + dV)
                if species_fractions is not None:
                    # Blend thermalised composition into cloud Y_i
                    total_before = n * self.V_cm3
                    total_after = n * (self.V_cm3 + dV)
                    frac = np.asarray(species_fractions, dtype=float)
                    if frac.shape == state.cloud.Y.shape and total_after > 0.0:
                        frac = frac / (frac.sum() or 1.0)
                        state.cloud.Y = (
                            total_before * state.cloud.Y + n_nuclei_cm3 * self.V_cm3 * frac
                        ) / total_after
