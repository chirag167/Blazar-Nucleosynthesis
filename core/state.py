# core/state.py
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class NetworkState:
    """
    Container for the evolving network state.
    """

    t: float
    Y: np.ndarray
    T: float = 1.0e9
    rho: float = 1.0

    # Derivative storage (initialized later)
    dY: np.ndarray = field(init=False)

    # --------------------------------------------------
    # Initialization hook
    # --------------------------------------------------

    def __post_init__(self):
        # Ensure Y is numpy array
        self.Y = np.array(self.Y, dtype=float)

        # Allocate derivative storage
        self.dY = np.zeros_like(self.Y)

    # --------------------------------------------------
    # Utility methods
    # --------------------------------------------------

    def copy(self) -> "NetworkState":
        new = NetworkState(
            t=float(self.t),
            Y=self.Y.copy(),
            T=float(self.T),
            rho=float(self.rho),
        )
        return new

    @property
    def n_species(self) -> int:
        return int(self.Y.size)

    # --------------------------------------------------
    # Derivative management
    # --------------------------------------------------

    def reset_derivatives(self):
        """Reset accumulated dY/dt."""
        self.dY[:] = 0.0

    # --------------------------------------------------
    # Time stepping
    # --------------------------------------------------

    def compute_dt(self):
        """
        Simple adaptive timestep:
            dt ~ 0.1 * min(Y / |dY|)
        """
        eps = 1e-30
        mask = np.abs(self.dY) > eps

        if not np.any(mask):
            return 1.0  # nothing happening

        dt_candidates = np.abs(self.Y[mask] / self.dY[mask])
        dt = 0.1 * np.min(dt_candidates)

        return max(min(dt, 1e6), 1e-12)

    def apply_update(self, dt):
        """Explicit Euler update."""
        self.Y += dt * self.dY

        # Prevent negative abundances
        self.Y[self.Y < 0] = 0.0

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate(self, *, require_finite: bool = True) -> None:
        if self.Y is None:
            raise ValueError("State.Y is None.")

        if self.Y.ndim != 1:
            raise ValueError(f"State.Y must be 1D, got shape={self.Y.shape}.")

        if require_finite:
            if not np.isfinite(self.t):
                raise ValueError("State.t is not finite.")
            if not np.all(np.isfinite(self.Y)):
                raise ValueError("State.Y contains non-finite values.")
            if not np.isfinite(self.T):
                raise ValueError("State.T is not finite.")
            if not np.isfinite(self.rho):
                raise ValueError("State.rho is not finite.")

        if self.T <= 0:
            raise ValueError(f"Temperature must be > 0, got T={self.T}.")
        if self.rho <= 0:
            raise ValueError(f"Density must be > 0, got rho={self.rho}.")
