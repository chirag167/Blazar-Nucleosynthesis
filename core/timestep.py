"""
timestep.py

Adaptive timestep control for the non-thermal abundance evolution.

This module implements the timestep prescription based on equation (4)
of Famiano et al. (2002):

    h_{n+1} = gamma * h_n * min_i (Y_i^{n+1} / D_i)

where
    - h_n       is the current timestep,
    - h_{n+1}   is the next timestep,
    - Y_i^{n+1} is the updated abundance of species i,
    - D_i       is the abundance change over the current step,
    - gamma << 1 is a safety factor.

In practice, numerical safeguards are required:
    - species with vanishingly small abundance are ignored,
    - species with zero or tiny abundance change are ignored,
    - absolute values are used in the ratio to enforce "small relative change",
    - dt is clipped to user/project-defined minimum and maximum values.

This module is intentionally independent of the network physics. The
engine should compute abundance changes, then call these functions to
choose the next timestep.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


DEFAULT_GAMMA = 0.01
DEFAULT_ABS_FLOOR = 1.0e-300
DEFAULT_REL_FLOOR = 1.0e-30


def _as_1d_float_array(x: np.ndarray | list | tuple, name: str) -> np.ndarray:
    """Convert input to a 1D float NumPy array."""
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")

    return arr


def _clip_dt(
    dt: float,
    dt_min: Optional[float],
    dt_max: Optional[float],
) -> float:
    """Apply lower/upper bounds to dt."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Computed timestep is invalid: {dt}")

    if dt_min is not None:
        if dt_min <= 0.0:
            raise ValueError("dt_min must be positive")
        dt = max(dt, dt_min)

    if dt_max is not None:
        if dt_max <= 0.0:
            raise ValueError("dt_max must be positive")
        dt = min(dt, dt_max)

    return dt


def compute_next_dt(
    y_new: np.ndarray | list | tuple,
    delta_y: np.ndarray | list | tuple,
    dt_current: float,
    gamma: float = DEFAULT_GAMMA,
    abs_floor: float = DEFAULT_ABS_FLOOR,
    rel_floor: float = DEFAULT_REL_FLOOR,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    max_growth: Optional[float] = 5.0,
) -> float:
    """
    Compute the next timestep using the Famiano equation (4) prescription.

    Parameters
    ----------
    y_new
        Updated abundance vector Y^{n+1}.
    delta_y
        Abundance increment over the current step:
            delta_y = Y^{n+1} - Y^n
        For an explicit Euler update this is typically dydt * dt_current.
    dt_current
        Current timestep h_n.
    gamma
        Safety factor in equation (4). Famiano uses a value much less than 1;
        gamma = 0.01 is the natural default.
    abs_floor
        Small absolute cutoff used to avoid division by zero.
    rel_floor
        Species with |Y_i| <= rel_floor are ignored in the ratio test.
    dt_min, dt_max
        Optional hard bounds on the returned timestep.
    max_growth
        Optional multiplicative cap on timestep growth:
            h_{n+1} <= max_growth * h_n
        Helps avoid sudden jumps when the solution temporarily becomes quiet.

    Returns
    -------
    float
        Next timestep h_{n+1}.

    Notes
    -----
    The literal paper expression uses Y_i^{n+1} / D_i. Numerically, since the
    stated goal is to keep abundance changes small for every species, we use

        |Y_i^{n+1}| / |D_i|

    and take the minimum over species that are numerically meaningful.

    Species are ignored if:
        - |Y_i^{n+1}| <= rel_floor
        - |D_i| <= abs_floor

    If no species survive the filtering, the timestep is kept unchanged
    (subject to clipping and growth cap).
    """
    y_new = _as_1d_float_array(y_new, "y_new")
    delta_y = _as_1d_float_array(delta_y, "delta_y")

    if y_new.shape != delta_y.shape:
        raise ValueError(
            f"Shape mismatch: y_new has shape {y_new.shape}, "
            f"delta_y has shape {delta_y.shape}"
        )

    if not np.isfinite(dt_current) or dt_current <= 0.0:
        raise ValueError("dt_current must be a positive finite float")

    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite float")

    if abs_floor <= 0.0 or rel_floor < 0.0:
        raise ValueError("abs_floor must be > 0 and rel_floor must be >= 0")

    y_abs = np.abs(y_new)
    d_abs = np.abs(delta_y)

    valid = (y_abs > rel_floor) & (d_abs > abs_floor)

    if np.any(valid):
        ratios = y_abs[valid] / d_abs[valid]
        ratio_min = np.min(ratios)
        dt_next = gamma * dt_current * ratio_min
    else:
        # Nothing changed in a numerically meaningful way, so do not force
        # the timestep to collapse or explode.
        dt_next = dt_current

    if max_growth is not None:
        if max_growth <= 1.0:
            raise ValueError("max_growth must be > 1 if provided")
        dt_next = min(dt_next, max_growth * dt_current)

    return _clip_dt(dt_next, dt_min=dt_min, dt_max=dt_max)


def estimate_initial_dt(
    y0: np.ndarray | list | tuple,
    dydt0: np.ndarray | list | tuple,
    gamma: float = DEFAULT_GAMMA,
    abs_floor: float = DEFAULT_ABS_FLOOR,
    rel_floor: float = DEFAULT_REL_FLOOR,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
) -> float:
    """
    Estimate a reasonable first timestep when no previous dt exists.

    This is not directly given by equation (4), because equation (4) assumes
    h_n is already known. For the very first step, we choose dt so that the
    expected relative abundance change is approximately gamma:

        dt_0 ~ gamma * min_i (|Y_i| / |dY_i/dt|)

    Parameters
    ----------
    y0
        Initial abundance vector.
    dydt0
        Initial abundance time derivative.
    gamma
        Safety factor, typically 0.01.
    abs_floor, rel_floor
        Numerical thresholds for ignoring tiny values.
    dt_min, dt_max
        Optional hard bounds.

    Returns
    -------
    float
        Initial timestep estimate.
    """
    y0 = _as_1d_float_array(y0, "y0")
    dydt0 = _as_1d_float_array(dydt0, "dydt0")

    if y0.shape != dydt0.shape:
        raise ValueError(
            f"Shape mismatch: y0 has shape {y0.shape}, dydt0 has shape {dydt0.shape}"
        )

    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite float")

    y_abs = np.abs(y0)
    rate_abs = np.abs(dydt0)

    valid = (y_abs > rel_floor) & (rate_abs > abs_floor)

    if not np.any(valid):
        raise ValueError(
            "Cannot estimate initial dt: no species have both "
            "significant abundance and nonzero time derivative."
        )

    dt0 = gamma * np.min(y_abs[valid] / rate_abs[valid])

    return _clip_dt(dt0, dt_min=dt_min, dt_max=dt_max)


def euler_increment(
    y: np.ndarray | list | tuple,
    dydt: np.ndarray | list | tuple,
    dt: float,
    enforce_nonnegative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper for an explicit Euler abundance update.

    Parameters
    ----------
    y
        Current abundance vector.
    dydt
        Current abundance derivative vector.
    dt
        Timestep.
    enforce_nonnegative
        If True, clip negative updated abundances to zero.

    Returns
    -------
    y_new, delta_y
        Updated abundances and abundance increment.

    Notes
    -----
    This helper is optional. If your engine already computes the update,
    you do not need to use it.
    """
    y = _as_1d_float_array(y, "y")
    dydt = _as_1d_float_array(dydt, "dydt")

    if y.shape != dydt.shape:
        raise ValueError(
            f"Shape mismatch: y has shape {y.shape}, dydt has shape {dydt.shape}"
        )

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite float")

    delta_y = dydt * dt
    y_new = y + delta_y

    if enforce_nonnegative:
        y_new = np.maximum(y_new, 0.0)
        delta_y = y_new - y

    return y_new, delta_y