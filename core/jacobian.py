"""
jacobian.py

Computes the Jacobian matrix for the cloud abundance evolution system:

    dY/dt = F(Y, t)

The Jacobian is defined as

    J_ij = dF_i / dY_j

This module currently provides a finite-difference Jacobian for generality.
It can later be extended with semi-analytic sparse assembly for better speed
and accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, identity, lil_matrix
from scipy.sparse.linalg import spsolve


Array = np.ndarray


@dataclass
class JacobianConfig:
    """
    Configuration for numerical Jacobian construction.

    Attributes
    ----------
    rel_step : float
        Relative perturbation size used for finite differences.
    abs_step : float
        Minimum absolute perturbation size.
    sparse : bool
        If True, return a scipy sparse CSR matrix.
    """
    rel_step: float = 1e-6
    abs_step: float = 1e-12
    sparse: bool = True


def finite_difference_jacobian(
    y: Array,
    t: float,
    rhs_func: Callable[[Array, float], Array],
    config: Optional[JacobianConfig] = None,
) -> csr_matrix | Array:
    """
    Compute the Jacobian J = dF/dY using forward finite differences.

    Parameters
    ----------
    y : np.ndarray
        Current abundance vector of shape (n_species,).
    t : float
        Current time.
    rhs_func : callable
        Function rhs_func(y, t) -> dydt with output shape (n_species,).
    config : JacobianConfig, optional
        Numerical Jacobian configuration.

    Returns
    -------
    scipy.sparse.csr_matrix or np.ndarray
        Jacobian matrix with shape (n_species, n_species).
    """
    if config is None:
        config = JacobianConfig()

    y = np.asarray(y, dtype=float)
    f0 = np.asarray(rhs_func(y, t), dtype=float)

    n = y.size

    if config.sparse:
        J = lil_matrix((n, n), dtype=float)
    else:
        J = np.zeros((n, n), dtype=float)

    for j in range(n):
        h = max(config.abs_step, config.rel_step * max(abs(y[j]), 1.0))

        y_pert = y.copy()
        y_pert[j] += h

        f1 = np.asarray(rhs_func(y_pert, t), dtype=float)
        col = (f1 - f0) / h

        if config.sparse:
            nz = np.flatnonzero(col)
            if nz.size > 0:
                J[nz, j] = col[nz]
        else:
            J[:, j] = col

    return J.tocsr() if config.sparse else J


def implicit_euler_step(
    y: Array,
    rhs_func: Callable[[Array], Array],
    dt: float,
    config: Optional[JacobianConfig] = None,
    enforce_nonnegative: bool = True,
    return_jacobian_diag: bool = False,
):
    """
    One linearized implicit (backward) Euler step, per Famiano eq. (2):

        (I/h - J) . Delta = f(Y(t))

    with J = dF/dY evaluated at the CURRENT state Y(t) via
    finite_difference_jacobian(). This is a single linear solve, not a
    Newton iteration to convergence -- i.e. the standard linearized/
    semi-implicit scheme used in stellar nucleosynthesis codes (Timmes
    1999, which Famiano cites for exactly this formalism), not full
    implicit Euler solved to a residual tolerance. For a genuinely stiff
    reaction network this one linear solve is what makes the step
    unconditionally stable (unlike explicit Euler); it does NOT guarantee
    the result satisfies dY/dt = f(Y_new) exactly, only to first order in
    dt.

    Parameters
    ----------
    y : np.ndarray
        Current abundance vector, shape (n_species,).
    rhs_func : callable
        Autonomous right-hand side: rhs_func(y) -> dydt, shape (n_species,).
        (No explicit t dependence -- the caller's physics, e.g.
        core/thermonuclear.py's compute_thermonuclear_dydt at fixed T9 and
        density, is autonomous over one step.)
    dt : float
        Timestep h.
    config : JacobianConfig, optional
        Passed through to finite_difference_jacobian().
    enforce_nonnegative : bool
        If True, clip negative updated abundances to zero (same convention
        as timestep.euler_increment()).
    return_jacobian_diag : bool
        If True, also return diag(J) = dF_i/dY_i at Y(t). For a reaction/
        decay network -diag(J)_i is species i's total destruction rate [1/s],
        so 1/(-diag(J)_i) is its destruction lifetime (used by the timestep
        controller to recognise species that equilibrate within a step).

    Returns
    -------
    y_new, delta_y, was_clipped
        Updated abundances, the abundance increment actually applied
        (post-clip, if enforce_nonnegative), and a boolean mask of which
        entries the nonnegative clip actually changed. Compare against
        this mask, not against rhs_func(y)*dt -- the raw explicit-Euler
        increment is a different quantity from the implicit solve's Delta
        for a stiff system and will disagree with it even when the
        nonnegative clip here never fires.

        With return_jacobian_diag=True a fourth value, diag(J), is appended.
    """
    y = np.asarray(y, dtype=float)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("implicit_euler_step requires a positive finite dt.")

    n = y.size
    f0 = np.asarray(rhs_func(y), dtype=float)

    J = finite_difference_jacobian(
        y=y, t=0.0, rhs_func=lambda yy, tt: rhs_func(yy), config=config,
    )

    if not isinstance(J, csr_matrix):
        J = csr_matrix(J)

    A = (identity(n, format="csc") / dt) - J.tocsc()

    try:
        delta = spsolve(A, f0)
    except Exception as exc:
        raise RuntimeError(
            "implicit_euler_step: linear solve (I/h - J).Delta = f(Y) failed "
            f"(dt={dt:.3e}, n_species={n}). The Jacobian may be singular or "
            "ill-conditioned at this state."
        ) from exc

    delta = np.asarray(delta, dtype=float).reshape(-1)
    if not np.all(np.isfinite(delta)):
        raise RuntimeError(
            f"implicit_euler_step: non-finite solution at dt={dt:.3e} "
            "(NaN/Inf in Delta) -- linear solve did not converge to a "
            "usable result."
        )

    y_new = y + delta
    was_clipped = np.zeros(n, dtype=bool)
    if enforce_nonnegative:
        y_new_clipped = np.maximum(y_new, 0.0)
        was_clipped = y_new_clipped != y_new
        y_new = y_new_clipped
        delta = y_new - y

    if return_jacobian_diag:
        return y_new, delta, was_clipped, np.asarray(J.diagonal(), dtype=float)
    return y_new, delta, was_clipped

def robust_implicit_euler_step(
    y: Array,
    rhs_func: Callable[[Array], Array],
    dt: float,
    significant_floor: float = 1e-20,
    config: Optional[JacobianConfig] = None,
    newton_max_iter: int = 10,
    newton_rtol: float = 1e-6,
    max_substep_halvings: int = 6,
):
    """
    Backward-Euler step that stays non-negative when dt is much longer than
    some species' lifetimes.

    1. Try one linearized step (implicit_euler_step). If it clips no species
       whose abundance matters (|Y| > significant_floor before or after the
       step), accept it -- this is the cheap path taken on most steps.
    2. Otherwise iterate the full backward-Euler equations
           G(Y) = (Y - Y0)/dt - f(Y) = 0
       with Newton's method (Jacobian re-evaluated each iteration), starting
       from the clipped linearized result. The single linearized step only
       matches backward Euler to first order; when dt >> a species'
       lifetime and its sources or partners change a lot within the step
       (e.g. n, t, 8Li once dt ~ 10 s at Model A densities), that error can
       exceed the species' own abundance and drive it negative. The
       converged solution does not.
    3. If Newton does not converge to a non-negative state, split dt into
       2, 4, ... sub-steps of this same procedure.

    Returns (y_new, delta_y, was_clipped, jacobian_diag, info), where
    jacobian_diag is diag(J) at the starting state and info is a dict with
    'path' ('linear' | 'newton' | 'substeps') and 'newton_iters'.
    """
    y0 = np.asarray(y, dtype=float)
    y_lin, d_lin, clipped, J_diag = implicit_euler_step(
        y0, rhs_func, dt, config=config, enforce_nonnegative=True, return_jacobian_diag=True,
    )
    matters = (np.abs(y0) > significant_floor) | (np.abs(y0 + d_lin) > significant_floor)
    if not np.any(clipped & matters):
        return y_lin, d_lin, clipped, J_diag, {"path": "linear", "newton_iters": 0}

    n = y0.size
    yk = y_lin.copy()
    converged = False
    it = 0
    for it in range(1, newton_max_iter + 1):
        J = finite_difference_jacobian(y=yk, t=0.0, rhs_func=lambda yy, tt: rhs_func(yy), config=config)
        J = csr_matrix(J) if not isinstance(J, csr_matrix) else J
        G = (yk - y0) / dt - np.asarray(rhs_func(yk), dtype=float)
        A = (identity(n, format="csc") / dt) - J.tocsc()
        step = np.asarray(spsolve(A, -G), dtype=float).reshape(-1)
        if not np.all(np.isfinite(step)):
            break
        yk = yk + step
        scale = np.maximum(np.abs(yk), significant_floor)
        if np.max(np.abs(step) / scale) < newton_rtol:
            converged = True
            break

    if converged and np.all(yk >= -significant_floor):
        y_new = np.maximum(yk, 0.0)
        return y_new, y_new - y0, y_new != yk, J_diag, {"path": "newton", "newton_iters": it}

    if max_substep_halvings <= 0:
        return y_lin, d_lin, clipped, J_diag, {"path": "linear-fallback", "newton_iters": it}

    y_sub = y0
    for _ in range(2):
        y_sub, _, _, _, _ = robust_implicit_euler_step(
            y_sub, rhs_func, dt / 2.0, significant_floor, config,
            newton_max_iter, newton_rtol, max_substep_halvings - 1,
        )
    clipped_sub = np.zeros(n, dtype=bool)
    return y_sub, y_sub - y0, clipped_sub, J_diag, {"path": "substeps", "newton_iters": it}
