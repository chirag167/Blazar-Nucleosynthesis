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
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


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