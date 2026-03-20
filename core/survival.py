"""
survival.py

Discrete survival fractions and bin-by-bin yields for the non-thermal solver.

This module follows Famiano's discrete formalism:
- S_j^i is the cumulative survival fraction down to energy bin i.
- Delta S_j^i = S_j(E_i; E0) - S_j(E_{i-1}; E0) is the destruction probability
  in a single bin. 
- beta_k^i is the destruction branching fraction for reaction i in projectile
  energy bin k. 
- The discrete yield is
      y_pq^(i,n) = tau_pq^(i,k) * beta_k^i * Delta S_k^i / S_n^i
  as given in eq. (11). 
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------
# Basic grid helpers
# ---------------------------------------------------------------------

def bin_centers_from_edges(E_edges: np.ndarray) -> np.ndarray:
    E_edges = np.asarray(E_edges, dtype=float)
    if E_edges.ndim != 1 or len(E_edges) < 2:
        raise ValueError("E_edges must be a 1D array of length >= 2.")
    return 0.5 * (E_edges[:-1] + E_edges[1:])


def bin_widths_from_edges(E_edges: np.ndarray) -> np.ndarray:
    E_edges = np.asarray(E_edges, dtype=float)
    if E_edges.ndim != 1 or len(E_edges) < 2:
        raise ValueError("E_edges must be a 1D array of length >= 2.")
    dE = np.diff(E_edges)
    if np.any(dE <= 0.0):
        raise ValueError("E_edges must be strictly increasing.")
    return dE


# ---------------------------------------------------------------------
# Bin-averaged cross sections
# ---------------------------------------------------------------------

def bin_average_cross_section(
    sigma_func,
    E_edges: np.ndarray,
    n_sub: int = 8,
    **sigma_kwargs,
) -> np.ndarray:
    """
    Average a cross section sigma(E) over each energy bin.

    Parameters
    ----------
    sigma_func : callable
        Function sigma(E, **sigma_kwargs) -> cross section.
    E_edges : array, shape (n_bins + 1,)
        Energy bin edges.
    n_sub : int
        Number of interior points for bin averaging.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Bin-averaged cross section values.
    """
    E_edges = np.asarray(E_edges, dtype=float)
    n_bins = len(E_edges) - 1
    out = np.zeros(n_bins, dtype=float)

    for k in range(n_bins):
        E_lo = E_edges[k]
        E_hi = E_edges[k + 1]
        E_sample = np.linspace(E_lo, E_hi, n_sub + 2)[1:-1]
        vals = np.asarray(sigma_func(E_sample, **sigma_kwargs), dtype=float)
        out[k] = np.mean(vals)

    return out


# ---------------------------------------------------------------------
# Survival fractions
# ---------------------------------------------------------------------

def compute_total_destruction_coefficient(
    sigma_bin: np.ndarray,
    target_densities: np.ndarray,
) -> np.ndarray:
    """
    Compute the total destruction coefficient in each projectile-energy bin:

        Lambda_k = sum_m N_m * sigma_mk

    Parameters
    ----------
    sigma_bin : array, shape (n_rxn, n_bins)
        Bin-averaged destruction cross sections for all reactions destroying
        the projectile.
    target_densities : array, shape (n_rxn,)
        Target number density for each destruction channel.

    Returns
    -------
    np.ndarray, shape (n_bins,)
    """
    sigma_bin = np.asarray(sigma_bin, dtype=float)
    target_densities = np.asarray(target_densities, dtype=float)

    if sigma_bin.ndim != 2:
        raise ValueError("sigma_bin must have shape (n_rxn, n_bins).")
    if target_densities.shape != (sigma_bin.shape[0],):
        raise ValueError("target_densities must have shape (n_rxn,).")

    return np.sum(target_densities[:, None] * sigma_bin, axis=0)


def compute_survival_table(
    epsilon_bin: np.ndarray,
    sigma_bin: np.ndarray,
    target_densities: np.ndarray,
    E_edges: np.ndarray,
    initial_bin: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the cumulative discrete survival fraction S_k for a projectile
    injected into bin `initial_bin`.

    This is the discrete analog of Famiano eqs. (5), (6), and (8). :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

    Parameters
    ----------
    epsilon_bin : array, shape (n_bins,)
        Bin-averaged stopping power [MeV/cm].
    sigma_bin : array, shape (n_rxn, n_bins)
        Bin-averaged destruction cross sections.
    target_densities : array, shape (n_rxn,)
        Number densities of target particles for each destruction channel.
    E_edges : array, shape (n_bins + 1,)
        Energy bin edges.
    initial_bin : int, optional
        Bin index of the injected projectile energy. Defaults to the top bin.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        S_k = fraction surviving down to and through bin k.

    Convention
    ----------
    - Bin index increases with energy.
    - If injected in bin n, then S[n] = 1.
    - Lower-energy bins k < n are built recursively downward.
    - Higher-energy bins k > n are left at 0.
    """
    epsilon_bin = np.asarray(epsilon_bin, dtype=float)
    E_edges = np.asarray(E_edges, dtype=float)
    dE = np.diff(E_edges)

    n_bins = len(dE)
    if epsilon_bin.shape != (n_bins,):
        raise ValueError("epsilon_bin must have shape (n_bins,).")

    if initial_bin is None:
        initial_bin = n_bins - 1

    if not (0 <= initial_bin < n_bins):
        raise ValueError("initial_bin out of range.")

    Lambda = compute_total_destruction_coefficient(sigma_bin, target_densities)

    S = np.zeros(n_bins, dtype=float)
    S[initial_bin] = 1.0

    # March downward from the injection bin.
    # Using the discrete analog:
    #   DeltaS_k ~ S_k * (Lambda_k / epsilon_k) * dE_k
    #   S_{k-1} = S_k - DeltaS_k
    for k in range(initial_bin, 0, -1):
        eps_k = max(epsilon_bin[k], 1e-300)
        deltaS_k = S[k] * (Lambda[k] / eps_k) * dE[k]
        deltaS_k = np.clip(deltaS_k, 0.0, S[k])
        S[k - 1] = S[k] - deltaS_k

    return S


def compute_delta_survival(S: np.ndarray) -> np.ndarray:
    """
    Compute Delta S_k, the destruction fraction in each energy bin.

    Famiano eq. (9):
        Delta S_k = S(E_k; E0) - S(E_{k-1}; E0)  [up to indexing convention]
    with the lowest bin defined directly from S(E_1; E0). :contentReference[oaicite:15]{index=15}

    Our ascending-energy-bin convention gives:
        DeltaS[k] = S[k] - S[k-1]   for k >= 1
        DeltaS[0] = S[0]

    Since S decreases toward lower energy when marching downward from the
    injection bin, we return a positive destruction fraction by taking:
        DeltaS[k] = S[k] - S[k-1]   for k >= 1
    and clipping small negatives from roundoff.
    """
    S = np.asarray(S, dtype=float)
    n_bins = len(S)

    deltaS = np.zeros(n_bins, dtype=float)
    deltaS[0] = S[0]

    for k in range(1, n_bins):
        deltaS[k] = S[k] - S[k - 1]

    deltaS = np.maximum(deltaS, 0.0)
    return deltaS


def compute_normalized_survival_between_bins(
    S: np.ndarray,
    lower_bin: int,
    upper_bin: int,
) -> float:
    """
    Famiano eq. (6):
        S(E2; E1) = S(E2; E0) / S(E1; E0)

    Here `upper_bin` corresponds to E1 and `lower_bin` to E2, with lower_bin <= upper_bin.
    """
    S = np.asarray(S, dtype=float)
    if not (0 <= lower_bin < len(S) and 0 <= upper_bin < len(S)):
        raise ValueError("Bin index out of range.")
    if lower_bin > upper_bin:
        raise ValueError("Require lower_bin <= upper_bin.")

    denom = S[upper_bin]
    if denom <= 0.0:
        return 0.0
    return float(S[lower_bin] / denom)


# ---------------------------------------------------------------------
# Destruction fractions beta
# ---------------------------------------------------------------------

def compute_beta(
    sigma_bin: np.ndarray,
    target_densities: np.ndarray,
) -> np.ndarray:
    """
    Compute Famiano's destruction fraction beta_k^i for each reaction i and
    projectile-energy bin k:

        beta_k^i = sigma_k^i N_i / sum_m sigma_k^m N_m   . :contentReference[oaicite:16]{index=16}

    Parameters
    ----------
    sigma_bin : array, shape (n_rxn, n_bins)
    target_densities : array, shape (n_rxn,)

    Returns
    -------
    np.ndarray, shape (n_rxn, n_bins)
    """
    sigma_bin = np.asarray(sigma_bin, dtype=float)
    target_densities = np.asarray(target_densities, dtype=float)

    numer = target_densities[:, None] * sigma_bin
    denom = np.sum(numer, axis=0, keepdims=True)

    beta = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer),
        where=denom > 0.0,
    )
    return beta


# ---------------------------------------------------------------------
# Discrete yield tensor
# ---------------------------------------------------------------------

def compute_discrete_yield(
    tau: np.ndarray,
    beta_rxn_bin: np.ndarray,
    deltaS_bin: np.ndarray,
    initial_bin: int,
) -> np.ndarray:
    """
    Compute Famiano's discrete yield:

        y_pq^(i,n) = tau_pq^(i,k) * beta_k^i * DeltaS_k / S_n  . :contentReference[oaicite:17]{index=17}

    Since the injected projectile bin n is normalized so that S_n = 1 in
    compute_survival_table(), this reduces to:
        y_pq^(i,n) = tau_pq^(i,k) * beta_k^i * DeltaS_k

    Parameters
    ----------
    tau : array, shape (n_rxn, n_proj_bins, n_products, n_prod_bins)
        Product-energy distribution tensor.
        For fixed (rxn, proj_bin, product), tau over product bins must sum to 1.
    beta_rxn_bin : array, shape (n_rxn, n_proj_bins)
        Destruction fractions beta.
    deltaS_bin : array, shape (n_proj_bins,)
        Per-bin destruction fractions.
    initial_bin : int
        Injection bin index. Included for clarity and future generalization.

    Returns
    -------
    np.ndarray, shape (n_rxn, n_proj_bins, n_products, n_prod_bins)
        Discrete yield tensor for one injected projectile.
    """
    tau = np.asarray(tau, dtype=float)
    beta_rxn_bin = np.asarray(beta_rxn_bin, dtype=float)
    deltaS_bin = np.asarray(deltaS_bin, dtype=float)

    if tau.ndim != 4:
        raise ValueError(
            "tau must have shape (n_rxn, n_proj_bins, n_products, n_prod_bins)."
        )

    n_rxn, n_proj_bins, _, _ = tau.shape

    if beta_rxn_bin.shape != (n_rxn, n_proj_bins):
        raise ValueError("beta_rxn_bin must have shape (n_rxn, n_proj_bins).")
    if deltaS_bin.shape != (n_proj_bins,):
        raise ValueError("deltaS_bin must have shape (n_proj_bins,).")

    factor = beta_rxn_bin * deltaS_bin[None, :]
    y = tau * factor[:, :, None, None]
    return y


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def build_survival_and_yield(
    epsilon_bin: np.ndarray,
    sigma_bin: np.ndarray,
    target_densities: np.ndarray,
    tau: np.ndarray,
    E_edges: np.ndarray,
    initial_bin: Optional[int] = None,
):
    """
    Convenience wrapper returning all key discrete quantities:
    S, DeltaS, beta, y

    Returns
    -------
    dict
        {
            "S": cumulative survival fractions,
            "deltaS": per-bin nuclear destruction fractions (bin 0 uses actual
                      nuclear rate, NOT the full S[0] thermalized fraction),
            "beta": destruction fractions by reaction,
            "yield": discrete yield tensor,
            "s_thermalized": fraction of projectiles that thermalize without
                             nuclear reaction (survive to below-grid energies).
        }

    Notes
    -----
    The ``compute_delta_survival`` function uses the convention
    ``deltaS[0] = S[0]`` as an absorbing-boundary shortcut, treating ALL
    particles reaching the lowest energy bin as having undergone nuclear
    reactions there.  This overestimates yields by many orders of magnitude
    for a tenuous cloud where the nuclear mean-free-path >> stopping range.

    We correct this by replacing ``deltaS[0]`` with the actual nuclear
    destruction fraction at bin 0: ``S[0] * Lambda[0] / eps[0] * dE[0]``.
    The remainder ``S[0] - deltaS_nuc[0]`` is the thermalized (non-reacting)
    fraction and is returned separately as ``s_thermalized``.
    """
    n_bins = len(E_edges) - 1
    if initial_bin is None:
        initial_bin = n_bins - 1

    S = compute_survival_table(
        epsilon_bin=epsilon_bin,
        sigma_bin=sigma_bin,
        target_densities=target_densities,
        E_edges=E_edges,
        initial_bin=initial_bin,
    )

    deltaS = compute_delta_survival(S)

    # --- Fix bin-0 boundary condition ---
    # compute_delta_survival sets deltaS[0] = S[0], which incorrectly counts
    # ALL thermalized projectiles as nuclear-reaction products.  Replace with
    # the physical nuclear destruction fraction at bin 0.
    dE = np.diff(np.asarray(E_edges, dtype=float))
    Lambda_0 = compute_total_destruction_coefficient(sigma_bin, target_densities)[0]
    eps_0 = max(float(epsilon_bin[0]), 1e-300)
    deltaS_nuc_0 = float(S[0]) * (Lambda_0 / eps_0) * float(dE[0])
    deltaS_nuc_0 = np.clip(deltaS_nuc_0, 0.0, float(S[0]))
    s_thermalized = float(S[0]) - deltaS_nuc_0  # survives to below grid
    deltaS[0] = deltaS_nuc_0

    beta_rxn_bin = compute_beta(sigma_bin, target_densities)
    y = compute_discrete_yield(
        tau=tau,
        beta_rxn_bin=beta_rxn_bin,
        deltaS_bin=deltaS,
        initial_bin=initial_bin,
    )

    return {
        "S": S,
        "deltaS": deltaS,
        "beta": beta_rxn_bin,
        "yield": y,
        "s_thermalized": s_thermalized,
    }