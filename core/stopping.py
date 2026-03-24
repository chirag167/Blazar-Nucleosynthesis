"""
stopping.py

Stopping power and bin-averaged energy-loss helpers for the non-thermal solver.

This module is written for an energy grid defined by bin edges:
    E_edges[k] < E < E_edges[k+1]

Notes
-----
- Famiano (2002) uses the stopping power epsilon(E) in the survival-fraction
  formalism and works with discrete energy bins. This module therefore exposes
  bin-averaged epsilon_k values suitable for use in survival.py. :contentReference[oaicite:4]{index=4}
- The actual fast/slow ion energy-loss expressions below follow the structure
  of your current draft, but are reorganized so they are usable on a bin grid.
  Your draft currently computes dE/dt and epsilon(E)= (1/v) dE/dt. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from utils.utils import beta, lorentz_factor, c, AMU_TO_MEV, m_e


# ---------------------------------------------------------------------
# Mean excitation potential helper
# ---------------------------------------------------------------------

I_VALS = {
    "H": 19.2,
    "He": 41.8,
    "Li": 40.0,
    "Be": 63.7,
    "C": 78.0,
    "N": 82.0,
    "O": 95.0,
    "F": 115.0,
    "Ne": 137.0,
    "Na": 149.0,
    "Mg": 156.0,
}


def mean_excitation_potential(
    Z_cloud: np.ndarray,
    A_cloud: np.ndarray,
    composition_weights: np.ndarray,
    element_symbols: Optional[list[str]] = None,
) -> float:
    """
    Weighted mean excitation potential of the cloud in MeV.

    Parameters
    ----------
    Z_cloud, A_cloud : arrays
        Atomic and mass numbers of cloud species.
    composition_weights : array
        Relative cloud composition weights for each species.
    element_symbols : list[str], optional
        Element symbols aligned with Z_cloud / A_cloud. If omitted,
        the empirical fallback formula is used for all species.

    Returns
    -------
    float
        Weighted mean excitation potential in MeV.
    """
    Z_cloud = np.asarray(Z_cloud, dtype=float)
    A_cloud = np.asarray(A_cloud, dtype=float)
    composition_weights = np.asarray(composition_weights, dtype=float)

    if element_symbols is None:
        element_symbols = [None] * len(Z_cloud)

    I_list = []
    weights = []

    for Zi, Ai, wi, sym in zip(Z_cloud, A_cloud, composition_weights, element_symbols):
        if sym in I_VALS:
            I_eV = I_VALS[sym]
        else:
            I_eV = (9.76 + 58.8 * Zi ** (-1.19)) * Zi

        I_list.append(I_eV)
        weights.append(wi * Zi / Ai)

    I_arr = np.asarray(I_list, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if np.sum(weights) <= 0.0:
        raise ValueError("Composition weights must sum to a positive value.")

    I_avg_eV = np.exp(np.sum(weights * np.log(I_arr)) / np.sum(weights))
    return I_avg_eV * 1e-6  # eV -> MeV


# ---------------------------------------------------------------------
# Stopping model
# ---------------------------------------------------------------------

def energy_loss_rate(
    A_cl: np.ndarray,
    X_cl: np.ndarray,
    X_ion: float,
    Z_proj: int,
    E: np.ndarray | float,
    A_proj: int,
    n_e: float,
    T_e: float,
    eps: float = 1e-2,
) -> np.ndarray:
    """
    Continuous energy-loss rate dE/dt [MeV/s] for a fast ion in a PLASMA.

    Uses the relativistic Bethe-Bloch formula with the plasma frequency
    cutoff replacing the mean excitation potential I → e_0 = ħω_p
    (Ginzburg & Syrovatskii, "The Origin of Cosmic Rays", eqs. 7.6–7.7):

        dE/dt = v × (K/N_A) × n_e × Z_eff²/β²
                × [0.5 ln(2 m_e β² γ² T_max / e_0²) − β²]

    where
        e_0 = ħω_p = 3.71×10⁻¹¹ √n_e  eV   (plasma frequency cutoff)
        T_max = 2 m_e β² γ²                   (heavy-projectile limit, M ≫ m_e)
        K/N_A = 4π r_e² m_e c² = 5.099×10⁻²⁵ MeV cm²

    For v < v₀ = c/137 the Bohr effective charge is applied:
        Z_eff = Z (v/v₀)^(1/3)   (G&S eq. 7.11)

    The heavy-projectile T_max approximation (= 2 m_e β² γ²) is valid for
    E ≪ (M/m_e) M c² ≈ 1.72 × 10⁶ MeV for protons, covering our full
    0–400 MeV injection energy range.

    Parameters
    ----------
    A_cl, X_cl : arrays
        Cloud mass numbers and composition fractions (unused in plasma formula
        but kept for interface consistency).
    X_ion : float
        Ionization fraction of the cloud (unused here; weighting done in
        ``stopping_power()``).
    Z_proj, A_proj : int
        Projectile charge and mass number.
    E : array or float
        Projectile kinetic energy [MeV].
    n_e : float
        Free electron number density [cm⁻³].
    T_e : float
        Electron temperature [K]. Retained for interface stability.
    eps : float
        Unused; retained for interface compatibility.

    Returns
    -------
    np.ndarray
        Positive magnitude of plasma energy-loss rate dE/dt [MeV/s].
    """
    E = np.asarray(E, dtype=float)
    E = np.maximum(E, 1e-300)

    M_proj = A_proj * AMU_TO_MEV          # projectile rest mass [MeV/c²]
    gamma_arr = 1.0 + E / M_proj
    beta_arr = np.sqrt(np.maximum(1.0 - 1.0 / gamma_arr**2, 1e-30))
    beta2 = beta_arr**2

    # --- Bohr effective charge (G&S eq. 7.11) ---
    # For v < v₀ = αc = c/137 the ion is not fully stripped; Z_eff < Z.
    beta_0 = 1.0 / 137.0
    Z_eff_sq = np.where(
        beta_arr < beta_0,
        float(Z_proj)**2 * (beta_arr / beta_0) ** (2.0 / 3.0),
        float(Z_proj)**2,
    )

    # --- Plasma frequency cutoff energy (G&S eq. 7.7) ---
    # e_0 = ħω_p = 3.71×10⁻¹¹ √n_e  eV  →  MeV
    e_0 = np.maximum(3.71e-11 * np.sqrt(n_e) * 1e-6, 1e-30)   # MeV

    # --- Maximum energy transfer (heavy-projectile limit) ---
    T_max = 2.0 * m_e * beta2 * gamma_arr**2                    # MeV

    # --- Bethe-Bloch logarithmic term ---
    arg = np.maximum(
        2.0 * m_e * beta2 * gamma_arr**2 * T_max / e_0**2,
        1.0 + 1e-15,
    )
    log_term = 0.5 * np.log(arg) - beta2
    log_term = np.maximum(log_term, 0.0)

    # --- dE/dx for plasma [MeV/cm] ---
    # K/N_A = 4π r_e² m_e c² = 0.307 MeV cm²/g / (6.022×10²³ mol⁻¹)
    K_over_NA = 0.307 / 6.022e23          # MeV cm²
    dEdx = K_over_NA * Z_eff_sq * n_e / np.maximum(beta2, 1e-30) * log_term

    # --- dE/dt = v × dE/dx ---
    c_cm_s = c * 1e2                      # speed of light [cm/s]
    v = beta_arr * c_cm_s
    dEdt = dEdx * v

    dEdt = np.nan_to_num(dEdt, nan=0.0, posinf=0.0, neginf=0.0)
    dEdt = np.maximum(dEdt, 0.0)
    return dEdt


def _bethe_bloch_neutral(
    A_cl: np.ndarray,
    X_cl: np.ndarray,
    Z_proj: int,
    A_proj: int,
    E: np.ndarray,
    n_cl_total: float,
) -> np.ndarray:
    """
    Relativistic Bethe-Bloch stopping power for a fast ion in a *neutral* medium
    [MeV/cm].

    Uses the standard relativistic expression (Bethe 1930):
        dE/dx = K * z^2 * (Z/A)_target * rho * (1/beta^2) *
                [0.5 * ln(2 m_e c^2 beta^2 gamma^2 T_max / I^2) - beta^2]

    where T_max ~ 2 m_e c^2 beta^2 gamma^2 for M_proj >> m_e (projectile much
    heavier than electron), and K = 4pi N_A r_e^2 m_e c^2 = 0.307 MeV cm^2/g.

    Parameters
    ----------
    A_cl, X_cl : arrays
        Cloud mass numbers and mass fractions.
    Z_proj, A_proj : int
        Projectile charge and mass number.
    E : array
        Projectile kinetic energy [MeV].
    n_cl_total : float
        Total cloud baryon number density [cm^-3].

    Returns
    -------
    np.ndarray
        Stopping power [MeV/cm].
    """
    K = 0.307  # MeV cm^2 / g
    AMU_G = 1.66054e-24  # g

    M_proj = A_proj * AMU_TO_MEV        # MeV/c^2
    gamma_arr = 1.0 + E / M_proj
    beta_arr = np.sqrt(np.maximum(1.0 - 1.0 / gamma_arr**2, 1e-30))
    beta2 = beta_arr**2

    # Weighted (Z/A) and mean excitation potential for the neutral cloud
    A_arr = np.asarray(A_cl, dtype=float)
    X_arr = np.asarray(X_cl, dtype=float)

    # Approximate Z = A/2 for heavier nuclei, exact for H (Z=1, A=1)
    # Use A as proxy for mass, approximate Z_i ≈ A_i/2 for A>1, Z=1 for H
    Z_arr = np.where(A_arr <= 1.0, 1.0, 0.5 * A_arr)

    # Z/A weighted by mass fraction
    ZoverA = np.sum(X_arr * Z_arr / A_arr)

    # Mean excitation potential (log average weighted by Z/A)
    I_arr_eV = np.array([
        I_VALS.get("H", 19.2) if a <= 1.0 else
        I_VALS.get("He", 41.8) if a <= 4.5 else
        (9.76 + 58.8 * z**(-1.19)) * z
        for a, z in zip(A_arr, Z_arr)
    ])
    weights = X_arr * Z_arr / A_arr
    if weights.sum() > 0:
        I_mean_eV = np.exp(np.sum(weights * np.log(np.maximum(I_arr_eV, 1.0))) / weights.sum())
    else:
        I_mean_eV = 19.2
    I_mean_MeV = I_mean_eV * 1e-6

    # Target mass density from number density and mean atomic mass
    A_mean = np.sum(X_arr * A_arr)  # weighted mean A
    rho = n_cl_total * A_mean * AMU_G  # g/cm^3

    # --- Bohr effective charge (G&S eq. 7.11) ---
    # For v < v₀ = αc = c/137 the ion has not yet shed bound electrons; Z_eff < Z.
    beta_0 = 1.0 / 137.0
    Z_eff_sq = np.where(
        beta_arr < beta_0,
        float(Z_proj)**2 * (beta_arr / beta_0) ** (2.0 / 3.0),
        float(Z_proj)**2,
    )

    # T_max ~ 2 m_e c^2 beta^2 gamma^2 (heavy projectile limit, M ≫ m_e)
    T_max = 2.0 * m_e * beta2 * gamma_arr**2  # MeV

    arg = np.maximum(2.0 * m_e * beta2 * gamma_arr**2 * T_max / I_mean_MeV**2, 1.0 + 1e-15)
    BB = 0.5 * np.log(arg) - beta2
    BB = np.maximum(BB, 0.0)

    dEdx = K * Z_eff_sq * ZoverA * rho * (1.0 / np.maximum(beta2, 1e-30)) * BB
    return np.maximum(dEdx, 0.0)


def stopping_power(
    A_cl: np.ndarray,
    X_cl: np.ndarray,
    X_ion: float,
    Z_proj: int,
    E: np.ndarray | float,
    A_proj: int,
    n_e: float,
    T_e: float,
    eps: float = 1e-2,
) -> np.ndarray:
    """
    Stopping power epsilon(E) [MeV/cm].

    Total stopping = ionized-plasma contribution (``energy_loss_rate / v``)
    weighted by the ionized fraction + neutral-atom Bethe-Bloch contribution
    weighted by the neutral fraction.  For BLR clouds with X_ion ~ 0.001 the
    neutral Bethe-Bloch term dominates by three orders of magnitude.

    Returns
    -------
    np.ndarray
        Positive stopping power [MeV/cm].
    """
    E = np.asarray(E, dtype=float)

    # --- Plasma (free-electron) stopping ---
    dEdt = energy_loss_rate(
        A_cl=A_cl,
        X_cl=X_cl,
        X_ion=X_ion,
        Z_proj=Z_proj,
        E=E,
        A_proj=A_proj,
        n_e=n_e,
        T_e=T_e,
        eps=eps,
    )
    c_cm_s = c * 1e2
    M_proj = A_proj * AMU_TO_MEV
    gamma = 1.0 + E / M_proj
    beta_arr = np.sqrt(np.maximum(1.0 - 1.0 / gamma**2, 1e-30))
    v = beta_arr * c_cm_s
    eps_plasma = dEdt / np.maximum(v, 1e-300)

    # --- Neutral-atom Bethe-Bloch stopping ---
    # n_total = n_e / X_ion for X_ion > 0, else estimate from context.
    # The neutral fraction is (1 - X_ion) of the total baryon density.
    X_ion_clamped = float(np.clip(X_ion, 1e-9, 1.0))
    n_total = n_e / X_ion_clamped           # total baryon density [cm^-3]
    n_neutral = n_total * (1.0 - X_ion_clamped)
    eps_neutral = _bethe_bloch_neutral(
        A_cl=A_cl,
        X_cl=X_cl,
        Z_proj=Z_proj,
        A_proj=A_proj,
        E=E,
        n_cl_total=n_neutral,
    )

    epsilon_arr = X_ion_clamped * eps_plasma + (1.0 - X_ion_clamped) * eps_neutral
    epsilon_arr = np.nan_to_num(epsilon_arr, nan=0.0, posinf=0.0, neginf=0.0)
    epsilon_arr = np.maximum(epsilon_arr, 0.0)
    return epsilon_arr


# ---------------------------------------------------------------------
# Bin-grid helpers
# ---------------------------------------------------------------------

def bin_centers_from_edges(E_edges: np.ndarray) -> np.ndarray:
    """
    Energy bin centers from edges.
    """
    E_edges = np.asarray(E_edges, dtype=float)
    if E_edges.ndim != 1 or len(E_edges) < 2:
        raise ValueError("E_edges must be a 1D array of length >= 2.")
    return 0.5 * (E_edges[:-1] + E_edges[1:])


def bin_widths_from_edges(E_edges: np.ndarray) -> np.ndarray:
    """
    Energy bin widths from edges.
    """
    E_edges = np.asarray(E_edges, dtype=float)
    if E_edges.ndim != 1 or len(E_edges) < 2:
        raise ValueError("E_edges must be a 1D array of length >= 2.")
    dE = np.diff(E_edges)
    if np.any(dE <= 0.0):
        raise ValueError("E_edges must be strictly increasing.")
    return dE


def stopping_power_bin_average(
    A_cl: np.ndarray,
    X_cl: np.ndarray,
    X_ion: float,
    Z_proj: int,
    E_edges: np.ndarray,
    A_proj: int,
    n_e: float,
    T_e: float,
    n_sub: int = 8,
    eps: float = 1e-2,
) -> np.ndarray:
    """
    Bin-averaged stopping power epsilon_k for each energy bin.

    Parameters
    ----------
    E_edges : array, shape (n_bins + 1,)
        Bin edges in MeV.
    n_sub : int
        Number of sub-samples used for averaging epsilon(E) in each bin.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Bin-averaged stopping power [MeV/cm].
    """
    E_edges = np.asarray(E_edges, dtype=float)
    n_bins = len(E_edges) - 1
    eps_bin = np.zeros(n_bins, dtype=float)

    for k in range(n_bins):
        E_lo = E_edges[k]
        E_hi = E_edges[k + 1]

        E_sample = np.linspace(E_lo, E_hi, n_sub + 2)[1:-1]
        vals = stopping_power(
            A_cl=A_cl,
            X_cl=X_cl,
            X_ion=X_ion,
            Z_proj=Z_proj,
            E=E_sample,
            A_proj=A_proj,
            n_e=n_e,
            T_e=T_e,
            eps=eps,
        )
        eps_bin[k] = np.mean(vals)

    return eps_bin


def energy_loss_rate_bin_average(
    A_cl: np.ndarray,
    X_cl: np.ndarray,
    X_ion: float,
    Z_proj: int,
    E_edges: np.ndarray,
    A_proj: int,
    n_e: float,
    T_e: float,
    n_sub: int = 8,
    eps: float = 1e-2,
) -> np.ndarray:
    """
    Bin-averaged dE/dt for each energy bin.
    """
    E_edges = np.asarray(E_edges, dtype=float)
    n_bins = len(E_edges) - 1
    loss_bin = np.zeros(n_bins, dtype=float)

    for k in range(n_bins):
        E_lo = E_edges[k]
        E_hi = E_edges[k + 1]
        E_sample = np.linspace(E_lo, E_hi, n_sub + 2)[1:-1]
        vals = energy_loss_rate(
            A_cl=A_cl,
            X_cl=X_cl,
            X_ion=X_ion,
            Z_proj=Z_proj,
            E=E_sample,
            A_proj=A_proj,
            n_e=n_e,
            T_e=T_e,
            eps=eps,
        )
        loss_bin[k] = np.mean(vals)

    return loss_bin