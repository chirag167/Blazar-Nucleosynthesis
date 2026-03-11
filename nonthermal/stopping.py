# nonthermal/stopping.py
"""
nonthermal.stopping

Stopping power ε(E) and continuous energy loss.

Famiano (2002) uses "fast and slow ion formulas" (citing Ginzburg & Syrovatskii 1964)
and notes that a 4He projectile of 400 MeV has a range of ~1e11 cm in a medium with
electron density ~1e11 cm^-3 (paper text).

Because the user does not have the original reference and wants a debuggable starting
point, this module implements a *surrogate* stopping power model:

    ε(E, n_e) = ε_ref * (n_e / n_e_ref)

with ε_ref calibrated using a reference "range" at (E_ref, n_e_ref).

Later upgrades
--------------
When you move beyond exact reproduction tests or adopt published stopping-power tables,
consider using NIST PSTAR/ASTAR/ESTAR tables and/or a Bethe–Bloch implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from utils import *
import json

# mean excitation potentials for Z <=12 
# Source: Seltzer & Berger, Int. J. Appl. Radiat. Isot. Vol. 35, No, 7, pp. 665-676, 1984
I_vals = {
'H': 19.2, 'He': 41.8, 'Li': 40, 'Be': 63.7,
'C': 78, 'N': 82, 'O': 95, 'F': 115,
'Ne': 137, 'Na': 149, 'Mg': 156
}

def mean_ex_potential(Z, A):
    """
    Returns the mean excitation potential of the cloud in MeV.
    
    Inputs:
    Z : array of atomic numbers
    A : array of mass numbers
    """

    with open("cloud_composition.json") as f:
        cloud_composition = json.load(f)

    I_list = []
    weights = []

    for i in range(len(cloud_composition['elements'])):

        element = cloud_composition['elements'][i]

        # mean excitation potential
        if element in I_vals:
            I_val = I_vals[element]
        else:
            I_val = (9.76 + 58.8 * Z[i]**(-1.19)) * Z[i]

        I_list.append(I_val)

        # composition weight
        weights.append(cloud_composition['weights'][i] * Z[i] / A[i])

    I_arr = np.array(I_list)
    weights = np.array(weights)

    I_avg = np.exp(np.sum(weights * np.log(I_arr)) / np.sum(weights)) * 1e-6 # MeV

    return I_avg

# def stopping_power(N,Z,E,A,I):

#     '''Calculation of stopping power for a beam of electrons, deuterons, alpha particles, postirons, and negatrons (beta-) particles.
#     Inputs:
#     N = number density of the cloud.
#     Z = effective atomic number of the cloud.
#     E = incoming energy of the beam (MeV)
#     A = atomic mass number of the particles in the beam
#     I = mean excitation potential of the cloud.

#     Source: Michael F. L' Annunziata, NUCLEAR RADIATION, ITS INTERACTION WITH MATTER AND RADIOISOTOPE DECAY, pg. 87
    
#     Returns: The stopping power (epsilon) in units of MeV/m'''

#     beta = utils.beta(E,A)

#     gamma = utils.lorentz_factor(beta)

#     r_e = utils.r_e

#     m_e  = utils.m_e

#     I = mean_ex_potential(Z,A)

#     with open("jet_composition.json") as f:
#         jet_composition = json.load(f)

#     frac_p = jet_composition.get('p', 0)
#     frac_d = jet_composition.get('d', 0)
#     frac_a = jet_composition.get('a', 0)
#     frac_e = jet_composition.get('e', 0)
#     frac_bp = jet_composition.get('b+', 0)
    
#     epsilon = 0
   
#     epsilon += (4 * np.pi * r_e**2 * 1 * m_e/beta**2 * N * Z * (np.log(2*m_e**2/I * beta**2 * gamma**2) - beta**2)) * (frac_p + frac_d)

    
#     epsilon += (4 * np.pi * r_e**2 * 4 * m_e/beta**2 * N * Z * (np.log(2*m_e**2/I * beta**2 * gamma**2) - beta**2)) * frac_a
    

#     epsilon += (4 * np.pi * r_e**2 * m_e/beta**2 * N * Z * (np.log(beta * gamma * np.sqrt(gamma - 1) / I * m_e) + 1/(2*gamma**2) * ((gamma - 1)**2/8 + 1 - (gamma**2 + 2*gamma - 1)*np.log(2)))) * frac_e


#     epsilon += (4 * np.pi * r_e**2 * m_e/beta**2 * N * Z * (np.log(beta * gamma * np.sqrt(gamma - 1) / I * m_e) + beta**2/24 * (23 + 14/(gamma+1) + 10/(gamma+1)**2 + 4/(gamma+1)**3) + np.log(2)/2) ) * frac_bp

#     return epsilon


def energy_loss(A_cl, X_cl, X_ion, Z, E, A, n_e, dt, T_e, eps=1e-2):
    """
    Calculates the energy loss of the beam as it travels through the cloud. 

    Inputs: 
    A_cl = array of all the mass numbers present in the cloud. 
    X_cl = fraction of each element in the cloud. 
    X_ion = ionization fraction in the cloud. 
    Z = atomic numbers of the elements in the cloud. 
    E = array of energies of the beam. 
    A = atomic mass number of the particles in the beam. 
    n_e = concentration of all atomic electrons in the cloud (1/cm^-3). 
    dt = timescale of energy loss (in s) -- determined from Euler step
    T_e = electron temperature 
    
    Source: Ginzburg & Syrovatskii, The Origin of Cosmic Rays, 1964. 
    
    Returns: The energy loss of the beam as it traverses through the cloud.
    """

    E = np.asarray(E)

    beta = utils.beta(E, A)

    A_arr = np.array(A_cl)
    X_arr = np.array(X_cl)

    A_eff = np.sum(A_arr * X_arr)

    M = A_eff * utils.mass_num_to_mev   # MeV/c^2
    m_e = utils.m_e
    c = utils.c

    kappa = 7.62e-9
    E_c = 3.7e-11 * np.sqrt(n_e) * 1e-6  # MeV

    dEdt = np.zeros_like(E)

    # -------- Regime masks --------

    mask1 = E < eps * (M/m_e) * M
    mask2 = (E >= eps * (M/m_e) * M) & (E < eps * M)
    mask3 = (E >= 1/eps * M) & (E < eps * (M/m_e) * M)
    mask4 = E >= 1/eps * (M/m_e) * M

    mask_else = ~(mask1 | mask2 | mask3 | mask4)

    # -------- Regime 1 --------

    if np.any(mask1):
        dEdt[mask1] = (
            kappa * Z**2 * n_e * (1/beta[mask1]) *
            (22.2 + 4*np.log(E[mask1]/m_e) +
             2*np.log(beta[mask1]**2) - 2*beta[mask1]**2)
        ) * 1e-6

    # -------- Regime 2 --------

    if np.any(mask2):
        dEdt[mask2] = (
            kappa * Z**2 * n_e *
            np.sqrt(2 * M / E_c) *
            (11.8 + np.log(E[mask2] / M))
        ) * 1e-6

    # -------- Regime 3 --------

    if np.any(mask3):
        dEdt[mask3] = (
            kappa * Z**2 * n_e *
            (4 * np.log(E[mask3] / M) + 20.2)
        ) * 1e-6

    # -------- Regime 4 --------

    if np.any(mask4):
        dEdt[mask4] = (
            kappa * Z**2 * n_e *
            (3*np.log(E[mask4] / M) +
             np.log(M / m_e) + 19.5)
        ) * 1e-6

    # -------- Neutral / Ionized regime --------

    if np.any(mask_else):

        Z_c = np.sum(Z * X_arr/A_arr) / np.sum(X_arr/A_arr)

        v = beta[mask_else] * c * 1e2  # cm/s

        dEdt_neutral = 2.34e-23 * n_e * (Z + Z_c) * v**2 * 1e-6
        dEdt_ion = 1.8e-12 * Z**2 * n_e / (A * T_e**1.5) * 1e-6

        dEdt_mix = X_ion * dEdt_ion + (1 - X_ion) * dEdt_neutral

        dEdt[mask_else] = dEdt_mix

    # -------- Euler step --------

    E_new = E - dEdt * dt

    return dEdt, E_new
    

def stopping_power(A_cl, X_cl, X_ion, Z, E, A, n_e, dt, T_e, eps=1e-2):

    dEdt, E = energy_loss(A_cl, X_cl, X_ion, Z, E, A, n_e, dt, T_e, eps=1e-2)

    c = utils.c * 1e2   # convert m/s → cm/s if utils.c is SI
    M = A * utils.mass_num_to_mev  # rest mass energy (MeV)

    gamma = 1 + E / M

    beta = np.sqrt(1 - 1 / gamma**2)

    v = beta * c # cm /s

    epsilon = 1/v * dEdt # MeV/cm

    return epsilon


    

# @dataclass(frozen=True)
# class IonizationModel:
#     """
#     Ionization fraction model x_ion(t).

#     Default: constant scalar (x0).
#     Optional: linear ramp from x0 to x1 over t_ramp_s.
#     """
#     x0: float = 0.0
#     x1: Optional[float] = None
#     t_ramp_s: Optional[float] = None

#     def x(self, t_s: float) -> float:
#         if self.x1 is None or self.t_ramp_s is None or self.t_ramp_s <= 0:
#             return float(np.clip(self.x0, 0.0, 1.0))
#         t = max(0.0, float(t_s))
#         frac = min(1.0, t / float(self.t_ramp_s))
#         return float(np.clip(self.x0 + frac * (self.x1 - self.x0), 0.0, 1.0))


# @dataclass(frozen=True)
# class MediumPrimordial:
#     """
#     Primordial (H/He) cloud medium.

#     nH_cm3: hydrogen nuclei number density [cm^-3]
#     nHe_cm3: helium nuclei number density [cm^-3]
#     ionization: controls fraction of *fully-ionized* electron density:
#         n_e,full = nH + 2*nHe
#         n_e(t) = x_ion(t) * n_e,full
#     """
#     nH_cm3: float
#     nHe_cm3: float
#     ionization: IonizationModel

#     def electron_density_cm3(self, t_s: float) -> float:
#         ne_full = self.nH_cm3 + 2.0 * self.nHe_cm3
#         return self.ionization.x(t_s) * ne_full

#     def target_density_cm3(self, species: str) -> float:
#         s = species.strip()
#         if s in ("H", "H1", "p"):
#             return self.nH_cm3
#         if s in ("He", "He4", "alpha"):
#             return self.nHe_cm3
#         return 0.0  # primordial-only default


# @dataclass(frozen=True)
# class CalibratedStoppingPower:
#     """
#     Surrogate stopping model calibrated by a reference range.

#     eps_ref = E_ref / range_ref
#     eps(E, n_e) = eps_ref * (n_e / n_e_ref)

#     Returns eps in MeV/cm.
#     """
#     E_ref_mev: float
#     n_e_ref_cm3: float
#     range_ref_cm: float

#     def __post_init__(self) -> None:
#         if self.E_ref_mev <= 0 or self.n_e_ref_cm3 <= 0 or self.range_ref_cm <= 0:
#             raise ValueError("Invalid calibration parameters.")

#     @property
#     def eps_ref_mev_per_cm(self) -> float:
#         return self.E_ref_mev / self.range_ref_cm

#     def stopping_power_mev_per_cm(self, *, E_mev: float, n_e_cm3: float) -> float:
#         if E_mev < 0:
#             raise ValueError("E_mev must be >= 0.")
#         if n_e_cm3 < 0:
#             raise ValueError("n_e_cm3 must be >= 0.")
#         if n_e_cm3 == 0:
#             # Surrogate assumption: no Coulomb/ionization loss when no free electrons.
#             return 0.0
#         return self.eps_ref_mev_per_cm * (n_e_cm3 / self.n_e_ref_cm3)

#     def energy_after_path_mev(self, *, E0_mev: float, n_e_cm3: float, L_cm: float) -> float:
#         if L_cm < 0:
#             raise ValueError("L_cm must be >= 0.")
#         eps = self.stopping_power_mev_per_cm(E_mev=E0_mev, n_e_cm3=n_e_cm3)
#         return max(0.0, E0_mev - eps * L_cm)

#     def range_cm(self, *, E0_mev: float, n_e_cm3: float) -> float:
#         eps = self.stopping_power_mev_per_cm(E_mev=E0_mev, n_e_cm3=n_e_cm3)
#         if eps <= 0:
#             return float("inf")
#         return E0_mev / eps
