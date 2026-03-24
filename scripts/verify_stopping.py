"""
verify_stopping.py

Compare _bethe_bloch_neutral() output against NIST PSTAR reference data
for protons in hydrogen gas.

NIST PSTAR data (I = 19.2 eV, hydrogen gas):
    https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html

Run from the project root:
    python scripts/verify_stopping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.stopping import _bethe_bloch_neutral, stopping_power

# ---------------------------------------------------------------------------
# NIST PSTAR reference: electronic stopping power S_el/rho [MeV cm^2/g]
# for protons in liquid/gas hydrogen (I = 19.2 eV).
# ---------------------------------------------------------------------------
NIST_E_MEV = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0])
NIST_S_MeV_cm2_g = np.array([676.4, 388.5, 182.5, 101.9, 56.79, 26.49, 15.30, 9.328, 6.238])

# ---------------------------------------------------------------------------
# Cloud parameters: pure hydrogen, neutral
# ---------------------------------------------------------------------------
A_cl = np.array([1.0])   # hydrogen
X_cl = np.array([1.0])   # 100% hydrogen
Z_proj = 1               # proton
A_proj = 1

# Pure neutral hydrogen: n_total such that density rho = some reference.
# For comparison to NIST mass stopping power (S/rho), we can use any density
# since S/rho is density-independent.  Choose n_total = 1 cm^-3 and recover
# S/rho by dividing dE/dx by rho.
AMU_G = 1.66054e-24      # g
n_total = 1.0            # cm^-3  (will cancel in S/rho)
rho_ref = n_total * 1.0 * AMU_G   # g/cm^3

# ---------------------------------------------------------------------------
# Compute neutral Bethe-Bloch stopping for our grid
# ---------------------------------------------------------------------------
dEdx_arr = _bethe_bloch_neutral(
    A_cl=A_cl,
    X_cl=X_cl,
    Z_proj=Z_proj,
    A_proj=A_proj,
    E=NIST_E_MEV,
    n_cl_total=n_total,
)

# Convert dE/dx [MeV/cm] → S/rho [MeV cm^2/g]
S_calc = dEdx_arr / rho_ref

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
header = f"{'E (MeV)':>10}  {'NIST S/ρ':>14}  {'Calc S/ρ':>14}  {'Ratio':>8}  {'% err':>7}"
print(header)
print("-" * len(header))
for E, S_ref, S_c in zip(NIST_E_MEV, NIST_S_MeV_cm2_g, S_calc):
    ratio = S_c / S_ref
    pct = (S_c / S_ref - 1.0) * 100.0
    print(f"{E:>10.1f}  {S_ref:>14.3f}  {S_c:>14.3f}  {ratio:>8.4f}  {pct:>+7.2f}%")

rms_pct = np.sqrt(np.mean(((S_calc / NIST_S_MeV_cm2_g) - 1.0)**2)) * 100.0
print(f"\nRMS percentage error: {rms_pct:.2f}%")

# ---------------------------------------------------------------------------
# Cross-check stopping_power() in fully-neutral mode.
# stopping_power() derives n_total = n_e / X_ion internally, so set
# n_e = n_total * X_ion to keep the same baryon density as above.
# ---------------------------------------------------------------------------
print("\n--- stopping_power() cross-check (X_ion=1e-9, consistent n_e) ---")
X_ion_xc = 1e-9                      # nearly fully neutral
n_e_xc = n_total * X_ion_xc          # n_e consistent with n_total = 1 cm^-3

eps_total = stopping_power(
    A_cl=A_cl,
    X_cl=X_cl,
    X_ion=X_ion_xc,
    Z_proj=Z_proj,
    E=NIST_E_MEV,
    A_proj=A_proj,
    n_e=n_e_xc,
    T_e=1e4,
)
# Inside stopping_power(): n_neutral = (n_e/X_ion)*(1-X_ion) ≈ n_total
# so rho_neutral ≈ rho_ref defined above.
S_total = eps_total / rho_ref

print(f"{'E (MeV)':>10}  {'NIST S/ρ':>14}  {'Total S/ρ':>14}  {'% err':>7}")
for E, S_ref, St in zip(NIST_E_MEV, NIST_S_MeV_cm2_g, S_total):
    pct = (St / S_ref - 1.0) * 100.0
    print(f"{E:>10.1f}  {S_ref:>14.3f}  {St:>14.3f}  {pct:>+7.2f}%")
