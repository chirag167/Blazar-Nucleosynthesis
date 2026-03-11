"""
transport.cross_sections
========================

Defines energy-dependent nuclear cross section models σ(E)
for non-thermal beam-target transport calculations.

This module supports loading tabulated cross section files
with flexible unit handling.

Internal unit conventions:
    Energy -> MeV
    Cross section -> cm^2

Supported input units:
    Energy: keV, MeV, GeV
    Sigma:  b (barn), mb (millibarn), ub (microbarn)

Optional uncertainty column is supported.
"""

import numpy as np
import re


# ============================================================
# Unit conversion constants
# ============================================================

# Energy conversions to MeV
ENERGY_CONVERSIONS = {
    "keV": 1e-3,
    "MeV": 1.0,
    "GeV": 1e3,
}

# Cross section conversions to cm^2
# 1 barn = 1e-24 cm^2
SIGMA_CONVERSIONS = {
    "b": 1e-24,
    "mb": 1e-27,
    "ub": 1e-30,
    "microb": 1e-30,
}


# ============================================================
# Base class
# ============================================================

class CrossSection:
    """
    Base class for energy-dependent cross sections.
    """

    def sigma(self, E):
        raise NotImplementedError

    def sigma_grid(self, E_grid):
        raise NotImplementedError


# ============================================================
# Tabulated cross section
# ============================================================

class TabulatedCrossSection(CrossSection):
    """
    Cross section defined by tabulated data.

    Data is internally stored in:
        Energy: MeV
        Sigma: cm^2
    """

    def __init__(self, E_grid, sigma_values, d_sigma=None):
        """
        Parameters
        ----------
        E_grid : np.ndarray
            Energy values (MeV).
        sigma_values : np.ndarray
            Cross section values (cm^2).
        d_sigma : np.ndarray, optional
            Uncertainty values (cm^2).
        """
        self.E_grid = np.asarray(E_grid)
        self.sigma_values = np.asarray(sigma_values)
        self.d_sigma = None if d_sigma is None else np.asarray(d_sigma)

    # --------------------------------------------------------

    def sigma(self, E):
        """
        Interpolate σ(E) at scalar energy.
        """
        return np.interp(
            E,
            self.E_grid,
            self.sigma_values,
            left=0.0,
            right=0.0
        )

    # --------------------------------------------------------

    def sigma_grid(self, E_grid):
        """
        Interpolate σ(E) over array of energies.
        """
        E = np.asarray(E_grid)

        return np.interp(
            E,
            self.E_grid,
            self.sigma_values,
            left=0.0,
            right=0.0
        )

    # --------------------------------------------------------
    # Static loader
    # --------------------------------------------------------

    @staticmethod
    def from_file(filename):
        """
        Load cross section data from file.

        File formats supported:

        1) No header:
           Columns: E  sigma  [d_sigma]
           Assumes:
               E in MeV
               sigma in mb

        2) Header with unit encoding:
           E_[unit], sigma_[unit], [d_sigma_[unit]]

        Returns
        -------
        TabulatedCrossSection
        """

        # Read first line to inspect header
        with open(filename, "r") as f:
            first_line = f.readline().strip()

        # Detect header by presence of non-numeric characters
        has_header = any(c.isalpha() for c in first_line)

        if has_header:
            # Load with header
            data = np.genfromtxt(filename, names=True)

            # Extract column names
            columns = data.dtype.names

            # Identify energy column
            E_col = [c for c in columns if c.startswith("E")][0]
            sigma_col = [c for c in columns if c.startswith("sigma")][0]

            # Extract units from header using regex
            E_unit = re.search(r"\[(.*?)\]", E_col)
            sigma_unit = re.search(r"\[(.*?)\]", sigma_col)

            E_unit = E_unit.group(1) if E_unit else "MeV"
            sigma_unit = sigma_unit.group(1) if sigma_unit else "mb"

            # Convert energy to MeV
            E = data[E_col] * ENERGY_CONVERSIONS[E_unit]

            # Convert sigma to cm^2
            sigma = data[sigma_col] * SIGMA_CONVERSIONS[sigma_unit]

            # Optional uncertainty column
            d_sigma = None
            for c in columns:
                if c.startswith("d_sigma"):
                    d_sigma = data[c] * SIGMA_CONVERSIONS[sigma_unit]

            return TabulatedCrossSection(E, sigma, d_sigma)

        else:
            # No header: assume default units
            data = np.loadtxt(filename)

            E = data[:, 0]  # assumed MeV
            sigma = data[:, 1] * SIGMA_CONVERSIONS["mb"]

            d_sigma = None
            if data.shape[1] > 2:
                d_sigma = data[:, 2] * SIGMA_CONVERSIONS["mb"]

            return TabulatedCrossSection(E, sigma, d_sigma)
