"""
transport.spectra
=================

Defines particle energy spectrum models for non-thermal transport
calculations (e.g., blazar jet spectra).

All spectra are expected to implement:

    - phi(E)
    - phi_grid(E_grid)

This version does NOT use Python's abstract base class (ABC) mechanism.
Instead, it relies on a lightweight interface pattern where subclasses
override methods and base methods raise NotImplementedError.

Typical Φ(E) units:
    particles / (cm^2 s MeV)

Unit consistency is the user's responsibility.
"""

import numpy as np


class Spectrum:
    """
    Base class for particle spectra.

    This class defines the expected interface for spectrum objects.
    It is not meant to be instantiated directly.

    Any subclass should implement:
        - phi(E)
        - phi_grid(E_grid)
    """

    def phi(self, E):
        """
        Differential flux at scalar energy E.

        Parameters
        ----------
        E : float
            Energy value.

        Returns
        -------
        float
            Differential flux Φ(E).
        """
        # This forces subclasses to implement the method.
        raise NotImplementedError(
            "Subclasses of Spectrum must implement phi(E)."
        )

    def phi_grid(self, E_grid):
        """
        Differential flux evaluated on array of energies.

        Parameters
        ----------
        E_grid : np.ndarray
            Array of energies.

        Returns
        -------
        np.ndarray
            Flux values Φ(E).
        """
        # This forces subclasses to implement the method.
        raise NotImplementedError(
            "Subclasses of Spectrum must implement phi_grid(E_grid)."
        )


class PowerLawSpectrum(Spectrum):
    """
    Power-law spectrum:

        Φ(E) = norm * E^{-alpha}

    defined within energy bounds [Emin, Emax].

    Parameters
    ----------
    alpha : float
        Spectral index.
    Emin : float
        Minimum energy cutoff.
    Emax : float
        Maximum energy cutoff.
    norm : float
        Normalization constant.
    """

    def __init__(self, alpha, Emin, Emax, norm=1.0):
        """
        Initialize power-law spectrum.

        Parameters
        ----------
        alpha : float
            Spectral index.
        Emin : float
            Minimum energy.
        Emax : float
            Maximum energy.
        norm : float
            Normalization constant.
        """
        self.alpha = alpha
        self.Emin = Emin
        self.Emax = Emax
        self.norm = norm

    def phi(self, E):
        """
        Evaluate Φ(E) at scalar energy E.

        Returns zero outside the allowed energy range.
        """

        # If energy is outside bounds, flux is zero
        if E < self.Emin or E > self.Emax:
            return 0.0

        # Apply power-law formula
        return self.norm * E ** (-self.alpha)

    def phi_grid(self, E_grid):
        """
        Vectorized evaluation of Φ(E) over energy array.

        Parameters
        ----------
        E_grid : np.ndarray
            Energy array.

        Returns
        -------
        np.ndarray
            Array of Φ(E) values.
        """

        # Ensure input is numpy array
        E = np.asarray(E_grid)

        # Initialize output array with zeros
        phi_vals = np.zeros_like(E)

        # Create mask for energies inside allowed range
        mask = (E >= self.Emin) & (E <= self.Emax)

        # Apply power law only where mask is True
        phi_vals[mask] = self.norm * E[mask] ** (-self.alpha)

        return phi_vals
