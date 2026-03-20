"""
grids.py

Defines the discretized energy grid used by the non-thermal cascade solver.

Purpose
-------
This module provides a reusable representation of the projectile energy grid
used throughout the codebase. In the Famiano-style non-thermal reaction network,
particles are tracked in discrete energy intervals, so the solver needs a
consistent way to define:

- energy bin edges
- bin centers
- bin widths
- indexing helpers
- validation checks

This module does NOT evolve the system and does NOT compute reaction rates.
It only constructs and manages the numerical grid on which those calculations
are performed.

Main output
-----------
The main output of this module is an EnergyGrid object, which can be passed to
state.py, engine.py, and any operator modules that require knowledge of the
energy discretization.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EnergyGrid:
    """
    Container for a 1D energy grid.

    Parameters
    ----------
    edges : np.ndarray
        Monotonically increasing array of bin edges with shape (n_bins + 1,).

    Notes
    -----
    If edges = [E0, E1, E2, ..., En], then the bins are:
        [E0, E1), [E1, E2), ..., [E_{n-1}, E_n)

    The solver can interpret the i-th bin as the interval:
        E_i <= E < E_{i+1}
    """

    edges: np.ndarray

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=float)

        if edges.ndim != 1:
            raise ValueError("EnergyGrid.edges must be a 1D array.")
        if len(edges) < 2:
            raise ValueError("EnergyGrid requires at least 2 edges.")
        if np.any(edges <= 0.0):
            raise ValueError("All energy grid edges must be positive.")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("Energy grid edges must be strictly increasing.")

        object.__setattr__(self, "edges", edges)

    @property
    def n_bins(self) -> int:
        """Number of energy bins."""
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        """Arithmetic bin centers."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def geometric_centers(self) -> np.ndarray:
        """Geometric bin centers, useful for log-spaced grids."""
        return np.sqrt(self.edges[:-1] * self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        """Bin widths ΔE_i = E_{i+1} - E_i."""
        return self.edges[1:] - self.edges[:-1]

    @property
    def log_widths(self) -> np.ndarray:
        """Logarithmic bin widths ΔlnE_i."""
        return np.log(self.edges[1:] / self.edges[:-1])

    @property
    def e_min(self) -> float:
        """Minimum energy on the grid."""
        return float(self.edges[0])

    @property
    def e_max(self) -> float:
        """Maximum energy on the grid."""
        return float(self.edges[-1])

    def contains(self, energy: float) -> bool:
        """
        Return True if the energy lies within the grid bounds.
        """
        return self.e_min <= energy <= self.e_max

    def find_bin(self, energy: float, clip: bool = False) -> int:
        """
        Find the bin index containing a given energy.

        Parameters
        ----------
        energy : float
            Energy value to locate.
        clip : bool, optional
            If True, energies outside the grid are clipped to the nearest bin.
            If False, a ValueError is raised for out-of-range energies.

        Returns
        -------
        int
            Bin index i such that edges[i] <= energy < edges[i+1],
            except at the upper boundary where energy == edges[-1] is assigned
            to the last bin.
        """
        energy = float(energy)

        if energy < self.e_min:
            if clip:
                return 0
            raise ValueError(f"Energy {energy} is below grid minimum {self.e_min}.")

        if energy > self.e_max:
            if clip:
                return self.n_bins - 1
            raise ValueError(f"Energy {energy} is above grid maximum {self.e_max}.")

        idx = int(np.searchsorted(self.edges, energy, side="right") - 1)

        if idx == self.n_bins:
            idx = self.n_bins - 1

        return idx

    def bin_interval(self, i: int) -> tuple[float, float]:
        """
        Return the (left_edge, right_edge) of bin i.
        """
        if i < 0 or i >= self.n_bins:
            raise IndexError(f"Bin index {i} out of bounds for {self.n_bins} bins.")
        return float(self.edges[i]), float(self.edges[i + 1])

    def as_dict(self) -> dict:
        """
        Serialize the grid to a JSON-friendly dictionary.
        """
        return {
            "edges": self.edges.tolist(),
            "n_bins": self.n_bins,
            "e_min": self.e_min,
            "e_max": self.e_max,
        }


def make_log_energy_grid(e_min: float, e_max: float, n_bins: int) -> EnergyGrid:
    """
    Construct a logarithmically spaced energy grid.

    Parameters
    ----------
    e_min : float
        Minimum energy (> 0).
    e_max : float
        Maximum energy (> e_min).
    n_bins : int
        Number of bins.

    Returns
    -------
    EnergyGrid
        Logarithmically spaced energy grid.
    """
    if e_min <= 0.0:
        raise ValueError("e_min must be > 0 for a logarithmic grid.")
    if e_max <= e_min:
        raise ValueError("e_max must be greater than e_min.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    edges = np.logspace(np.log10(e_min), np.log10(e_max), n_bins + 1)
    return EnergyGrid(edges=edges)


def make_linear_energy_grid(e_min: float, e_max: float, n_bins: int) -> EnergyGrid:
    """
    Construct a linearly spaced energy grid.

    Parameters
    ----------
    e_min : float
        Minimum energy.
    e_max : float
        Maximum energy (> e_min).
    n_bins : int
        Number of bins.

    Returns
    -------
    EnergyGrid
        Linearly spaced energy grid.
    """
    if e_max <= e_min:
        raise ValueError("e_max must be greater than e_min.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    edges = np.linspace(e_min, e_max, n_bins + 1)
    return EnergyGrid(edges=edges)


def make_energy_grid(config: dict) -> EnergyGrid:
    """
    Build an energy grid from a configuration dictionary.

    Expected keys
    -------------
    config["type"]   : "log" or "linear"
    config["e_min"]  : float
    config["e_max"]  : float
    config["n_bins"] : int

    Example
    -------
    config = {
        "type": "log",
        "e_min": 1.0e-2,
        "e_max": 1.0e3,
        "n_bins": 200
    }
    """
    grid_type = str(config.get("type", "log")).strip().lower()
    e_min = float(config["e_min"])
    e_max = float(config["e_max"])
    n_bins = int(config["n_bins"])

    if grid_type == "log":
        return make_log_energy_grid(e_min=e_min, e_max=e_max, n_bins=n_bins)
    if grid_type == "linear":
        return make_linear_energy_grid(e_min=e_min, e_max=e_max, n_bins=n_bins)

    raise ValueError(
        f"Unsupported energy grid type '{grid_type}'. Use 'log' or 'linear'."
    )