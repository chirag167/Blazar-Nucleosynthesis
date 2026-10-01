"""
tau_loader.py

Loads the product-energy distribution tensor tau from the per-reaction
E_bins/ CSV files produced by fresco_code/scripts/energy_bins_reaction_dir.py.

Tensor shape: (n_rxn, n_proj_bins, n_products, n_prod_bins)

For fixed (rxn, proj_bin, product), tau[r, k, p, :] is the normalised
energy distribution eta, so sum_q tau[r, k, p, q] == 1 wherever data exist.
Bins with no matching CSV file are left as zero.

This is the phi tensor of Famiano et al. (2002) Eq. (11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reactions import ProductDistributionModel

# Matches the suffix  _{E_label}MeV_{product}_Ebins.csv
# E_label is produced by safe_energy_label(): integer energies give "5MeV",
# fractional energies give "5p5MeV" (decimal point replaced by 'p').
_EBINS_SUFFIX_RE = re.compile(
    r"_([0-9]+(?:p[0-9]+)?)MeV_([A-Za-z0-9]+)_Ebins\.csv$",
    re.IGNORECASE,
)


def _label_to_mev(label: str) -> float:
    """Convert an E_label string back to a float in MeV."""
    return float(label.replace("p", "."))


def _parse_eta_csv(path: Path) -> np.ndarray:
    """Return the eta column from an E_bins CSV as a 1-D float array."""
    eta: list[float] = []
    with open(path, newline="") as fh:
        header: list[str] | None = None
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cols = [c.strip() for c in line.split(",")]
            if header is None:
                header = [c.lower() for c in cols]
                continue
            row = dict(zip(header, cols))
            try:
                eta.append(float(row["eta"]))
            except (KeyError, ValueError):
                eta.append(0.0)
    return np.asarray(eta, dtype=float)


def _energy_to_bin(E_mev: float, edges: np.ndarray) -> int | None:
    """Return bin index k such that edges[k] <= E < edges[k+1].

    Accepts a value exactly at the upper boundary (i.e. the last bin).
    """
    k = int(np.searchsorted(edges, E_mev, side="right")) - 1
    n = len(edges) - 1
    if 0 <= k < n:
        return k
    # Allow a small overshoot at the top edge
    if k == n and abs(E_mev - edges[-1]) < 0.5:
        return n - 1
    return None


def load_tau_tensor(
    e_bins_root: Path | str,
    reactions: list[str],
    energy_edges_mev: np.ndarray,
    product_species: list[str],
    infer_recoil: "dict[str, tuple[str, str, float]] | None" = None,
) -> np.ndarray:
    """Build the product-energy distribution tensor tau.

    Parameters
    ----------
    e_bins_root:
        Root directory that contains ``<reaction>/E_bins/`` subdirectories.
        Typically ``fresco_code/newruns/``.
    reactions:
        Ordered reaction keys, e.g. ``["he4_he4_n_be7", "he4_he4_p_li7"]``.
        Determines axis 0 of tau.
    energy_edges_mev:
        Shared projectile/product energy bin edges in MeV, length n_bins+1.
        Example: ``np.arange(0, 405, 5)`` for 80 bins at 5 MeV step.
    product_species:
        Ordered product species names, e.g. ``["n", "p", "d", "t", "he3"]``.
        Determines axis 2 of tau.
    infer_recoil : dict, optional
        For reactions that only have FRESCO data for the light ejectile, infer
        the heavy-recoil energy distribution from two-body energy conservation::

            T_recoil = T_proj_bin_center + Q - T_ejectile_bin_center

        Map: ``reaction_name -> (ejectile_key, recoil_key, Q_MeV)`` where keys
        are lowercase (e.g. ``"n"``, ``"be7"``).  Q < 0 for endothermic reactions.

    Returns
    -------
    np.ndarray, shape (n_rxn, n_proj_bins, n_products, n_prod_bins)
        tau[r, k, p, q] = fraction of product p landing in product-energy
        bin q when reaction r fires with projectile in energy bin k.
        Zero where no CSV file exists for that (reaction, energy, product).
    """
    e_bins_root = Path(e_bins_root)
    edges = np.asarray(energy_edges_mev, dtype=float)
    n_proj_bins = len(edges) - 1
    n_prod_bins = n_proj_bins   # product grid reuses the same axis
    n_rxn = len(reactions)
    n_products = len(product_species)

    tau = np.zeros((n_rxn, n_proj_bins, n_products, n_prod_bins), dtype=float)

    rxn_index = {r: i for i, r in enumerate(reactions)}
    prod_index = {p: i for i, p in enumerate(product_species)}

    for rxn_name, r_idx in rxn_index.items():
        ebins_dir = e_bins_root / rxn_name / "E_bins"
        if not ebins_dir.is_dir():
            continue

        for csv_path in sorted(ebins_dir.glob("*_Ebins.csv")):
            m = _EBINS_SUFFIX_RE.search(csv_path.name)
            if m is None:
                continue

            try:
                E_proj = _label_to_mev(m.group(1))
            except ValueError:
                continue

            product = m.group(2).lower()
            if product not in prod_index:
                continue

            k = _energy_to_bin(E_proj, edges)
            if k is None:
                continue

            p_idx = prod_index[product]

            eta = _parse_eta_csv(csv_path)

            # Pad or truncate to match the grid
            if len(eta) != n_prod_bins:
                padded = np.zeros(n_prod_bins, dtype=float)
                n = min(len(eta), n_prod_bins)
                padded[:n] = eta[:n]
                eta = padded

            # Renormalise defensively (eta should already sum to 1)
            s = eta.sum()
            if s > 0.0:
                eta = eta / s

            tau[r_idx, k, p_idx, :] = eta

    # -----------------------------------------------------------------------
    # Kinematic inference of heavy-recoil distributions.
    # For each (rxn, proj_bin), derive T_recoil = T_proj_center + Q - T_ej_center.
    # This is exact for 2-body reactions where energy conservation holds
    # event-by-event, so the angle-integrated distributions satisfy this transform.
    # -----------------------------------------------------------------------
    if infer_recoil:
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        for rxn_name, (ej_key, recoil_key, Q_mev) in infer_recoil.items():
            if rxn_name not in rxn_index:
                continue
            r = rxn_index[rxn_name]
            p_ej = prod_index.get(ej_key)
            p_rec = prod_index.get(recoil_key)
            if p_ej is None or p_rec is None:
                continue

            for k in range(n_proj_bins):
                eta_ej = tau[r, k, p_ej, :]
                if eta_ej.sum() == 0.0:
                    continue
                T_avail = bin_centers[k] + Q_mev

                eta_rec = np.zeros(n_prod_bins, dtype=float)
                for q_ej in range(n_prod_bins):
                    if eta_ej[q_ej] == 0.0:
                        continue
                    T_rec = T_avail - bin_centers[q_ej]
                    if T_rec <= 0.0:
                        continue
                    q_rec = _energy_to_bin(T_rec, edges)
                    if q_rec is not None:
                        eta_rec[q_rec] += eta_ej[q_ej]

                s = eta_rec.sum()
                if s > 0.0:
                    tau[r, k, p_rec, :] = eta_rec / s

    return tau


# ---------------------------------------------------------------------------
# Species name normalisation
# ---------------------------------------------------------------------------

def _canonical_to_lower(name: str) -> str:
    """Normalise a species name to the lowercase folder-key format.

    Converts the canonical ``{number}{Symbol}`` form used in reactions.py
    to the ``{symbol}{number}`` lowercase form used in E_bins filenames:

        '4He' → 'he4',  '7Li' → 'li7',  '3He' → 'he3',  'n' → 'n'

    Already-lowercase names (e.g. 'he4') pass through unchanged.
    """
    m = re.match(r'^(\d+)([A-Za-z]+)$', name.strip())
    if m:
        return (m.group(2) + m.group(1)).lower()
    return name.strip().lower()


# ---------------------------------------------------------------------------
# ProductDistributionModel backed by the tau tensor
# ---------------------------------------------------------------------------

@dataclass
class TauProductModel(ProductDistributionModel):
    """
    ProductDistributionModel that looks up precomputed eta distributions
    from the tau tensor loaded by :func:`load_tau_tensor`.

    All models for a given run share a single tau array (no copies).
    Assign one instance per Group 1 ReactionChannel::

        rxn.product_distribution_model = models["he4_he4_n_be7"]

    Parameters
    ----------
    tau : np.ndarray, shape (n_rxn, n_proj_bins, n_products, n_prod_bins)
        Full tau tensor.
    rxn_index : int
        Index of this reaction in axis 0.
    prod_index : dict[str, int]
        Maps *lowercase* product species key (e.g. ``'he4'``) to axis-2 index.
    energy_edges_mev : np.ndarray, shape (n_proj_bins + 1,)
        Bin edges used when the tensor was built.
    """

    tau: np.ndarray = field(repr=False)
    rxn_index: int
    prod_index: dict
    energy_edges_mev: np.ndarray = field(repr=False)

    def distribution(
        self,
        *,
        projectile_energy_mev: float,
        product_species: str,
        product_energy_edges_mev: np.ndarray,
    ) -> np.ndarray:
        """
        Return the normalised product-energy distribution eta[q].

        Parameters
        ----------
        projectile_energy_mev : float
            Lab-frame projectile kinetic energy.
        product_species : str
            Canonical species name, e.g. ``'n'``, ``'4He'``, ``'3He'``.
        product_energy_edges_mev : np.ndarray
            Must have the same number of bins as the stored grid.

        Returns
        -------
        np.ndarray, shape (n_prod_bins,)
            Normalised distribution (sums to 1 where data exist, zeros
            where no E_bins CSV was found for this energy / product).
        """
        n_prod_bins = len(self.energy_edges_mev) - 1

        req_bins = len(product_energy_edges_mev) - 1
        if req_bins != n_prod_bins:
            raise ValueError(
                f"TauProductModel: caller passed {req_bins}-bin product grid "
                f"but tau was built on {n_prod_bins} bins."
            )

        k = _energy_to_bin(projectile_energy_mev, self.energy_edges_mev)
        if k is None:
            return np.zeros(n_prod_bins, dtype=float)

        key = _canonical_to_lower(product_species)
        p = self.prod_index.get(key)
        if p is None:
            return np.zeros(n_prod_bins, dtype=float)

        dist = self.tau[self.rxn_index, k, p, :].copy()
        s = dist.sum()
        if s > 0.0:
            dist /= s
        return dist


# ---------------------------------------------------------------------------
# Factory: build models for a set of reactions
# ---------------------------------------------------------------------------

def build_tau_models(
    e_bins_root: Path | str,
    reactions: list[str],
    energy_edges_mev: np.ndarray,
    product_species: list[str],
    infer_recoil: "dict[str, tuple[str, str, float]] | None" = None,
) -> dict[str, TauProductModel]:
    """
    Load the tau tensor and return one :class:`TauProductModel` per reaction.

    All returned models share the same underlying numpy array.

    Parameters
    ----------
    e_bins_root : Path
        Root directory containing ``<reaction>/E_bins/`` subdirectories
        (typically ``fresco_code/newruns/``).
    reactions : list[str]
        Ordered reaction folder names, e.g.
        ``["he4_he4_n_be7", "he4_he4_p_li7", "he4_p_d_he3", "li7_p_he4_he4"]``.
    energy_edges_mev : np.ndarray
        Common projectile/product bin edges (length n_bins + 1).
    product_species : list[str]
        Product species in any capitalisation; normalised internally.
        E.g. ``["n", "p", "d", "t", "3He", "4He", "7Li", "7Be"]``.

    Returns
    -------
    dict[str, TauProductModel]
        Maps reaction folder name → model.  Assign via
        :func:`assign_tau_models_to_library` or manually.

    Example
    -------
    ::

        models = build_tau_models(
            e_bins_root="fresco_code/newruns",
            reactions=["he4_he4_n_be7", "he4_he4_p_li7",
                       "he4_p_d_he3",   "li7_p_he4_he4"],
            energy_edges_mev=np.arange(0, 405, 5),
            product_species=["n", "p", "d", "t", "3He", "4He", "7Li", "7Be"],
        )
    """
    # Normalise product species to lowercase keys for the tensor
    product_species_lower = [_canonical_to_lower(sp) for sp in product_species]

    tau = load_tau_tensor(
        e_bins_root=e_bins_root,
        reactions=reactions,
        energy_edges_mev=energy_edges_mev,
        product_species=product_species_lower,
        infer_recoil=infer_recoil,
    )

    edges = np.asarray(energy_edges_mev, dtype=float)
    prod_index = {sp: i for i, sp in enumerate(product_species_lower)}

    return {
        rxn_name: TauProductModel(
            tau=tau,
            rxn_index=i,
            prod_index=prod_index,
            energy_edges_mev=edges,
        )
        for i, rxn_name in enumerate(reactions)
    }


# ---------------------------------------------------------------------------
# Helper: wire models into a loaded ReactionLibrary
# ---------------------------------------------------------------------------

def assign_tau_models_to_library(
    models: dict[str, TauProductModel],
    lib,
) -> int:
    """
    Assign :class:`TauProductModel` instances to matching Group 1 channels
    in a :class:`~reactions.ReactionLibrary`.

    Matching is done by converting each channel's ``(target, projectile,
    ejectile_label, residual)`` to the lowercase ``{target}_{proj}_{ej}_{res}``
    folder-key format and looking it up in *models*.

    Parameters
    ----------
    models : dict[str, TauProductModel]
        As returned by :func:`build_tau_models`.
    lib : ReactionLibrary
        Library whose Group 1 channels will receive product-distribution models.

    Returns
    -------
    int
        Number of channels successfully assigned.
    """
    assigned = 0
    for rxn in lib.by_group("group1"):
        key = (
            f"{_canonical_to_lower(rxn.target)}"
            f"_{_canonical_to_lower(rxn.projectile)}"
            f"_{_canonical_to_lower(rxn.ejectile_label)}"
            f"_{_canonical_to_lower(rxn.residual)}"
        )
        if key in models:
            rxn.product_distribution_model = models[key]
            assigned += 1
    return assigned
