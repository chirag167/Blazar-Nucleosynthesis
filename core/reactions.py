"""
reactions.py

Auto-discovers and manages non-thermal jet-cloud reaction channels.

This module is responsible for:
1. Discovering tabulated cross section files in:
       data/CrossSections/Group1/
       data/CrossSections/Group2/
2. Parsing filenames of the form:
       {target}_{projectile}{ejectile}_{residual}_{index}.csv
3. Supporting multi-particle ejectiles, e.g.
       3n3p, 2p, dn, np, 2np
4. Grouping multiple files belonging to the same physical reaction channel
   and averaging them when sigma(E) is queried.
5. Building canonical reaction objects with stoichiometric reactants/products.
6. Assigning compact indices to species and reactions for the solver.
7. Providing cross section access on arbitrary energies and edge-defined bins.

Internal unit conventions
-------------------------
Energy:
    MeV
Cross section:
    mb

Notes
-----
- This file does NOT evolve the state in time.
- This file does NOT calculate stopping powers or survival fractions.
- Q-values are delegated to utils/qvalue.py if available.
- DWBA product-energy distributions are intentionally left as a stub until
  their file format is finalized.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import importlib
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Optional: relativistic CM→lab energy conversion from utils/utils.py.
# Imported lazily so reactions.py can be used standalone if needed.
try:
    from utils.utils import cm_to_lab_energy_mev as _cm_to_lab_energy
    from utils.utils import reaction_threshold_lab_mev as _reaction_threshold_lab
except ImportError:  # pragma: no cover
    _cm_to_lab_energy = None       # type: ignore[assignment]
    _reaction_threshold_lab = None  # type: ignore[assignment]


def _compute_threshold(projectile: str, target: str, q_value: Optional[float]) -> Optional[float]:
    """
    Return the lab-frame reaction threshold [MeV] for endothermic channels.
    Returns None if Q >= 0 (exothermic) or if the mass table is missing a species.
    """
    if q_value is None or q_value >= 0.0 or _reaction_threshold_lab is None:
        return None
    try:
        return float(_reaction_threshold_lab(projectile, target, q_value))
    except KeyError:
        return None


# =============================================================================
# Units
# =============================================================================

_ENERGY_TO_MEV: Dict[str, float] = {
    "ev": 1.0e-6,
    "kev": 1.0e-3,
    "mev": 1.0,
    "gev": 1.0e3,
}

_SIGMA_TO_MB: Dict[str, float] = {
    "b": 1.0e3,
    "mb": 1.0,
    "microb": 1.0e-3,
    "mub": 1.0e-3,
    "nb": 1.0e-6,
    "pb": 1.0e-9,
    # "cm_B" appears in some datasets as "sigma measured in CM frame, in barns".
    # Total cross sections are Lorentz invariant so the value is identical to
    # the lab-frame barn — only the energy axis needs CM→lab conversion.
    "cm_b": 1.0e3,
}


def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    return unit.strip().lower()


def energy_unit_factor_to_mev(unit: Optional[str]) -> float:
    unit = _normalize_unit(unit) or "mev"
    if unit not in _ENERGY_TO_MEV:
        raise ValueError(f"Unsupported energy unit '{unit}'.")
    return _ENERGY_TO_MEV[unit]


def sigma_unit_factor_to_mb(unit: Optional[str]) -> float:
    unit = _normalize_unit(unit) or "mb"
    if unit not in _SIGMA_TO_MB:
        raise ValueError(f"Unsupported sigma unit '{unit}'.")
    return _SIGMA_TO_MB[unit]


def parse_column_and_unit(colname: str) -> Tuple[str, Optional[str]]:
    """
    Parse:
        E, sigma, dsigma
        E_keV, sigma_b, dsigma_nb
    into:
        ("E", None), ("sigma", "b"), ...
    """
    col = str(colname).strip()
    if "_" not in col:
        return col, None
    base, unit = col.split("_", 1)
    return base.strip(), unit.strip()


# =============================================================================
# Species helpers
# =============================================================================

# Ordered longest-first so token matching is unambiguous.
KNOWN_LIGHT_PARTICLE_TOKENS: Tuple[str, ...] = (
    "a",  # optional synonym support if it ever appears
    "3He",
    "4He",
    "6Li",
    "7Li",
    "7Be",
    "8Li",
    "8B",
    "9Be",
    "10B",
    "10Be",
    "11B",
    "11C",
    "12C",
    "13C",
    "13N",
    "14C",
    "14N",
    "15N",
    "15O",
    "16N",
    "16O",
    "17F",
    "17O",
    "18F",
    "18O",
    "20Ne",
    "d",
    "t",
    "n",
    "p",
)

# Only these are treated as concatenatable tokens inside ejectile strings.
# Heavy residual nuclei should not appear inside the compact ejectile field.
EJECTILE_TOKENS: Tuple[str, ...] = (
    "a",
    "3He",
    "4He",
    "d",
    "t",
    "n",
    "p",
)

ALIAS_TO_CANONICAL_SPECIES: Dict[str, str] = {
    "a": "4He",
}


def canonical_species_name(name: str) -> str:
    s = str(name).strip()
    return ALIAS_TO_CANONICAL_SPECIES.get(s, s)


def default_can_continue_nonthermal(species: str) -> bool:
    """
    Default continuation rule for energetic (non-thermal) descendants.

    Famiano (2002) Section 3:
        "only particles with A < 8 are treated as energetic projectiles in
        the network, while heavier reaction products are assumed to thermalize
        immediately in the cloud."

    Particles with A >= 8 return False and are deposited into the thermal
    cloud abundance pool rather than being tracked as non-thermal projectiles.
    """
    sp = canonical_species_name(species)
    # A < 8 species that can remain non-thermal
    _nonthermal_species = {"n", "p", "d", "t", "3He", "4He", "6Li", "7Li"}
    return sp in _nonthermal_species


def stoich_dict_add(target: Dict[str, int], species: str, amount: int = 1) -> None:
    species = canonical_species_name(species)
    target[species] = target.get(species, 0) + int(amount)
    if target[species] == 0:
        del target[species]


def stoich_dict_to_sorted_tuple(stoich: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((canonical_species_name(k), int(v)) for k, v in stoich.items() if int(v) != 0))


def format_stoich(stoich: Mapping[str, int]) -> str:
    parts = []
    for species, count in sorted(stoich.items()):
        if count == 1:
            parts.append(species)
        else:
            parts.append(f"{count}{species}")
    return " + ".join(parts) if parts else "0"


# =============================================================================
# Parsing compact ejectile strings
# =============================================================================

def parse_compact_species_string(compact: str) -> Dict[str, int]:
    """
    Parse a compact ejectile string like:
        "3n3p" -> {"n": 3, "p": 3}
        "2p"   -> {"p": 2}
        "dn"   -> {"d": 1, "n": 1}
        "np"   -> {"n": 1, "p": 1}
        "2np"  -> {"n": 2, "p": 1}

    Matching is left-to-right:
      [optional integer multiplicity][known token]

    The integer multiplicity applies only to the immediately following token.
    """
    s = str(compact).strip()
    if not s:
        return {}

    out: Dict[str, int] = {}
    i = 0
    n = len(s)

    while i < n:
        # Read optional multiplicity.
        j = i
        while j < n and s[j].isdigit():
            j += 1
        mult = int(s[i:j]) if j > i else 1

        matched = None
        for tok in EJECTILE_TOKENS:
            if s.startswith(tok, j):
                matched = tok
                break

        if matched is None:
            raise ValueError(
                f"Could not parse ejectile string '{compact}' near position {j}: '{s[j:]}'"
            )

        stoich_dict_add(out, matched, mult)
        i = j + len(matched)

    return out


# =============================================================================
# Filename parsing
# =============================================================================

_FILENAME_RE = re.compile(
    r"^(?P<target>[^_]+)_(?P<pe>[^_]+)_(?P<residual>[^_]+)_(?P<index>\d+)\.csv$"
)


@dataclass(frozen=True)
class ParsedReactionFilename:
    target: str
    projectile: str
    ejectile_label: str
    ejectile_stoich: Tuple[Tuple[str, int], ...]
    residual: str
    dataset_index: int
    filename: str

    @property
    def canonical_key(self) -> Tuple[str, str, Tuple[Tuple[str, int], ...], str]:
        return (
            canonical_species_name(self.target),
            canonical_species_name(self.projectile),
            self.ejectile_stoich,
            canonical_species_name(self.residual),
        )


def parse_reaction_filename(filename: Union[str, Path], allowed_projectiles: Optional[Sequence[str]] = None) -> ParsedReactionFilename:
    """
    Parse:
        {target}_{projectile}{ejectile}_{residual}_{index}.csv

    Examples:
        4He_p2p_3H_1.csv
        12C_p3n3p_7Be_2.csv
        4He_pdn_2H_1.csv   # if such naming ever appears

    Because projectile is concatenated with ejectile, we determine the projectile
    by matching against the allowed projectile list, longest-first. If no list is
    provided, we fall back to common projectile candidates.
    """
    name = Path(filename).name
    m = _FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"Filename does not match expected reaction format: {name}")

    target = canonical_species_name(m.group("target"))
    pe = m.group("pe")
    residual = canonical_species_name(m.group("residual"))
    index = int(m.group("index"))

    # Build a reverse map: canonical name → all aliases that appear in filenames.
    # e.g., "4He" → ["4He", "a"] so filenames like "10B_an_13N_1.csv" are parsed.
    _CANONICAL_TO_ALIASES: Dict[str, List[str]] = {}
    for alias, canon in ALIAS_TO_CANONICAL_SPECIES.items():
        _CANONICAL_TO_ALIASES.setdefault(canon, [canon]).append(alias)

    if allowed_projectiles is None:
        projectile_candidates = ["4He", "a", "3He", "d", "t", "n", "p"]
    else:
        cands: set = set()
        for p in allowed_projectiles:
            canon = canonical_species_name(p)
            cands.add(canon)
            cands.update(_CANONICAL_TO_ALIASES.get(canon, []))
        projectile_candidates = sorted(cands, key=len, reverse=True)

    projectile = None
    ejectile_label = None
    for cand in projectile_candidates:
        if pe.startswith(cand):
            projectile = canonical_species_name(cand)  # normalise alias → canonical
            ejectile_label = pe[len(cand):]
            break

    if projectile is None or ejectile_label is None:
        raise ValueError(
            f"Could not determine projectile/ejectile in filename '{name}'. "
            f"Allowed projectiles were: {projectile_candidates}"
        )

    ejectile_stoich = stoich_dict_to_sorted_tuple(parse_compact_species_string(ejectile_label))

    return ParsedReactionFilename(
        target=target,
        projectile=projectile,
        ejectile_label=ejectile_label,
        ejectile_stoich=ejectile_stoich,
        residual=residual,
        dataset_index=index,
        filename=name,
    )


# =============================================================================
# Cross section tables
# =============================================================================

@dataclass
class CrossSectionTable:
    """
    One tabulated cross section dataset.

    Internal units:
        energy_mev : MeV
        sigma_mb   : mb
        dsigma_mb  : mb, optional
    """
    energy_mev: np.ndarray
    sigma_mb: np.ndarray
    dsigma_mb: Optional[np.ndarray] = None
    source_file: Optional[Path] = None

    use_inverse_e_extrapolation: bool = True
    inverse_e_high_only: bool = True
    floor_sigma_mb: float = 0.0
    # True when energies are CM-frame and have NOT been converted to lab frame.
    # Set by from_file(); a future preprocessing step should convert these properly.
    is_cm_frame: bool = False

    def __post_init__(self) -> None:
        self.energy_mev = np.asarray(self.energy_mev, dtype=float)
        self.sigma_mb = np.asarray(self.sigma_mb, dtype=float)
        self.dsigma_mb = None if self.dsigma_mb is None else np.asarray(self.dsigma_mb, dtype=float)

        if self.energy_mev.ndim != 1 or self.sigma_mb.ndim != 1:
            raise ValueError("energy_mev and sigma_mb must be 1D arrays.")
        if len(self.energy_mev) != len(self.sigma_mb):
            raise ValueError("energy_mev and sigma_mb must have the same length.")
        if len(self.energy_mev) < 1:
            raise ValueError("Cross section table requires at least one data point.")
        if np.any(self.energy_mev <= 0.0):
            raise ValueError("All tabulated energies must be > 0 for interpolation/extrapolation.")

        order = np.argsort(self.energy_mev)
        self.energy_mev = self.energy_mev[order]
        self.sigma_mb = self.sigma_mb[order]
        if self.dsigma_mb is not None:
            if len(self.dsigma_mb) != len(self.energy_mev):
                raise ValueError("dsigma_mb must match energy_mev length.")
            self.dsigma_mb = self.dsigma_mb[order]

        # Merge duplicate energies within a single file by averaging.
        self._collapse_duplicate_energies()

    def _collapse_duplicate_energies(self) -> None:
        unique_e = np.unique(self.energy_mev)
        if len(unique_e) == len(self.energy_mev):
            return

        sigma_new = []
        dsigma_new = [] if self.dsigma_mb is not None else None

        for e in unique_e:
            mask = self.energy_mev == e
            sigma_new.append(np.mean(self.sigma_mb[mask]))
            if dsigma_new is not None:
                dsigma_new.append(np.mean(self.dsigma_mb[mask]))

        self.energy_mev = unique_e.astype(float)
        self.sigma_mb = np.asarray(sigma_new, dtype=float)
        if dsigma_new is not None:
            self.dsigma_mb = np.asarray(dsigma_new, dtype=float)

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        *,
        use_inverse_e_extrapolation: bool = True,
        inverse_e_high_only: bool = True,
        floor_sigma_mb: float = 0.0,
        projectile: Optional[str] = None,
        target: Optional[str] = None,
    ) -> "CrossSectionTable":
        """
        Load a cross section CSV file.

        Supported energy column names (case-insensitive base):
            E, E_MeV               → MeV, lab frame
            E_keV, E_kev           → keV, lab frame
            E_cm, E_CM             → MeV, CM frame  [WARNING: not converted to lab frame]
            E_cm_keV, E_cm_kev     → keV, CM frame  [WARNING: not converted to lab frame]
            E_sum_CM               → MeV, CM frame  [WARNING: not converted to lab frame]
            E_min / E_max pair     → bin edges; midpoints are used as E

        Files with 'arb_units' sigma columns or 'theta' energy columns are skipped
        (a ValueError is raised so the caller can catch and skip the file).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Cross section file not found: {path}")

        try:
            df = pd.read_csv(path, comment="#")
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", comment="#")

        if df.shape[1] < 2:
            raise ValueError(f"{path}: fewer than 2 columns.")

        cols_lower = {c.lower(): c for c in df.columns}

        # --- Reject files with angular (theta) energy axis ---
        if "theta" in cols_lower:
            raise ValueError(
                f"{path.name}: first column is 'theta' (angular differential cross section). "
                "This file contains dσ/dΩ data at a fixed beam energy, not σ(E). "
                "Replace it with a σ(E) excitation function dataset."
            )

        # --- Reject files with arbitrary-unit sigma ---
        for col in df.columns:
            if "arb_unit" in col.lower():
                raise ValueError(
                    f"{path.name}: sigma column '{col}' is in arbitrary units. "
                    "Cannot use for physics calculations — replace with absolute cross section data."
                )

        # --- Handle E_min / E_max bin-edge format ---
        if "e_min" in cols_lower and "e_max" in cols_lower:
            e_min_arr = pd.to_numeric(df[cols_lower["e_min"]], errors="coerce").to_numpy(dtype=float)
            e_max_arr = pd.to_numeric(df[cols_lower["e_max"]], errors="coerce").to_numpy(dtype=float)
            # Drop rows where either edge is NaN (first row sometimes has a blank E_min)
            valid = np.isfinite(e_min_arr) & np.isfinite(e_max_arr)
            # Keep energy_raw full-length (NaN for invalid rows) so the shared
            # filtering block below can apply valid to both energy_raw and sigma_raw
            # uniformly (sigma_raw is also full-length at that point).
            energy_raw = 0.5 * (e_min_arr + e_max_arr)
            log.warning(
                "%s: E_min/E_max bin-edge format detected; using bin midpoints as E. "
                "Verify that energy units are MeV.", path.name
            )
        else:
            # --- Find the energy column ---
            e_col = None
            e_unit = None
            is_cm_frame: bool = False

            # Priority order: plain E > E_MeV > E_keV > E_cm > E_sum_CM > E_cm_keV
            for candidate in ("e", "e_mev", "e_kev", "e_cm", "e_sum_cm", "e_cm_kev", "e_cm_kev"):
                if candidate in cols_lower:
                    e_col = cols_lower[candidate]
                    break

            if e_col is None:
                raise ValueError(
                    f"{path.name}: no recognisable energy column found. "
                    f"Columns present: {list(df.columns)}"
                )

            col_lower = e_col.lower()
            if col_lower in ("e", "e_mev"):
                e_unit = "mev"
            elif col_lower in ("e_kev", "e_kev"):
                e_unit = "kev"
            elif col_lower in ("e_cm", "e_sum_cm"):
                e_unit = "mev"
                is_cm_frame = True
                log.warning(
                    "%s: energy column '%s' appears to be CM-frame energy. "
                    "Lab-frame conversion requires projectile/target masses and is NOT "
                    "applied automatically. Verify results against the paper.", path.name, e_col
                )
            elif col_lower == "e_cm_kev":
                e_unit = "kev"
                is_cm_frame = True
                log.warning(
                    "%s: energy column '%s' appears to be CM-frame energy in keV. "
                    "Lab-frame conversion is NOT applied automatically.", path.name, e_col
                )
            else:
                # Generic fallback: try to infer unit from suffix
                _, inferred_unit = parse_column_and_unit(e_col)
                try:
                    energy_unit_factor_to_mev(inferred_unit)
                    e_unit = inferred_unit
                except ValueError:
                    raise ValueError(
                        f"{path.name}: energy column '{e_col}' has unrecognised unit "
                        f"'{inferred_unit}'. Expected MeV or keV."
                    )

            energy_raw = df[e_col].to_numpy(dtype=float) * energy_unit_factor_to_mev(e_unit)

            # --- CM → lab frame conversion ---
            if is_cm_frame:
                if projectile is not None and target is not None and _cm_to_lab_energy is not None:
                    try:
                        energy_raw = _cm_to_lab_energy(energy_raw, projectile, target)
                        log.info(
                            "%s: converted CM-frame energies to lab frame "
                            "(projectile=%s, target=%s).",
                            path.name, projectile, target,
                        )
                        is_cm_frame = False  # conversion applied; mark as lab frame now
                    except (KeyError, ValueError) as exc:
                        log.warning(
                            "%s: CM→lab conversion failed (%s). "
                            "Energies remain in CM frame.", path.name, exc,
                        )
                else:
                    log.warning(
                        "%s: CM-frame energies detected but projectile/target not "
                        "provided — conversion skipped. Pass projectile= and target= "
                        "to CrossSectionTable.from_file() to enable conversion.",
                        path.name,
                    )

            valid = np.ones(len(energy_raw), dtype=bool)

        # --- Find sigma column ---
        sigma_col = None
        sigma_unit = None
        for col in df.columns:
            base, unit = parse_column_and_unit(col)
            if base.lower() == "sigma":
                sigma_col = col
                sigma_unit = unit
                break

        if sigma_col is None:
            raise ValueError(
                f"{path.name}: no 'sigma' column found. Columns: {list(df.columns)}"
            )

        try:
            sigma_factor = sigma_unit_factor_to_mb(sigma_unit)
        except ValueError:
            raise ValueError(
                f"{path.name}: sigma column '{sigma_col}' has unrecognised unit "
                f"'{sigma_unit}'. Expected b, mb, μb, nb, or pb."
            )

        sigma_raw = df[sigma_col].to_numpy(dtype=float) * sigma_factor

        # --- Optional dsigma column ---
        dsigma_mb = None
        for col in df.columns:
            base, unit = parse_column_and_unit(col)
            if base.lower() == "dsigma":
                try:
                    du = sigma_unit_factor_to_mb(unit)
                    dsigma_mb = df[col].to_numpy(dtype=float) * du
                except ValueError:
                    pass  # ignore dsigma column with unrecognised unit
                break

        # Apply the valid mask (from E_min/E_max path) and finite-value filter
        energy_mev = energy_raw[valid] if valid is not True else energy_raw
        sigma_mb_arr = sigma_raw[valid] if valid is not True else sigma_raw

        finite_mask = np.isfinite(energy_mev) & np.isfinite(sigma_mb_arr)
        if dsigma_mb is not None:
            dsigma_arr = dsigma_mb[valid] if valid is not True else dsigma_mb
            finite_mask &= np.isfinite(dsigma_arr)
            dsigma_mb = dsigma_arr[finite_mask]

        # is_cm_frame is only defined inside the else-branch; default False for
        # the E_min/E_max path where the frame is assumed to be lab.
        _is_cm = locals().get("is_cm_frame", False)

        return cls(
            energy_mev=energy_mev[finite_mask],
            sigma_mb=sigma_mb_arr[finite_mask],
            dsigma_mb=dsigma_mb,
            source_file=path,
            use_inverse_e_extrapolation=use_inverse_e_extrapolation,
            inverse_e_high_only=inverse_e_high_only,
            floor_sigma_mb=floor_sigma_mb,
            is_cm_frame=_is_cm,
        )

    @property
    def emin(self) -> float:
        return float(self.energy_mev[0])

    @property
    def emax(self) -> float:
        return float(self.energy_mev[-1])

    def sigma_interpolate(self, energy_mev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolate within the tabulated range.
        Above the highest tabulated energy, optionally extrapolate as 1/E.
        Below the lowest tabulated energy:
            - hold constant if inverse_e_high_only=True
            - else also extrapolate as 1/E
        """
        scalar_input = np.isscalar(energy_mev)
        E = np.atleast_1d(np.asarray(energy_mev, dtype=float))

        out = np.interp(
            E,
            self.energy_mev,
            self.sigma_mb,
            left=self.sigma_mb[0],
            right=self.sigma_mb[-1],
        )

        if self.use_inverse_e_extrapolation:
            high = E > self.emax
            if np.any(high):
                out[high] = self.sigma_mb[-1] * (self.emax / E[high])

            if not self.inverse_e_high_only:
                low = E < self.emin
                if np.any(low):
                    out[low] = self.sigma_mb[0] * (self.emin / E[low])

        out = np.maximum(out, self.floor_sigma_mb)

        if scalar_input:
            return float(out[0])
        return out

    def average_sigma_on_bins(self, energy_edges_mev: np.ndarray) -> np.ndarray:
        """
        Return the average cross section for each edge-defined bin.

        Current implementation: midpoint evaluation.
        Easy to replace later with a more accurate quadrature if desired.
        """
        edges = np.asarray(energy_edges_mev, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("energy_edges_mev must be a 1D array with length >= 2.")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("energy_edges_mev must be strictly increasing.")

        mids = 0.5 * (edges[:-1] + edges[1:])
        return np.asarray(self.sigma_interpolate(mids), dtype=float)


@dataclass
class AggregatedCrossSection:
    """
    A physical reaction channel may have multiple tabulated datasets.

    At query time, all datasets are interpolated/extrapolated to the requested
    energy (or bins), then averaged.
    """
    tables: List[CrossSectionTable]
    combine_mode: str = "average"

    def __post_init__(self) -> None:
        if not self.tables:
            raise ValueError("AggregatedCrossSection requires at least one table.")
        if self.combine_mode not in {"average", "first"}:
            raise ValueError("combine_mode must be 'average' or 'first'.")

    def sigma_interpolate(self, energy_mev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        vals = [tbl.sigma_interpolate(energy_mev) for tbl in self.tables]
        arr = np.asarray(vals, dtype=float)

        if self.combine_mode == "first":
            out = arr[0]
        else:
            out = np.mean(arr, axis=0)

        if np.isscalar(energy_mev):
            return float(np.atleast_1d(out)[0])
        return out

    def average_sigma_on_bins(self, energy_edges_mev: np.ndarray) -> np.ndarray:
        vals = [tbl.average_sigma_on_bins(energy_edges_mev) for tbl in self.tables]
        arr = np.asarray(vals, dtype=float)

        if self.combine_mode == "first":
            return arr[0]
        return np.mean(arr, axis=0)

    @property
    def source_files(self) -> List[Path]:
        return [tbl.source_file for tbl in self.tables if tbl.source_file is not None]


# =============================================================================
# Product distribution stubs
# =============================================================================

class ProductDistributionModel:
    def distribution(
        self,
        *,
        projectile_energy_mev: float,
        product_species: str,
        product_energy_edges_mev: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DeltaAtThermalModel(ProductDistributionModel):
    """
    Put all product weight into the lowest energy bin.
    """
    def distribution(
        self,
        *,
        projectile_energy_mev: float,
        product_species: str,
        product_energy_edges_mev: np.ndarray,
    ) -> np.ndarray:
        nbins = len(product_energy_edges_mev) - 1
        out = np.zeros(nbins, dtype=float)
        out[0] = 1.0
        return out


class DWBAStubModel(ProductDistributionModel):
    """
    Placeholder until the DWBA output format is finalized.
    """
    def distribution(
        self,
        *,
        projectile_energy_mev: float,
        product_species: str,
        product_energy_edges_mev: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("DWBA output format has not been specified yet.")


# =============================================================================
# Species registry
# =============================================================================

@dataclass
class SpeciesRegistry:
    species_to_index: Dict[str, int] = field(default_factory=dict)
    index_to_species: List[str] = field(default_factory=list)

    def add(self, species: str) -> int:
        species = canonical_species_name(species)
        if species not in self.species_to_index:
            self.species_to_index[species] = len(self.index_to_species)
            self.index_to_species.append(species)
        return self.species_to_index[species]

    def add_many(self, species_list: Iterable[str]) -> None:
        for s in species_list:
            self.add(s)

    def index(self, species: str) -> int:
        return self.species_to_index[canonical_species_name(species)]

    def __contains__(self, species: str) -> bool:
        return canonical_species_name(species) in self.species_to_index

    def __len__(self) -> int:
        return len(self.index_to_species)

    def as_list(self) -> List[str]:
        return list(self.index_to_species)


# =============================================================================
# Reaction products / channels
# =============================================================================

@dataclass(frozen=True)
class ReactionProduct:
    species: str
    multiplicity: int = 1
    can_continue_nonthermal: bool = True


@dataclass
class ReactionChannel:
    reaction_index: int
    group: str  # "group1" or "group2"

    target: str
    projectile: str
    ejectile_label: str
    residual: str

    reactants_stoich: Dict[str, int]
    products_stoich: Dict[str, int]

    cross_section: AggregatedCrossSection

    q_value_mev: Optional[float] = None
    threshold_mev: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    product_distribution_model: Optional[ProductDistributionModel] = None
    allow_nonthermal_descendants: bool = True

    source_files: List[Path] = field(default_factory=list)

    def canonical_key(self) -> Tuple[str, str, Tuple[Tuple[str, int], ...], str]:
        ejectile_only = dict(self.products_stoich)
        residual = canonical_species_name(self.residual)
        ejectile_only[residual] = ejectile_only.get(residual, 0) - 1
        if ejectile_only[residual] == 0:
            del ejectile_only[residual]

        return (
            canonical_species_name(self.target),
            canonical_species_name(self.projectile),
            stoich_dict_to_sorted_tuple(ejectile_only),
            residual,
        )

    def name(self) -> str:
        return f"{self.target}({self.projectile},{self.ejectile_label}){self.residual}"

    def reaction_equation(self) -> str:
        return f"{format_stoich(self.reactants_stoich)} -> {format_stoich(self.products_stoich)}"

    def is_open(self, projectile_energy_mev: float) -> bool:
        if self.threshold_mev is None:
            return projectile_energy_mev > 0.0
        return projectile_energy_mev >= self.threshold_mev

    def sigma_mb(self, projectile_energy_mev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        E = np.asarray(projectile_energy_mev, dtype=float)
        sigma = self.cross_section.sigma_interpolate(E)

        if self.threshold_mev is not None:
            sigma = np.where(E >= self.threshold_mev, sigma, 0.0)

        if np.isscalar(projectile_energy_mev):
            return float(np.atleast_1d(sigma)[0])
        return sigma

    def sigma_bin_average_mb(self, energy_edges_mev: np.ndarray) -> np.ndarray:
        avg = self.cross_section.average_sigma_on_bins(energy_edges_mev)
        if self.threshold_mev is None:
            return avg

        mids = 0.5 * (energy_edges_mev[:-1] + energy_edges_mev[1:])
        return np.where(mids >= self.threshold_mev, avg, 0.0)

    def products_as_objects(self) -> List[ReactionProduct]:
        out: List[ReactionProduct] = []
        for species, mult in sorted(self.products_stoich.items()):
            can_continue = self.allow_nonthermal_descendants and default_can_continue_nonthermal(species)
            out.append(
                ReactionProduct(
                    species=species,
                    multiplicity=mult,
                    can_continue_nonthermal=can_continue,
                )
            )
        return out


# =============================================================================
# Q-value integration
# =============================================================================

def try_compute_q_value_mev(
    reactants_stoich: Mapping[str, int],
    products_stoich: Mapping[str, int],
) -> Optional[float]:
    """
    Best-effort hook into utils/qvalue.py.

    Expected one of the following to exist:
        q_value_mev(reactants_stoich, products_stoich)
        compute_q_value_mev(reactants_stoich, products_stoich)

    If unavailable, returns None.
    """
    try:
        mod = importlib.import_module("utils.qvalue")
    except Exception:
        return None

    for fname in ("q_value_mev", "compute_q_value_mev"):
        fn = getattr(mod, fname, None)
        if callable(fn):
            return float(fn(dict(reactants_stoich), dict(products_stoich)))

    return None


# =============================================================================
# Reaction library
# =============================================================================

@dataclass
class ReactionLibrary:
    species_registry: SpeciesRegistry
    reactions: List[ReactionChannel]

    reaction_to_index: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_projectile: Dict[str, List[ReactionChannel]] = {}
        self._by_target: Dict[str, List[ReactionChannel]] = {}
        self._by_group: Dict[str, List[ReactionChannel]] = {}
        self._by_product_species: Dict[str, List[ReactionChannel]] = {}

        for rxn in self.reactions:
            self._by_projectile.setdefault(rxn.projectile, []).append(rxn)
            self._by_target.setdefault(rxn.target, []).append(rxn)
            self._by_group.setdefault(rxn.group, []).append(rxn)

            for species in rxn.products_stoich:
                self._by_product_species.setdefault(species, []).append(rxn)

            self.reaction_to_index[rxn.name()] = rxn.reaction_index

    def all(self) -> List[ReactionChannel]:
        return list(self.reactions)

    def by_projectile(self, projectile: str) -> List[ReactionChannel]:
        return list(self._by_projectile.get(canonical_species_name(projectile), []))

    def by_target(self, target: str) -> List[ReactionChannel]:
        return list(self._by_target.get(canonical_species_name(target), []))

    def by_group(self, group: str) -> List[ReactionChannel]:
        return list(self._by_group.get(group.lower(), []))

    def by_product_species(self, species: str) -> List[ReactionChannel]:
        return list(self._by_product_species.get(canonical_species_name(species), []))

    def open_reactions(self, projectile: str, energy_mev: float) -> List[ReactionChannel]:
        projectile = canonical_species_name(projectile)
        return [rxn for rxn in self.by_projectile(projectile) if rxn.is_open(energy_mev)]

    def destruction_fraction_per_reaction(
        self,
        *,
        projectile: str,
        projectile_energy_mev: float,
        target_number_densities: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Implements the discrete reaction competition:
            beta_i(E_k) = sigma_i(E_k) * N_target_i / sum_m sigma_m(E_k) * N_target_m
        """
        relevant = self.open_reactions(projectile, projectile_energy_mev)

        weights: Dict[str, float] = {}
        denom = 0.0

        for rxn in relevant:
            Nt = float(target_number_densities.get(rxn.target, 0.0))
            w = float(rxn.sigma_mb(projectile_energy_mev)) * Nt
            weights[rxn.name()] = w
            denom += w

        if denom <= 0.0:
            return {name: 0.0 for name in weights}

        return {name: w / denom for name, w in weights.items()}

    def sigma_matrix_for_projectile(
        self,
        *,
        projectile: str,
        energy_edges_mev: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        projectile = canonical_species_name(projectile)
        return {
            rxn.name(): rxn.sigma_bin_average_mb(energy_edges_mev)
            for rxn in self.by_projectile(projectile)
        }

    @classmethod
    def from_directories(
        cls,
        *,
        base_dir: Union[str, Path] = ".",
        projectile_species: Optional[Sequence[str]] = None,
        target_species: Optional[Sequence[str]] = None,
        use_inverse_e_extrapolation: bool = True,
        inverse_e_high_only: bool = True,
        combine_mode: str = "average",
        group2_default_thermalize: bool = True,
        include_group1: bool = True,
        include_group2: bool = True,
    ) -> "ReactionLibrary":
        """
        Build the reaction library by scanning cross section directories.

        Parameters
        ----------
        projectile_species
            Allowed projectile species. If provided, only reactions with these
            projectiles are kept. This should typically come from jet.json/beam.json.
        target_species
            Allowed target species. If provided, only reactions with these targets
            are kept. This should typically come from cloud.json.
        """
        base_dir = Path(base_dir)
        species_registry = SpeciesRegistry()

        allowed_projectiles = None
        if projectile_species is not None:
            allowed_projectiles = [canonical_species_name(p) for p in projectile_species]

        allowed_targets = None
        if target_species is not None:
            allowed_targets = {canonical_species_name(t) for t in target_species}

        grouped: Dict[Tuple[str, str, Tuple[Tuple[str, int], ...], str, str], List[Path]] = {}

        def scan_group(group_name: str) -> None:
            xs_dir = base_dir / "data" / "CrossSections" / group_name
            if not xs_dir.exists():
                return

            for path in sorted(xs_dir.glob("*.csv")):
                try:
                    parsed = parse_reaction_filename(path.name, allowed_projectiles=allowed_projectiles)
                except ValueError as exc:
                    log.warning("Skipping %s (filename parse error): %s", path.name, exc)
                    continue

                if allowed_projectiles is not None and parsed.projectile not in allowed_projectiles:
                    continue
                if allowed_targets is not None and parsed.target not in allowed_targets:
                    continue

                key = (
                    group_name.lower(),  # group1 or group2
                    parsed.target,
                    parsed.projectile,
                    parsed.ejectile_stoich,
                    parsed.residual,
                )
                grouped.setdefault(key, []).append(path)

        if include_group1:
            scan_group("Group1")
        if include_group2:
            scan_group("Group2")

        reactions: List[ReactionChannel] = []

        for ridx, (key, files) in enumerate(sorted(grouped.items(), key=lambda x: str(x[0]))):
            group_name, target, projectile, ejectile_stoich, residual = key
            group_name = group_name.lower()

            # Reconstruct a compact ejectile label for display.
            ejectile_label = _stoich_to_compact_label(dict(ejectile_stoich))

            reactants_stoich: Dict[str, int] = {}
            products_stoich: Dict[str, int] = {}

            stoich_dict_add(reactants_stoich, projectile, 1)
            stoich_dict_add(reactants_stoich, target, 1)

            for species, count in dict(ejectile_stoich).items():
                stoich_dict_add(products_stoich, species, count)
            stoich_dict_add(products_stoich, residual, 1)

            # Load all datasets for this physical channel; skip files that cannot
            # be parsed (bad column names, angular data, arbitrary units, etc.).
            tables = []
            for fp in sorted(files):
                try:
                    tables.append(CrossSectionTable.from_file(
                        fp,
                        use_inverse_e_extrapolation=use_inverse_e_extrapolation,
                        inverse_e_high_only=inverse_e_high_only,
                        projectile=projectile,
                        target=target,
                    ))
                except (ValueError, FileNotFoundError) as exc:
                    log.warning("Skipping %s: %s", fp.name, exc)

            if not tables:
                log.warning(
                    "Skipping reaction channel %s(%s,%s)%s — all data files failed to load.",
                    target, projectile, ejectile_label, residual,
                )
                continue

            aggregated = AggregatedCrossSection(tables=tables, combine_mode=combine_mode)

            q_value = try_compute_q_value_mev(reactants_stoich, products_stoich)

            if group_name == "group1":
                allow_desc = True
                pdm: Optional[ProductDistributionModel] = DWBAStubModel()
            else:
                allow_desc = not group2_default_thermalize
                pdm = DeltaAtThermalModel() if group2_default_thermalize else None

            # Register all species automatically.
            species_registry.add_many(list(reactants_stoich.keys()) + list(products_stoich.keys()))

            reactions.append(
                ReactionChannel(
                    reaction_index=ridx,
                    group=group_name,
                    target=target,
                    projectile=projectile,
                    ejectile_label=ejectile_label,
                    residual=residual,
                    reactants_stoich=reactants_stoich,
                    products_stoich=products_stoich,
                    cross_section=aggregated,
                    q_value_mev=q_value,
                    threshold_mev=_compute_threshold(projectile, target, q_value),
                    metadata={},
                    product_distribution_model=pdm,
                    allow_nonthermal_descendants=allow_desc,
                    source_files=sorted(files),
                )
            )

        return cls(
            species_registry=species_registry,
            reactions=reactions,
        )


# =============================================================================
# Helper conversions
# =============================================================================

def _stoich_to_compact_label(stoich: Mapping[str, int]) -> str:
    """
    Convert stoichiometry back to a compact ejectile label.
    Examples:
        {"n": 3, "p": 3} -> "3n3p"
        {"d": 1, "n": 1} -> "dn"

    Order preference is physically conventional and stable for filenames/display.
    """
    order = ["alpha", "4He", "3He", "t", "d", "n", "p"]

    # Use canonical names in output.
    canon = {canonical_species_name(k): int(v) for k, v in stoich.items() if int(v) != 0}

    parts = []
    for tok in order:
        ctok = canonical_species_name(tok)
        if ctok in canon:
            count = canon.pop(ctok)
            if count == 1:
                # preserve "alpha" only if original token truly exists in stoich
                parts.append(ctok)
            else:
                parts.append(f"{count}{ctok}")

    for species in sorted(canon):
        count = canon[species]
        if count == 1:
            parts.append(species)
        else:
            parts.append(f"{count}{species}")

    return "".join(parts)


# =============================================================================
# Optional convenience hooks for beam/cloud JSON
# =============================================================================

def extract_projectiles_from_jet_config(jet_cfg: Mapping[str, object]) -> List[str]:
    """
    Best-effort helper. Adjust to your final jet/beam JSON schema later.

    Supported patterns:
        {"species": ["p", "4He", ...]}
        {"projectiles": ["p", "4He", ...]}
        {"beam_species": ["p", "4He", ...]}
    """
    for key in ("species", "projectiles", "beam_species"):
        if key in jet_cfg:
            vals = jet_cfg[key]
            if not isinstance(vals, Sequence) or isinstance(vals, (str, bytes)):
                raise ValueError(f"Expected a list-like value for jet config key '{key}'.")
            return [canonical_species_name(v) for v in vals]
    raise KeyError("Could not find projectile species list in jet config.")


def extract_targets_from_cloud_config(cloud_cfg: Mapping[str, object]) -> List[str]:
    """
    Best-effort helper. Adjust to your final cloud JSON schema later.

    Supported patterns:
        {"species": ["1H", "4He", ...]}
        {"targets": ["1H", "4He", ...]}
        {"cloud_species": ["1H", "4He", ...]}
        {"abundances": {"1H": ..., "4He": ...}}
    """
    for key in ("species", "targets", "cloud_species"):
        if key in cloud_cfg:
            vals = cloud_cfg[key]
            if not isinstance(vals, Sequence) or isinstance(vals, (str, bytes)):
                raise ValueError(f"Expected a list-like value for cloud config key '{key}'.")
            return [canonical_species_name(v) for v in vals]

    if "abundances" in cloud_cfg:
        abund = cloud_cfg["abundances"]
        if not isinstance(abund, Mapping):
            raise ValueError("cloud_cfg['abundances'] must be a mapping.")
        return [canonical_species_name(k) for k in abund.keys()]

    raise KeyError("Could not find target species list in cloud config.")