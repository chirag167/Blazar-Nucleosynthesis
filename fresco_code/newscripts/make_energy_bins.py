#!/usr/bin/env python3
"""
Build 5-MeV energy bins (5-400 MeV) for each Group I reaction.

FRESCO reactions: use scale-factor-normalised cross sections from
  newruns/plots/<rxn>_scale_factors.csv.  Bins where scale_factor==0
  (no data, no normalisation possible) are marked 'gap'.  Bins where
  extrapolated==1 are marked 'extrapolated'.  Everything else is 'scaled'.

Data-only reactions: log-linearly interpolate the experimental total cross
  sections in newruns/<rxn>/<rxn>_total_xs.csv onto the 5-MeV grid.
  Bins outside [E_min_data, E_max_data] are marked 'gap'.
  Bins between data points are marked 'interpolated'.
  Bins that land exactly on a data point are marked 'data'.

Output: newruns/bins/<rxn>_bins.csv  (one file per reaction)
        newruns/bins/all_reactions_bins.csv  (combined)

Columns: reaction, E_MeV, sigma_mb, d_sigma_mb, status
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

NEWRUNS = Path(__file__).parent.parent / "newruns"
OUT_DIR = NEWRUNS / "bins"
GRID = np.arange(5.0, 405.0, 5.0)   # 5, 10, ..., 400 MeV  (80 bins)

FRESCO_REACTIONS = [
    "he4_he4_n_be7",
    "he4_he4_p_li7",
    "he4_p_d_he3",
    "li7_p_he4_he4",
]

DATA_REACTIONS = [
    "he4_he4_n_li7",
    "he4_p_p_he4",
    "p_d_dp_n",
    "p_d_n_d",
    "p_d_n_p",
    "p_he4_p_he4",
    "p_li7_he4_he4",
    # Meyer (1972) Table 10 — p+He4 breakup channels
    "he4_p_he3_np",
    "he4_p_h3_2p",
    "he4_p_dd",
    "he4_p_d_np",
    "he4_p_2n3p",
    # Meyer (1972) Table 9 — p+d inelastic
    "p_d_inel",
]

ATOL = 0.5   # tolerance (MeV) for matching a grid point to a data point


def bin_fresco_reaction(rxn: str) -> pd.DataFrame:
    sf_path = NEWRUNS / "plots" / f"{rxn}_scale_factors.csv"
    sf = pd.read_csv(sf_path)

    rows = []
    for E in GRID:
        mask = np.abs(sf["E_MeV"] - E) < ATOL
        if not mask.any():
            rows.append({"E_MeV": E, "sigma_mb": np.nan,
                         "d_sigma_mb": np.nan, "status": "gap"})
            continue

        row = sf[mask].iloc[0]
        sigma_norm = row["sigma_norm_mb"]
        sf_val = row["scale_factor"]
        extrap = int(row["extrapolated"])

        if sf_val == 0 or (isinstance(sigma_norm, float) and np.isnan(sigma_norm)):
            status, sigma, d_sigma = "gap", np.nan, np.nan
        elif extrap:
            status, sigma, d_sigma = "extrapolated", float(sigma_norm), np.nan
        else:
            status, sigma, d_sigma = "scaled", float(sigma_norm), np.nan

        rows.append({"E_MeV": E, "sigma_mb": sigma,
                     "d_sigma_mb": d_sigma, "status": status})

    return pd.DataFrame(rows)


def bin_data_reaction(rxn: str) -> pd.DataFrame:
    xs_path = NEWRUNS / rxn / f"{rxn}_total_xs.csv"
    xs = pd.read_csv(xs_path).sort_values("E_MeV").reset_index(drop=True)

    E_data = xs["E_MeV"].values.astype(float)
    s_data = xs["sigma_mb"].values.astype(float)
    ds_data = xs["d_sigma_mb"].values.astype(float) if "d_sigma_mb" in xs.columns else np.full(len(E_data), np.nan)

    # Drop non-positive sigma (can't interpolate in log space)
    valid = s_data > 0
    E_data, s_data, ds_data = E_data[valid], s_data[valid], ds_data[valid]

    if len(E_data) < 2:
        return pd.DataFrame([
            {"E_MeV": E, "sigma_mb": np.nan, "d_sigma_mb": np.nan, "status": "gap"}
            for E in GRID
        ])

    E_min, E_max = E_data.min(), E_data.max()

    log_s_interp  = interp1d(E_data, np.log(s_data),  kind="linear",
                              bounds_error=False, fill_value=np.nan)
    ds_interp     = interp1d(E_data, ds_data,          kind="linear",
                              bounds_error=False, fill_value=np.nan)

    rows = []
    for E in GRID:
        if E < E_min - ATOL or E > E_max + ATOL:
            rows.append({"E_MeV": E, "sigma_mb": np.nan,
                         "d_sigma_mb": np.nan, "status": "gap"})
            continue

        # Check if E lands on a data point
        on_data = np.any(np.abs(E_data - E) < ATOL)

        log_s = log_s_interp(E)
        sigma = float(np.exp(log_s)) if not np.isnan(log_s) else np.nan
        d_sigma = float(ds_interp(E))
        if np.isnan(d_sigma):
            d_sigma = np.nan

        status = "data" if on_data else "interpolated"
        rows.append({"E_MeV": E, "sigma_mb": sigma,
                     "d_sigma_mb": d_sigma, "status": status})

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for rxn in FRESCO_REACTIONS:
        print(f"  FRESCO  {rxn}")
        df = bin_fresco_reaction(rxn)
        df.insert(0, "reaction", rxn)
        df.to_csv(OUT_DIR / f"{rxn}_bins.csv", index=False)
        _summarise(rxn, df)
        all_frames.append(df)

    for rxn in DATA_REACTIONS:
        print(f"  data    {rxn}")
        df = bin_data_reaction(rxn)
        df.insert(0, "reaction", rxn)
        df.to_csv(OUT_DIR / f"{rxn}_bins.csv", index=False)
        _summarise(rxn, df)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_reactions_bins.csv", index=False)
    print(f"\nWrote {len(all_frames)} reaction files + combined to {OUT_DIR}/")


def _summarise(rxn: str, df: pd.DataFrame):
    counts = df["status"].value_counts().to_dict()
    parts = [f"{s}={n}" for s, n in sorted(counts.items())]
    print(f"    {rxn}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
