#!/usr/bin/env python3
"""
Create reaction directories and data CSV files from Meyer (1972) A&AS 7, 417-467.

New data-only reactions added:
  he4_p_he3_np  -- p+4He -> 3He+n+p      (Meyer "He3",  Table 10, sigma_He3)
  he4_p_h3_2p   -- p+4He -> 3H+2p         (Meyer "H3",   Table 10, sigma_H3)
  he4_p_dd      -- p+4He -> d+d+p          (Meyer "dd",   Table 10, sigma_dd)
  he4_p_d_np    -- p+4He -> d+n+2p         (Meyer "d",    Table 10, sigma_d)
  he4_p_2n3p    -- p+4He -> 2n+3p          (Meyer "o",    Table 10, sigma_o)
  p_d_inel      -- p+d total inelastic     (Meyer Table 9, sigma_pd,inel)

All values extracted from tabulated data in Meyer (1972); errors from stated
experimental uncertainties (upper bound used when asymmetric).
"""

import csv
from pathlib import Path

NEWRUNS = Path(__file__).parent.parent / "newruns"

# Each entry: (E_MeV, sigma_mb, d_sigma_mb)
REACTIONS = {
    # Meyer Table 10, sigma_He3 column.
    # "He3" process: p + He4 -> He3 + n + p  (threshold 25.8 MeV lab).
    # Sources: Cairns et al. 1964 (53 MeV, radiochemical); Tannenwald 1953
    # renormalised to sigma_inel (90 MeV); Innes 1957 renormalised (300 MeV).
    "he4_p_he3_np": [
        (53.0,  49.0, 8.0),
        (90.0,  46.0, 8.0),
        (300.0, 27.0, 10.0),
    ],

    # Meyer Table 10, sigma_H3 column.
    # "H3" process: p + He4 -> H3 + 2p  (threshold 24.9 MeV lab).
    # Sources: Wickersham 1957 (28 MeV, direct); Tannenwald via sigma_H3/sigma_He3
    # = sigma_pp/sigma_pn (90 MeV); Innes 1957 renormalised (300 MeV).
    "he4_p_h3_2p": [
        (28.0,   8.9,  1.0),
        (90.0,  17.0,  3.0),
        (300.0, 16.5,  6.0),
    ],

    # Meyer Table 10, sigma_dd column.
    # "dd" process: p + He4 -> d + d + p  (threshold 29.8 MeV lab).
    # Sources: Tannenwald 1953 / Innes 1957 branching ratios renormalised to
    # sigma_inel = 105 mb (90 MeV) and 63 mb (300 MeV).
    "he4_p_dd": [
        (90.0,  5.1, 2.0),
        (300.0, 5.3, 2.5),
    ],

    # Meyer Table 10, sigma_d column.
    # "d" process: p + He4 -> d + n + 2p  (threshold 32.7 MeV lab).
    # Same sources as sigma_dd above.
    "he4_p_d_np": [
        (90.0,  10.1, 4.0),
        (300.0, 12.0, 5.0),
    ],

    # Meyer Table 10, sigma_o column.
    # "o" process: p + He4 -> 2n + 3p  (threshold 35.4 MeV lab).
    # Same sources as sigma_dd above.
    "he4_p_2n3p": [
        (90.0,  0.5, 0.3),
        (300.0, 0.8, 0.5),
    ],

    # Meyer Table 9, sigma_pd,inel.
    # Total p-d inelastic (inelastic threshold 3.3 MeV).
    # Gap from 14.1 to 77 MeV — no tabulated measurements exist in Meyer.
    # Low-energy: Gibbons+Macklin 1959 (proton, direct); Cranberg+Smith 1959
    # (proton); Catron et al. 1961 (neutron, valid proxy above ~6 MeV);
    # Nisimura 1961 (proton); Shirato+Korri 1968 (neutron).
    # High-energy: Davison et al. 1963 (proton, CB); Bloom+Chamberlain 1954 +
    # Chamberlain+Stern 1954 (proton, CO); Esten et al. 1966 (proton, CW).
    "p_d_inel": [
        (5.5,   34.0,  1.7),
        (6.5,   65.0, 10.0),
        (7.32,  88.0,  9.0),
        (8.26, 110.0, 10.0),
        (9.0,  110.0,  6.0),
        (10.2, 140.0, 15.0),
        (14.1, 180.0,  7.0),
        # ~60 MeV gap here (no Table 9 entries between 14.1 and 77 MeV)
        (77.0,  84.0, 10.0),
        (95.0,  67.0,  9.0),
        (135.0, 62.2,  3.0),
    ],
}


def write_reaction(rxn: str, data: list):
    rxn_dir = NEWRUNS / rxn
    rxn_dir.mkdir(parents=True, exist_ok=True)

    label = f"{rxn}_1"

    # Raw data file (E, sigma, d_sigma) — same style as existing *_1.csv files
    raw_path = rxn_dir / f"{label}.csv"
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E", "sigma", "d_sigma"])
        for row in data:
            w.writerow(row)

    # Total XS file — what make_energy_bins.py reads
    total_path = rxn_dir / f"{rxn}_total_xs.csv"
    with open(total_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "E_MeV", "sigma_mb", "d_sigma_mb", "coverage_frac"])
        for E, sigma, d_sigma in data:
            w.writerow([label, E, sigma, d_sigma, "1.000"])

    print(f"  {rxn}: {len(data)} data points  -> {rxn_dir.relative_to(NEWRUNS.parent)}")


def main():
    print(f"Writing to {NEWRUNS}/")
    for rxn, data in REACTIONS.items():
        write_reaction(rxn, data)
    print("Done.")


if __name__ == "__main__":
    main()
