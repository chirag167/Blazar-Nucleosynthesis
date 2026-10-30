#!/usr/bin/env python3
"""
integrate_xs.py
===============

Integrates dσ/dΩ angular distributions to produce total cross sections σ(E).

Sources processed for each reaction directory under newruns/:

  1. FRESCO .out files in <reaction>/outputs/ — one σ per energy file.
  2. Experimental .csv files in <reaction>/ — two cases:
       header 'theta' → angular distribution, integrated here.
       header 'E'     → already a total XS, written through unchanged.

The integration is:
    σ = 2π ∫ (dσ/dΩ) sin(θ) dθ   (θ in CM frame, radians)

Output: newruns/<reaction>/<reaction>_total_xs.csv
Columns: source, E_MeV, sigma_mb, d_sigma_mb, coverage_frac

coverage_frac is the fraction of [0°, 180°] spanned by the data.
Values below 1.0 mean σ is a lower bound (partial coverage).

Usage
-----
    python3 scripts/newscripts/integrate_xs.py
    python3 scripts/newscripts/integrate_xs.py --runs-dir newruns
    python3 scripts/newscripts/integrate_xs.py --reactions he4_p_d_he3 he4_he4_n_be7
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

# Resolve the project root from this script's location so the script works
# regardless of the working directory it is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Transfer channel definitions: reaction key -> (ejectile, residual) as FRESCO
# writes them in the output header.
TRANSFER_CHANNEL = {
    "he4_p_d_he3":   ("d",   "3He"),
    "he4_he4_n_be7": ("n",   "7Be"),
    "he4_he4_p_li7": ("p",   "7Li"),
    "li7_p_he4_he4": ("4He", "4He"),
}

_HEADER_RE = re.compile(
    r"CROSS SECTIONS FOR OUTGOING\s+(\S+)\s+&\s+(\S+)\s+in state",
    re.IGNORECASE,
)
_XS_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s+deg\.:\s+X-S\s*=\s*([0-9.+\-Ee*]+)\s+mb/sr,",
    re.MULTILINE,
)
_ENERGY_RE = re.compile(r"_(\d+(?:\.\d+)?)MeV\.out$", re.IGNORECASE)


def _safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def integrate(thetas_deg, xs_mb_sr):
    """
    σ = 2π ∫ (dσ/dΩ) sin(θ) dθ over the supplied angular grid.

    Returns (sigma_mb, coverage_frac) where coverage_frac = span / 180°.
    NaN or non-positive xs values are dropped before integration.
    """
    thetas = np.asarray(thetas_deg, dtype=float)
    xs     = np.asarray(xs_mb_sr,  dtype=float)
    mask   = np.isfinite(xs) & (xs > 0)
    thetas, xs = thetas[mask], xs[mask]
    if len(thetas) < 2:
        return float("nan"), 0.0
    thetas_rad  = np.radians(thetas)
    integrand   = xs * np.sin(thetas_rad)
    sigma       = 2.0 * math.pi * float(np.trapezoid(integrand, thetas_rad))
    coverage    = (float(thetas[-1]) - float(thetas[0])) / 180.0
    return sigma, coverage


def parse_fresco_block(path, ejec, resid):
    """Return [(theta_deg, xs_mb_sr), ...] for the named transfer block."""
    text    = path.read_text(errors="ignore")
    headers = list(_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        if m.group(1) == ejec and m.group(2) == resid:
            start = m.start()
            end   = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            chunk = text[start:end]
            return [(float(a), _safe_float(x)) for a, x in _XS_RE.findall(chunk)]
    return []


def energy_from_filename(path):
    m = _ENERGY_RE.search(path.name)
    return float(m.group(1)) if m else None


def fresco_total_xs(rdir, ejec, resid):
    """
    Integrate every .out file in rdir/outputs/ for the given channel.
    Returns sorted list of (E_MeV, sigma_mb, d_sigma_mb, coverage_frac).
    d_sigma is empty string (FRESCO gives no statistical uncertainty).
    """
    outdir = rdir / "outputs"
    if not outdir.is_dir():
        return []
    results = []
    for path in sorted(outdir.glob("*.out")):
        e = energy_from_filename(path)
        if e is None:
            continue
        pts = parse_fresco_block(path, ejec, resid)
        if not pts:
            continue
        thetas, xs = zip(*pts)
        sigma, cov = integrate(thetas, xs)
        if math.isfinite(sigma):
            results.append((e, sigma, "", cov))
    return sorted(results, key=lambda r: r[0])


def csv_total_xs(csv_path):
    """
    Read a CSV and return rows as (E_MeV, sigma_mb, d_sigma_mb, coverage_frac).

    'theta' header → integrate angular distribution; E_MeV will be '' (unknown).
    'E'     header → pass through; coverage_frac = 1.0.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
    if not rows:
        return []

    first_col = next(iter(rows[0])).strip().lower()

    if first_col == "theta":
        thetas  = [float(r["theta"])   for r in rows]
        xs      = [float(r["sigma"])   for r in rows]
        d_xs    = [float(r["d_sigma"]) for r in rows]
        sigma, cov      = integrate(thetas, xs)
        d_sigma, _      = integrate(thetas, d_xs)
        return [("", sigma, d_sigma, cov)]

    # E-based: already total XS
    return [
        (float(r["E"]), float(r["sigma"]), float(r["d_sigma"]), 1.0)
        for r in rows
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir",  default=str(_PROJECT_ROOT / "newruns"),
                    help="Root directory of reaction subdirs (default: <project>/newruns/)")
    ap.add_argument("--reactions", nargs="*", default=None,
                    help="Limit to these reaction keys (default: all)")
    args = ap.parse_args()

    runs_root  = Path(args.runs_dir)
    candidates = sorted(d for d in runs_root.iterdir() if d.is_dir())
    if args.reactions:
        candidates = [d for d in candidates if d.name in args.reactions]

    for rdir in candidates:
        reaction = rdir.name
        channel  = TRANSFER_CHANNEL.get(reaction)
        out_rows = []  # (source, E_MeV, sigma_mb, d_sigma_mb, coverage_frac)

        # --- FRESCO outputs ---
        if channel:
            ejec, resid = channel
            for e, sigma, d_sigma, cov in fresco_total_xs(rdir, ejec, resid):
                if cov < 0.9:
                    print(f"  [{reaction}] WARNING {e} MeV: coverage {cov:.2f} — integral is a lower bound")
                out_rows.append(("fresco", e, sigma, d_sigma, f"{cov:.3f}"))
        else:
            print(f"  [{reaction}] no transfer channel defined — FRESCO outputs skipped")

        # --- experimental CSVs ---
        out_name = f"{reaction}_total_xs.csv"
        for csv_path in sorted(rdir.glob("*.csv")):
            if csv_path.name == out_name:
                continue
            stem = csv_path.stem
            for e, sigma, d_sigma, cov in csv_total_xs(csv_path):
                if cov < 0.9:
                    print(f"  [{reaction}] WARNING {stem}: coverage {cov:.2f} — integral is a lower bound")
                out_rows.append((stem, e, sigma, d_sigma, f"{cov:.3f}"))

        if not out_rows:
            continue

        out_path = rdir / f"{reaction}_total_xs.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "E_MeV", "sigma_mb", "d_sigma_mb", "coverage_frac"])
            w.writerows(out_rows)
        print(f"  {reaction}: {len(out_rows)} rows -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
