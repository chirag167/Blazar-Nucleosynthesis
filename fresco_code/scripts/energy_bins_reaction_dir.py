#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from math import pi, cos, radians, sqrt, floor

XS_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s+deg\.:\s+X-S\s*=\s*([0-9.+\-Ee]+)\s+mb/sr,"
)

ENERGY_RE = re.compile(
    r"LABORATORY\s+\S+\s+ENERGY\s*=\s*([0-9.+\-Ee]+)\s+MeV",
    re.IGNORECASE,
)

CHANNEL_RE = re.compile(
    r"CROSS SECTIONS FOR OUTGOING\s+([^\s&;]+)\s*&\s*([^\s&;]+)",
    re.IGNORECASE,
)

FRESCO_NAMES = {
    "n": "n",
    "p": "p",
    "d": "d",
    "t": "t",
    "he3": "3He",
    "he4": "4He",
    "li7": "7Li",
    "be7": "7Be",
}

AMU_TO_MEV = 931.49410242

MASSES_U = {
    "n": 1.00866491588,
    "p": 1.007276466621,
    "d": 2.013553212745,
    "t": 3.01550071621,
    "he3": 3.014932247175,
    "he4": 4.001506179127,
    "li7": 7.014357,
    "be7": 7.014735,
}

REACTIONS = {
    # Two-body final states
    "he4_he4_n_be7": {
        "type": "two_body",
        "projectile": "he4",
        "target": "he4",
        "ejectile": "n",
        "residual": "be7",
    },
    "he4_he4_p_li7": {
        "type": "two_body",
        "projectile": "he4",
        "target": "he4",
        "ejectile": "p",
        "residual": "li7",
    },
    "he4_p_d_he3": {
        "type": "two_body",
        "projectile": "p",
        "target": "he4",
        "ejectile": "d",
        "residual": "he3",
    },
    "p_he4_d_he3": {
        "type": "two_body",
        "projectile": "he4",
        "target": "p",
        "ejectile": "d",
        "residual": "he3",
    },
    "li7_p_he4_he4": {
        "type": "two_body",
        "projectile": "p",
        "target": "li7",
        "ejectile": "he4",
        "residual": "he4",
    },
    "p_li7_he4_he4": {
        "type": "two_body",
        "projectile": "li7",
        "target": "p",
        "ejectile": "he4",
        "residual": "he4",
    },

    # Multi-body final states
    "p_he4_pn_he3": {
        "type": "multi_body",
        "projectile": "he4",
        "target": "p",
        "products": ["p", "n", "he3"],
    },
    "p_d_pn_p": {
        "type": "multi_body",
        "projectile": "d",
        "target": "p",
        "products": ["p", "n", "p"],
    },
    "p_he4_2p_t": {
        "type": "multi_body",
        "projectile": "he4",
        "target": "p",
        "products": ["p", "p", "t"],
    },
    "p_he3_dp_p": {
        "type": "multi_body",
        "projectile": "he3",
        "target": "p",
        "products": ["d", "p", "p"],
    },
    "p_he3_ppn_p": {
        "type": "multi_body",
        "projectile": "he3",
        "target": "p",
        "products": ["p", "p", "n", "p"],
    },
    "p_he4_p_2d": {
        "type": "multi_body",
        "projectile": "he4",
        "target": "p",
        "products": ["p", "d", "d"],
    },
    "p_t_dp_n": {
        "type": "multi_body",
        "projectile": "t",
        "target": "p",
        "products": ["d", "p", "n"],
    },
    "p_he4_2p_dn": {
        "type": "multi_body",
        "projectile": "he4",
        "target": "p",
        "products": ["p", "p", "d", "n"],
    },
    "p_t_pnn_p": {
        "type": "multi_body",
        "projectile": "t",
        "target": "p",
        "products": ["p", "n", "n", "p"],
    },
    "p_he4_2n_3p": {
        "type": "multi_body",
        "projectile": "he4",
        "target": "p",
        "products": ["n", "n", "p", "p", "p"],
    },
    "he4_p_pn_he3": {
        "type": "multi_body",
        "projectile": "p",
        "target": "he4",
        "products": ["p", "n", "he3"],
    },
    "he4_p_2p_t": {
        "type": "multi_body",
        "projectile": "p",
        "target": "he4",
        "products": ["p", "p", "t"],
    },
    "he4_p_p_2d": {
        "type": "multi_body",
        "projectile": "p",
        "target": "he4",
        "products": ["p", "d", "d"],
    },
    "he4_p_2p_dn": {
        "type": "multi_body",
        "projectile": "p",
        "target": "he4",
        "products": ["p", "p", "d", "n"],
    },
    "he4_p_2n_3p": {
        "type": "multi_body",
        "projectile": "p",
        "target": "he4",
        "products": ["n", "n", "p", "p", "p"],
    },
}

def parse_fresco_xs(path, reaction):
    theta_deg = []
    dsigma = []
    E_lab = None

    if reaction["type"] != "two_body":
        raise ValueError(
            "Channel-aware FRESCO parsing currently requires a two-body reaction"
        )

    desired_particles = {
        FRESCO_NAMES[reaction["ejectile"]].lower(),
        FRESCO_NAMES[reaction["residual"]].lower(),
    }

    in_desired_channel = False
    found_headers = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            e_match = ENERGY_RE.search(line)
            if e_match:
                E_lab = float(e_match.group(1))

            channel_match = CHANNEL_RE.search(line)
            if channel_match:
                outgoing_1 = channel_match.group(1).strip()
                outgoing_2 = channel_match.group(2).strip()

                found_headers.append((outgoing_1, outgoing_2))

                current_particles = {
                    outgoing_1.lower(),
                    outgoing_2.lower(),
                }

                # Use sets so "n & 7Be" and "7Be & n" both match.
                in_desired_channel = current_particles == desired_particles
                continue

            if in_desired_channel:
                xs_match = XS_RE.match(line)
                if xs_match:
                    theta_deg.append(float(xs_match.group(1)))
                    dsigma.append(float(xs_match.group(2)))

    if not theta_deg:
        headers_text = ", ".join(
            f"{a} & {b}" for a, b in found_headers
        )

        raise RuntimeError(
            f"No cross-section data found for outgoing "
            f"{FRESCO_NAMES[reaction['ejectile']]} & "
            f"{FRESCO_NAMES[reaction['residual']]} in {path}. "
            f"Found channel headers: {headers_text or 'none'}"
        )

    if E_lab is None:
        raise RuntimeError(f"Could not find lab energy in {path}")

    pairs = sorted(zip(theta_deg, dsigma))
    theta_deg, dsigma = zip(*pairs)

    return list(theta_deg), list(dsigma), E_lab


def delta_sigmas(theta_deg, dsigma):
    out = []

    for i in range(len(theta_deg) - 1):
        th1 = radians(theta_deg[i])
        th2 = radians(theta_deg[i + 1])

        avg = 0.5 * (dsigma[i] + dsigma[i + 1])
        domega = 2.0 * pi * (cos(th1) - cos(th2))
        d_sigma = avg * domega

        theta_mid = 0.5 * (theta_deg[i] + theta_deg[i + 1])
        out.append((theta_mid, d_sigma))

    return out


def normalize(vals):
    total = sum(vals)
    if total <= 0:
        return [0.0 for _ in vals]
    return [v / total for v in vals]


def q_value_two_body(r):
    mi = MASSES_U[r["projectile"]] + MASSES_U[r["target"]]
    mf = MASSES_U[r["ejectile"]] + MASSES_U[r["residual"]]
    return (mi - mf) * AMU_TO_MEV


def q_value_multibody(r):
    mi = MASSES_U[r["projectile"]] + MASSES_U[r["target"]]
    mf = sum(MASSES_U[p] for p in r["products"])
    return (mi - mf) * AMU_TO_MEV


def product_energy_two_body(E_lab, theta_cm_deg, r):
    m_a = MASSES_U[r["projectile"]]
    m_A = MASSES_U[r["target"]]
    m_b = MASSES_U[r["ejectile"]]
    m_B = MASSES_U[r["residual"]]

    Q = q_value_two_body(r)
    E_cm = (m_A / (m_a + m_A)) * E_lab
    available = E_cm + Q

    if available < 0:
        return None

    theta = radians(theta_cm_deg)

    E_b_star = (m_B / (m_b + m_B)) * available
    E_b_cm_motion = (m_a * m_b / (m_a + m_A) ** 2) * E_lab

    return (
        E_b_star
        + E_b_cm_motion
        + 2.0 * sqrt(E_b_star * E_b_cm_motion) * cos(theta)
    )


def add_to_bin(bins, E, amount, bin_width):
    if E is None or E < 0:
        return

    idx = int(floor(E / bin_width))

    if 0 <= idx < len(bins):
        bins[idx] += amount


def make_multibody_eta(E_lab, r, bin_width, max_energy, mode):
    n_bins = int(max_energy / bin_width)

    m_a = MASSES_U[r["projectile"]]
    m_A = MASSES_U[r["target"]]

    E_cm = (m_A / (m_a + m_A)) * E_lab
    Q = q_value_multibody(r)
    E_avail = E_cm + Q

    if E_avail <= 0:
        return [0.0] * n_bins, Q, E_avail

    E_max = min(E_avail, max_energy)

    allowed = [q for q in range(n_bins) if q * bin_width < E_max]
    weights = [0.0] * n_bins

    if not allowed:
        return weights, Q, E_avail

    if mode == "flat":
        for q in allowed:
            weights[q] = 1.0

    elif mode == "high_enhanced":
        cutoff = allowed[int(0.75 * len(allowed))]
        for q in allowed:
            weights[q] = 2.0 if q >= cutoff else 1.0

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return normalize(weights), Q, E_avail


def process_one_outfile(path, reaction_key, bin_width, max_energy, mode):
    r = REACTIONS[reaction_key]

    theta_deg, dsigma, E_lab = parse_fresco_xs(path, r)
    d_bins = delta_sigmas(theta_deg, dsigma)
    sigma_tot = sum(d_sigma for _, d_sigma in d_bins)

    n_bins = int(max_energy / bin_width)
    outputs = {}

    if r["type"] == "two_body":
        product = r["ejectile"]
        sigma_bins = [0.0] * n_bins

        for theta_mid, d_sigma in d_bins:
            E_prod = product_energy_two_body(E_lab, theta_mid, r)
            add_to_bin(sigma_bins, E_prod, d_sigma, bin_width)

        outputs[product] = {
            "sigma_bins": sigma_bins,
            "eta": normalize(sigma_bins),
        }

        return {
            "file": str(path),
            "reaction": reaction_key,
            "type": "two_body",
            "method": "two_body_kinematics",
            "E_lab": E_lab,
            "sigma_tot": sigma_tot,
            "Q": q_value_two_body(r),
            "outputs": outputs,
        }

    if r["type"] == "multi_body":
        eta, Q, E_avail = make_multibody_eta(E_lab, r, bin_width, max_energy, mode)

        for product in sorted(set(r["products"])):
            multiplicity = r["products"].count(product)
            sigma_bins = [multiplicity * sigma_tot * e for e in eta]

            outputs[product] = {
                "sigma_bins": sigma_bins,
                "eta": eta,
            }

        return {
            "file": str(path),
            "reaction": reaction_key,
            "type": "multi_body",
            "method": f"multibody_{mode}",
            "E_lab": E_lab,
            "sigma_tot": sigma_tot,
            "Q": Q,
            "E_available_cm": E_avail,
            "outputs": outputs,
        }

    raise ValueError(f"Unsupported reaction type: {r['type']}")


def write_product_bins(path, sigma_bins, eta, bin_width):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_low_MeV", "bin_high_MeV", "sigma_bin_mb", "eta"])

        for q, (sigma_q, eta_q) in enumerate(zip(sigma_bins, eta)):
            writer.writerow([
                q * bin_width,
                (q + 1) * bin_width,
                sigma_q,
                eta_q,
            ])


def safe_energy_label(E):
    if abs(E - round(E)) < 1e-9:
        return f"{int(round(E))}MeV"
    return f"{E:g}MeV".replace(".", "p")


def process_reaction_dir(reaction_dir, reaction_key, bin_width, max_energy, mode):
    reaction_dir = Path(reaction_dir)

    outputs_dir = reaction_dir / "outputs"
    if not outputs_dir.exists():
        raise RuntimeError(f"Could not find outputs directory: {outputs_dir}")

    e_bins_dir = reaction_dir / "E_bins"
    e_bins_dir.mkdir(exist_ok=True)

    outfiles = sorted(outputs_dir.glob("*.out"))
    if not outfiles:
        raise RuntimeError(f"No .out files found in {outputs_dir}")

    summary_rows = []
    sigma_tot_rows = []

    for outfile in outfiles:
        try:
            result = process_one_outfile(
                path=outfile,
                reaction_key=reaction_key,
                bin_width=bin_width,
                max_energy=max_energy,
                mode=mode,
            )
        except Exception as exc:
            print(f"[SKIP] {outfile}: {exc}")
            continue

        E_label = safe_energy_label(result["E_lab"])
        base = outfile.stem

        for product, data in result["outputs"].items():
            out_csv = e_bins_dir / f"{base}_{E_label}_{product}_Ebins.csv"

            write_product_bins(
                path=out_csv,
                sigma_bins=data["sigma_bins"],
                eta=data["eta"],
                bin_width=bin_width,
            )

        summary_rows.append({
            "input_file": outfile.name,
            "reaction": result["reaction"],
            "type": result["type"],
            "method": result["method"],
            "E_lab_MeV": result["E_lab"],
            "Q_MeV": result["Q"],
            "sigma_tot_mb": result["sigma_tot"],
            "products": ";".join(result["outputs"].keys()),
        })

        sigma_tot_rows.append({
            "E_lab_MeV": result["E_lab"],
            "sigma_tot_mb": result["sigma_tot"],
        })

        print(
            f"[OK] {outfile.name}: E={result['E_lab']} MeV, "
            f"sigma_tot={result['sigma_tot']:.8e} mb"
        )

    summary_path = e_bins_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "input_file",
            "reaction",
            "type",
            "method",
            "E_lab_MeV",
            "Q_MeV",
            "sigma_tot_mb",
            "products",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    sigma_tot_path = e_bins_dir / "sigma_tot.csv"
    sigma_tot_rows.sort(key=lambda row: row["E_lab_MeV"])
    with open(sigma_tot_path, "w", newline="") as f:
        fieldnames = ["E_lab_MeV", "sigma_tot_mb"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sigma_tot_rows)

    print(f"Wrote sigma_tot values to: {sigma_tot_path}")
    print(f"\nWrote outputs to: {e_bins_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build product energy-bin distributions for all FRESCO .out files in one reaction directory."
    )

    parser.add_argument(
        "--reaction-dir",
        required=True,
        help="Reaction directory containing outputs/*.out",
    )

    parser.add_argument(
        "--reaction",
        required=True,
        choices=sorted(REACTIONS.keys()),
        help="Reaction key",
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=5.0,
        help="Energy bin width in MeV",
    )

    parser.add_argument(
        "--max-energy",
        type=float,
        default=400.0,
        help="Maximum product energy in MeV",
    )

    parser.add_argument(
        "--mode",
        choices=["flat", "high_enhanced"],
        default="flat",
        help="Approximation for multi-body final states",
    )

    args = parser.parse_args()

    process_reaction_dir(
        reaction_dir=args.reaction_dir,
        reaction_key=args.reaction,
        bin_width=args.bin_width,
        max_energy=args.max_energy,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()