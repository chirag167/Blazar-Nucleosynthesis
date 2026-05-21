#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_fresco_xs(path):
    text = Path(path).read_text(errors="ignore")

    # Match pairs like:
    #  10.00 deg.: X-S =   1.2042E+04 mb/sr,
    # +                                         /R =   7.4209E-01
    pattern = re.compile(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s+deg\.:"
        r"\s+X-S\s*=\s*([0-9.+\-Ee]+)\s+mb/sr,\s*\n"
        r"\+\s*/R\s*=\s*([0-9.+\-Ee]+)",
        re.MULTILINE,
    )

    rows = []
    for angle, xs, ratio in pattern.findall(text):
        rows.append(
            {
                "theta_deg": float(angle),
                "dsigma_domega_mb_sr": float(xs),
                "ratio_to_rutherford": float(ratio),
            }
        )

    if not rows:
        raise RuntimeError(f"No FRESCO cross-section block found in {path}")

    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_fresco_xs.py outputs/p_16_8_10MeV.out")
        raise SystemExit(1)

    path = sys.argv[1]
    df = parse_fresco_xs(path)

    csv_path = Path(path).with_suffix(".xs.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Plot raw differential cross section.
    plt.figure()
    plt.semilogy(df["theta_deg"], df["dsigma_domega_mb_sr"])
    plt.xlabel(r"$\theta_{\rm lab}$ (deg)")
    plt.ylabel(r"$d\sigma/d\Omega$ (mb/sr)")
    plt.title(Path(path).name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(path).with_suffix(".xs.png"), dpi=200)

    # Plot ratio to Rutherford.
    plt.figure()
    plt.plot(df["theta_deg"], df["ratio_to_rutherford"])
    plt.xlabel(r"$\theta_{\rm lab}$ (deg)")
    plt.ylabel(r"$\sigma / \sigma_{\rm Rutherford}$")
    plt.title(Path(path).name + "  /R")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(path).with_suffix(".ratio.png"), dpi=200)

    plt.show()


if __name__ == "__main__":
    main()