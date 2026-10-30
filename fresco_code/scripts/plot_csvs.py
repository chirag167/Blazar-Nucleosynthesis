from pathlib import Path
import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt


def energy_sort_key(path):
    match = re.search(r"(\d+(?:\.\d+)?)\s*MeV", path.name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return float("inf")


def extract_numeric_table(path):
    """
    Reads a messy .xs/.out/.csv-like file and extracts rows that begin
    with at least two numeric values.

    Returns a DataFrame with columns:
    theta, value
    """
    rows = []

    number_pattern = re.compile(
        r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[Ee][-+]?\d+)?"
    )

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            numbers = number_pattern.findall(line)

            if len(numbers) >= 2:
                try:
                    x = float(numbers[0])
                    y = float(numbers[1])
                    rows.append((x, y))
                except ValueError:
                    continue

    if not rows:
        raise ValueError(f"No numeric x/y data found in {path.name}")

    return pd.DataFrame(rows, columns=["theta", "value"])


def plot_all_files(
    directory,
    pattern="*.csv",
    output_name=None,
    log_y=True,
    x_label=r"$\theta_{\mathrm{lab}}$ (deg)",
    y_label=r"$d\sigma/d\Omega$ (mb/sr)",
):
    directory = Path(directory)

    files = sorted(
        [p for p in directory.glob(pattern) if p.is_file()],
        key=energy_sort_key
    )

    if not files:
        raise FileNotFoundError(
            f"No files found in {directory} matching pattern '{pattern}'"
        )

    plt.figure(figsize=(12, 7))

    for file_path in files:
        df = extract_numeric_table(file_path)

        df = df.dropna()
        df = df.sort_values("theta")

        if log_y:
            df = df[df["value"] > 0]

        if df.empty:
            print(f"Skipping {file_path.name}: no plottable positive y-values")
            continue

        plt.plot(
            df["theta"],
            df["value"],
            linewidth=2,
            label=file_path.stem
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Combined Cross Section Comparison")

    if log_y:
        plt.yscale("log")

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    if output_name is None:
        if directory.name == "outputs":
            run_name = directory.parent.name
        else:
            run_name = directory.name

        output_name = f"{run_name}_combined_plots.png"

    output_path = directory / output_name
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Combine .xs/.out/CSV-style numeric data files into one plot."
    )

    parser.add_argument(
        "directory",
        help="Path to the directory containing the data files."
    )

    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to include, for example '*.xs', '*.out', or '*.csv'. Defaults to '*.csv"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG filename. Defaults to '<run_name>_combined_plots.png'."
    )

    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of a logarithmic y-axis."
    )

    args = parser.parse_args()

    plot_all_files(
        directory=args.directory,
        pattern=args.pattern,
        output_name=args.output,
        log_y=not args.linear_y
    )


if __name__ == "__main__":
    main()