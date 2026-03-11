from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

PathLike = Union[str, Path]

def interpolate_data(
    filepath: PathLike,
    x_interp: float,
    x_col: int | str = 0,
    y_col: int | str = 1,
    sep: str = r"\s+",
    comments: str | None = None,
    header: int | None = None,
    kind: str = "linear",
) -> float:
    """
    Perform 1D interpolation on a two-column dataset.

    Parameters
    ----------
    filepath : str or Path
        Input data file.
    x_interp : float
        X value at which to interpolate.
    x_col, y_col : int or str
        Column indices or names for x and y.
    sep, comments, header : see pandas.read_csv
    kind : str
        Interpolation kind, passed to scipy.interpolate.interp1d.

    Returns
    -------
    float
        Interpolated y(x_interp).
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, sep=sep, comment=comments, header=header)

    x = df.iloc[:, x_col] if isinstance(x_col, int) else df[x_col]
    y = df.iloc[:, y_col] if isinstance(y_col, int) else df[y_col]

    f = interp1d(x.to_numpy(), y.to_numpy(), kind=kind, bounds_error=True)
    return float(f(x_interp))


# testing the function

# hdp_interpolate = interpolate_data('../data/NACRE_2/2hdp3h.dat',0.33,comments='#')

# print(hdp_interpolate)

# ==================================================================================


import requests 
from bs4 import BeautifulSoup
import re
from pathlib import Path


def extract_cf88_tables(url: str, name: str, outdir: str | Path) -> Path | None:
    """
    Download a CF88 tables page, find the table for a given reaction name,
    save its (t9, reaction_rate) data to a .dat file in `outdir`,
    and return the path to that file.

    Parameters
    ----------
    url : str
        URL of the CF88 tables page (e.g. 'http://www.nuclear.csdb.cn/data/CF88/tables.html#008').
    name : str
        Reaction name exactly as it appears in the heading (e.g. 'H3 (D,N) HE4').
    outdir : str or Path
        Directory where the output .dat file should be written.

    Returns
    -------
    Path or None
        Path to the saved file, or None if the reaction table was not found / parsed.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) Download and parse HTML
    # -----------------------------
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")

    # -----------------------------
    # 2) Find the <pre> block for this reaction
    #    (assumes the first line of the block contains `name`)
    # -----------------------------
    target_pre = None
    target_lower = name.lower()

    for pre in soup.find_all("pre"):
        text = pre.get_text()
        lines = text.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip().lower()
        if target_lower in first_line:
            target_pre = pre
            break

    if target_pre is None:
        print(f"[extract_cf88_tables] Reaction '{name}' not found at {url}")
        return None

    # -----------------------------
    # 3) Parse t9 + rate from that block
    # -----------------------------
    lines = target_pre.get_text().splitlines()
    t9_vals = []
    rate_vals = []

    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            t9 = float(parts[0])
            rate = float(parts[1])
        except ValueError:
            # skip non-numeric lines (headers etc.)
            continue
        t9_vals.append(t9)
        rate_vals.append(rate)

    if not t9_vals:
        print(f"[extract_cf88_tables] No numeric rate data parsed for '{name}' at {url}")
        return None

    df = pd.DataFrame({"t9": t9_vals, "reaction_rate": rate_vals})

    # -----------------------------
    # 4) Build the filename from the reaction name (your logic)
    #    Example: 'H3 (D,N) HE4' -> '3hdn4he.dat'
    # -----------------------------
    s = name.lower().replace(",", "").replace(" ", "").replace("+", "").replace("-", "")

    # Split into tokens like 'h3', '(dn)', 'he4'
    parts = re.findall(r"[a-z]+[0-9]*|\([a-z]+\)", s)
    fixed = []

    for p in parts:
        p_new = p.replace("(", "").replace(")", "")
        m = re.match(r"([a-z]+)([0-9]+)", p_new)
        if m:
            elem = m.group(1)
            num = m.group(2)
            fixed.append(num + elem)  # h3 -> 3h, he4 -> 4he
        else:
            fixed.append(p_new)       # plain letters: d, n, etc.

    filename = "".join(fixed) + ".dat"
    outpath = outdir / filename

    # -----------------------------
    # 5) Save to .dat file
    # -----------------------------
    df.to_csv(outpath, sep=" ", index=False, header=False)
    print(f"[extract_cf88_tables] Saved {name!r} to {outpath}")

    return outpath


# import requests 
# from bs4 import BeautifulSoup
# import re
# from pathlib import Path


# #url = "http://www.nuclear.csdb.cn/data/CF88/tables.html#008"

# # -----------------------------
# # Convert reaction name → safe filename
# # Example: "H3 (D,N) HE4" → "3hdn4he.dat"
# # -----------------------------
# def extract_cf88_tables(url:str, name:str, outdir=str | Path):
    
#     # Remove spaces, commas, parentheses
#     html = requests.get(url).text
#     soup = BeautifulSoup(html, "html.parser")

#     s = name.lower().replace(",", "").replace(" ", "").replace("+","").replace("-","")

#     # Move numbers in isotope symbols:
#     # H3 → 3h, HE4 → 4he, D → d, N → n, etc.
#     # def fix_isotope(match):
#     #     elem = match.group(1)
#     #     num = match.group(2)
#     #     return f"{num}{elem}"

#     parts = re.findall(r'[a-z]+[0-9]*|\([a-z]+\)', s)
#     fixed = []

#     for p in parts:
#         # isotope with number (e.g. he4)
#         p_new = p.replace("(","").replace(")","")
#         m = re.match(r'([a-z]+)([0-9]+)', p_new)
#         if m:
#             elem = m.group(1)
#             num  = m.group(2)
#             fixed.append(num + elem)
#         else:
#             fixed.append(p_new)  # plain letters: d, n

#     return "".join(fixed) + ".dat"

    #return f"{s}.dat"

# -----------------------------
# Extract reaction blocks
# -----------------------------
# blocks = soup.find_all("pre")

# #target = "H3(D,N)"
# for block in blocks:
#     lines = block.text.strip().split("\n")

#     # first non-empty line is reaction name
#     reaction_line = ""
#     idx = 0
#     while idx < len(lines) and reaction_line.strip() == "":
#         reaction_line = lines[idx].strip()
#         idx += 1

#     reaction_name = reaction_line

#     #if target not in reaction_name:
#     #    continue

#     filename = reaction_to_filename(reaction_name)

#     # Now extract t9–rate pairs after the title
#     t9_vals = []
#     rate_vals = []

#     for line in lines[idx:]:
#         parts = line.split()
#         if len(parts) == 2:
#             t9_vals.append(parts[0])
#             rate_vals.append(parts[1])

#     if len(t9_vals) == 0:
#         continue  # skip irrelevant pre blocks

#     # Save table
#     df = pd.DataFrame({"t9": t9_vals, "reaction_rate": rate_vals})
#     df.to_csv(filename, sep=" ", index=False, header=False)

#     print("Saved:", filename)

