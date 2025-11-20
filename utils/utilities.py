from scipy.interpolate import interp1d
import pandas as pd 

def interpolate_data(filepath,x_interp,i=0,j=1,sep='\s+',comments=None,header=None):

	'''Function to perform 1D interpolation between data points.

	INPUTS: filepath = the name of the file in which the data needs to be interpolated,
	x_interp = the x value at which the interpolation needs to happen.
	i,j = the column name or number at which the x and y data are stored. Default is 0,1
	sep = delimiter in the file.
	comments = if comments start with a symbol (like #) it will be ignored.
	header = if the file has a header, specify the row number.

	RETURNS: the interpolated value of the dataset in filepath at x_interp. '''

	df = pd.read_csv(filepath,sep=sep,comment=comments,header=header)
	print(df.head())

	x = df[i]

	y = df[j]

	y_interp = interp1d(x,y)

	interp_vals = y_interp(x_interp)

	return interp_vals

# testing the function

# hdp_interpolate = interpolate_data('../data/NACRE_2/2hdp3h.dat',0.33,comments='#')

# print(hdp_interpolate)

# ==================================================================================

import requests
from bs4 import BeautifulSoup
import re


url = "http://www.nuclear.csdb.cn/data/CF88/tables.html#008"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

# -----------------------------
# Convert reaction name → safe filename
# Example: "H3 (D,N) HE4" → "3hdn4he.dat"
# -----------------------------
def reaction_to_filename(name):
	
    # Remove spaces, commas, parentheses
    s = name.lower().replace(",", "").replace(" ", "").replace("+","").replace("-","")

    # Move numbers in isotope symbols:
    # H3 → 3h, HE4 → 4he, D → d, N → n, etc.
    # def fix_isotope(match):
    #     elem = match.group(1)
    #     num = match.group(2)
    #     return f"{num}{elem}"

    parts = re.findall(r'[a-z]+[0-9]*|\([a-z]+\)', s)
    fixed = []

    for p in parts:
        # isotope with number (e.g. he4)
        p_new = p.replace("(","").replace(")","")
        m = re.match(r'([a-z]+)([0-9]+)', p_new)
        if m:
            elem = m.group(1)
            num  = m.group(2)
            fixed.append(num + elem)
        else:
            fixed.append(p_new)  # plain letters: d, n

    return "".join(fixed) + ".dat"

    #return f"{s}.dat"

# -----------------------------
# Extract reaction blocks
# -----------------------------
blocks = soup.find_all("pre")

#target = "H3(D,N)"
for block in blocks:
    lines = block.text.strip().split("\n")

    # first non-empty line is reaction name
    reaction_line = ""
    idx = 0
    while idx < len(lines) and reaction_line.strip() == "":
        reaction_line = lines[idx].strip()
        idx += 1

    reaction_name = reaction_line

    #if target not in reaction_name:
    #    continue

    filename = reaction_to_filename(reaction_name)

    # Now extract t9–rate pairs after the title
    t9_vals = []
    rate_vals = []

    for line in lines[idx:]:
        parts = line.split()
        if len(parts) == 2:
            t9_vals.append(parts[0])
            rate_vals.append(parts[1])

    if len(t9_vals) == 0:
        continue  # skip irrelevant pre blocks

    # Save table
    df = pd.DataFrame({"t9": t9_vals, "reaction_rate": rate_vals})
    df.to_csv(filename, sep=" ", index=False, header=False)

    print("Saved:", filename)

