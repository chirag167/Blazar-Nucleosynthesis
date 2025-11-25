from pathlib import Path
import pandas as pd
from typing import Union

PathLike = Union[str, Path]

def load_data_file(
    filepath: PathLike,
    sep: str = r"\s+",
    header: int | None = 0,
) -> pd.DataFrame:
    """
    Load a whitespace- or delimiter-separated data file into a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the data file.
    sep : str, optional
        Field separator passed to `pandas.read_csv`. Defaults to r"\\s+".
    header : int or None, optional
        Row number to use as column names. Defaults to 0.

    Returns
    -------
    pandas.DataFrame
        The loaded data.
    """
    filepath = Path(filepath)
    return pd.read_csv(filepath, sep=sep, header=header)
