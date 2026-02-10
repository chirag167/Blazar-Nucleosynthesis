import numpy as np


def check_non_negative(Y, tol=1e-14):
    """
    Ensure all abundances are non-negative.

    Small negative values can arise from numerical roundoff,
    but large negative values indicate instability or bugs.

    Parameters
    ----------
    Y : array-like
        Abundance vector.

    tol : float
        Allowed negative tolerance.

    Raises
    ------
    ValueError
        If any abundance is more negative than -tol.
    """

    Y = np.asarray(Y)

    if np.any(Y < -tol):
        raise ValueError(
            "Negative abundance detected: numerical instability or invalid reaction rates."
        )
