import numpy as np


def check_baryon_conservation(A, network, tol=1e-12):
    """
    Check baryon-number conservation for the reaction network.

    Baryon conservation requires:

        A · S = 0

    where:
        - A is the vector of mass numbers
        - S is the stoichiometry matrix

    Parameters
    ----------
    A : array-like
        Mass numbers of each species (length N).

    network : ReactionNetwork
        Reaction network containing the stoichiometry matrix.

    tol : float
        Numerical tolerance for the conservation check.

    Returns
    -------
    bool
        True if baryon number is conserved for all reactions.
    """

    A = np.asarray(A)
    return np.allclose(A @ network.S, 0.0, atol=tol)
