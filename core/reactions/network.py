import numpy as np


class ReactionNetwork:
    """
    Represents a complete nuclear reaction network.

    This class groups together:
        - the list of isotopes
        - the list of reactions
        - the stoichiometry matrix S

    It serves as the central bookkeeping object that connects
    reaction definitions to numerical solvers.

    Mathematical formulation
    -------------------------
    The abundance evolution is written as:

        dY/dt = S · R

    where:
        - Y is the abundance vector (size N)
        - R is the reaction flux vector (size M)
        - S is the stoichiometry matrix (N × M)

    This formulation ensures:
        - reaction ordering independence
        - automatic conservation-law enforcement
        - scalability to large networks
    """

    def __init__(self, isotopes, reactions):
        """
        Parameters
        ----------
        isotopes : list of str
            Names of nuclear species in the network.
            The index in this list defines the species index
            used throughout the code.

        reactions : list of Reaction
            List of Reaction objects defining the network topology.
        """

        self.isotopes = isotopes
        self.reactions = reactions

        # Number of species and reactions
        self.N = len(isotopes)
        self.M = len(reactions)

        # Build the stoichiometry matrix once at initialization
        self.S = self._build_stoichiometry()

    def _build_stoichiometry(self):
        """
        Construct the stoichiometry matrix S.

        S[i, r] is:
            - negative if species i is consumed in reaction r
            - positive if species i is produced in reaction r

        Returns
        -------
        numpy.ndarray
            Stoichiometry matrix of shape (N, M).
        """

        S = np.zeros((self.N, self.M))

        for r, rxn in enumerate(self.reactions):
            # Reactants contribute negative coefficients
            for i, nu in rxn.reactants:
                S[i, r] -= nu

            # Products contribute positive coefficients
            for i, nu in rxn.products:
                S[i, r] += nu

        return S
