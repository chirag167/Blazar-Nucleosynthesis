import numpy as np


class ThermonuclearOperator:
    """
    Operator that applies thermonuclear reaction source terms
    to the network state.

    This operator:
        - evaluates reaction fluxes R_r
        - applies the stoichiometry matrix
        - updates dY/dt for all species

    It does NOT:
        - control time stepping
        - modify the state directly
        - assume any specific reaction physics

    All nuclear physics enters via Reaction.rate_func.
    """

    def __init__(self, network):
        """
        Parameters
        ----------
        network : ReactionNetwork
            The reaction network containing reactions and
            stoichiometric information.
        """
        self.network = network

    def apply(self, state, dt):
        """
        Compute and accumulate abundance derivatives dY/dt
        due to thermonuclear reactions.

        Parameters
        ----------
        state : NetworkState
            Current thermodynamic and abundance state.

        dt : float
            Time step (unused here, but included for a uniform
            operator interface).
        """

        # Compute reaction fluxes for all reactions
        rates = []

        for rxn in self.network.reactions:
            # Evaluate the reaction rate (can depend on T, rho, time, etc.)
            lam = rxn.rate_func(state)

            # Compute reaction flux:
            # R = λ × Π_i Y_i^{ν_i}
            R = lam
            for i, nu in rxn.reactants:
                R *= state.Y[i] ** nu

            rates.append(R)

        rates = np.array(rates)

        # Apply stoichiometry: dY += S · R
        state.dY += self.network.S @ rates
