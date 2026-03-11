import numpy as np


class ThermonuclearOperator:
    """
    Operator that applies thermonuclear reaction source terms
    to the network state.

    This operator:
        - delegates reaction flux evaluation to the network
        - applies stoichiometry via the network
        - accumulates dY/dt contributions in the state

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

        # Delegate all reaction + stoichiometry logic to the network
        dYdt = self.network.change_in_abund(state)

        # Accumulate into state derivatives
        state.dY += dYdt
