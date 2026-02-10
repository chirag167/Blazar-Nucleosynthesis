from collections import defaultdict

class Reaction:
    """
    Represents a single nuclear reaction with fixed stoichiometry
    and a temperature/state-dependent reaction rate.

    A reaction is written in the generic form:

        sum_i (nu_i * X_i)  -->  sum_j (nu_j * Y_j)

    where:
        - X_i and Y_j are nuclear species
        - nu_i and nu_j are stoichiometric coefficients
        - the reaction rate is supplied externally via `rate_func`

    This class is intentionally lightweight:
        - it stores reaction topology (stoichiometry)
        - it does NOT perform any time integration
        - it does NOT evaluate reaction fluxes
    """

    def __init__(self, reactants, products, rate_func, name=None):
        """
        Parameters
        ----------
        reactants : list of (int, int)
            List of (species_index, stoichiometric_coefficient) pairs
            for reactants. Duplicate species entries are allowed and
            will be automatically combined.

        products : list of (int, int)
            Same as `reactants`, but for products.

        rate_func : callable
            Function that computes the reaction rate given the current
            network state. Typical usage:

                rate = rate_func(state)

            This design allows for:
                - temperature-dependent rates
                - density-dependent rates
                - tabulated or empirical rate laws

        name : str, optional
            Human-readable name for the reaction (useful for debugging
            and diagnostics).
        """

        # Store reactants and products in compressed (canonical) form
        # to avoid duplicate species entries and ensure correct bookkeeping
        self.reactants = self._compress(reactants)
        self.products = self._compress(products)

        # Store the rate function (physics is injected, not hard-coded)
        self.rate_func = rate_func

        # Optional name for logging / debugging
        self.name = name or "unnamed"

    def _compress(self, species):
        """
        Combine duplicate species entries by summing their
        stoichiometric coefficients.

        Example
        -------
        Input:
            [(0, 1), (0, 1), (2, 1)]

        Output:
            [(0, 2), (2, 1)]

        Parameters
        ----------
        species : list of (int, int)
            Raw list of (species_index, stoichiometric_coefficient) pairs.

        Returns
        -------
        list of (int, int)
            Canonicalized stoichiometry with unique species indices.
        """

        # defaultdict ensures missing keys default to zero
        d = defaultdict(int)

        # Accumulate stoichiometric coefficients for each species
        for i, nu in species:
            d[i] += nu

        # Convert dictionary back to list of (index, coefficient) pairs
        return list(d.items())
