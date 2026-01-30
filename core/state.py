import numpy as np

class NetworkState:
    def __init__(
        self,
        isotopes,
        Y0,
        temperature,
        density,
        volume,
        t0=0.0
    ):
        self.isotopes = isotopes            # list of isotope names
        self.Y = np.array(Y0, dtype=float)  # abundances Yi
        self.T = temperature
        self.rho = density
        self.V = volume

        self.t = t0

        # bookkeeping
        self.dY = np.zeros_like(self.Y)

    def reset_derivatives(self):
        self.dY[:] = 0.0

    def apply_update(self, dt):
        self.Y += self.dY * dt

    def compute_dt(self, safety=0.01,ymin=1e-12):
        """
        Mimics Eq. (4): limit timestep so fractional abundance
        change stays small.
        """
        mask = (self.Y > ymin) & (self.dY != 0.0)
        if not np.any(mask):
            return 1e-3  # fall back dt if no changes

        dt_candidates = safety * np.abs(self.Y[mask] / self.dY[mask])
        dt = np.min(dt_candidates)
        return max(dt,1e-12)  # avoid too small dt