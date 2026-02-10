# core/reactions/rate_laws.py
import numpy as np

def constant_rate(lam):
    def rate(T):
        return lam
    return rate


def simple_arrhenius(A, Q):
    """
    λ(T) = A * exp(-Q / T)
    """
    def rate(T):
        return A * np.exp(-Q / T)
    return rate
