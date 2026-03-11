# core/reactions/rate_laws.py
import numpy as np

def constant_rate(lamda_):
    def rate(state):
        return lamda_
    return rate


def simple_arrhenius(A, Q):
    """
    λ(T) = A * exp(-Q / T)
    """
    def rate(T):
        return A * np.exp(-Q / T)
    return rate
