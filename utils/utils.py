import numpy as np

c = 3*10**8 # speed of light -- m/s
mass_num_to_mev = 931.5 # conversion from mass number to MeV/c^2
m_e = 0.511 # mass of electron -- MeV/c^2
m_p = 938.272 # mass of proton -- MeV/c^2

r_e = 2.818e-15 # electron radius in m

def beta(E, A):

    m = A * mass_num_to_mev

    v = c * np.sqrt(1 - 1 / (1 + E / m)**2 )

    beta = v/c

    return beta


def lorentz_factor(beta):

    gamma = 1 / np.sqrt(1 - beta**2)

    return gamma

