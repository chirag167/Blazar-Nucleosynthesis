import numpy as np
from stopping import energy_loss, stopping_power

def survival(A_cl, X_cl, X_ion, Z, E, A, n_e, dt, T_e, S, n_target, sigma, eps=1e-2):
    '''
    Calculates the survival fraction of jet particles after interaction with the cloud.
    
    Inputs: 
    A_cl = array of all the mass numbers present in the cloud. 
    X_cl = fraction of each element in the cloud. 
    X_ion = ionization fraction in the cloud. 
    Z = atomic numbers of the elements in the cloud. 
    E = array of energies of the beam. 
    A = atomic mass number of the particles in the beam. 
    n_e = concentration of all atomic electrons in the cloud (1/cm^-3). 
    dt = timescale of energy loss (in s) -- determined from Euler step
    T_e = electron temperature 
    S = current survival fraction
    n_target = abundance of cloud particles for a particular reaction
    
    Source: Famiano et al., The Astrophysical Journal, 576:89-100, 2002 (eq. 4)
    
    Returns: the survival fraction of jet particles.
    
    '''

    # --- stopping power ---
    epsilon = stopping_power(A_cl, X_cl, X_ion, Z, E, A, n_e, dt, T_e, eps)  # MeV/cm

    # --- evolve energy ---
    dEdt, E_new = energy_loss(A_cl, X_cl, X_ion, Z,E, A, n_e, dt, T_e, eps)

    # energy step
    dE = np.abs(E_new - E)

    # Famiano integrand
    integrand = S * n_target * sigma / np.abs(epsilon)

    # exponential update
    S_new = 1 - np.sum(integrand * dE)

    return S_new