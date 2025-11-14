# Famiano Model API and Code Specification


## Table of Contents

1. [Definitions and Variables](#definitions-and-variables) 
2. [Cloud Geometry, Evolution, and Composition](#cloud-geometry-evolution-and-composition)
3. [Jet Energy, Evolution, and Composition](#jet-energy-evolution-and-composition)
4. [Jet-Cloud Reaction Mechanisms](#jet-cloud-reaction-mechanism)
5. [Reaction Network](#reaction-network)
6. [Time-Stepping Scheme](#time-stepping-scheme)
7. [Survival Fractions](#survival-fractions)
8. [Yields](#yields)
9. [Data Inputs](#data-inputs)
10. [Output Quanitities](#output-quantities)
11. [Planned Code Module Mapping](#planned-code-module-mapping)

\newpage

## Definitions and Variables

This section contains a summary of all the variables, along with their definitions and units (if applicable). The variables used in the Famiano et al., 2002 paper were:

1. $Y_i$: Abundance per baryon belonging to species $i$. It is a unitless quantity.
2. \textbf{Y}: A vector containing the current abundances of all species used in the reaction network. It is unitless.
3. $f(\textbf{Y})$: Time rate of change (in $s^{-1}$) of the abundance of each species. This is given by the sum of the reactions that create and destroy each species.
4. h: A discrete time step (in seconds) taken in the network evolution.
5. $\epsilon$: Factor to ensure that the time step is small (therefore, the change in abundance is small per time step).
6. $E_0$: Initial energy of the jet. For consistency let's keep this in MeV.
7. $S(E,E_0)$: The fraction of particles in the jet that survive to energy, E. This is also called the survival fraction.
8. $N_m$: Abundance of the particles in the cloud participating in reaction $m$.
9. $\epsilon_i$: Stopping power of the incident particles in the medium. Stopping power tells us essentially how far the jet particles can travel in the cloud before coming to a complete stop.
10. $y_k$: Yield for a particle, k, produced by the reactions between cloud and jet particles.
11. $\zeta_{ik}$: Destruction fraction ($\zeta$) is the fraction of reacting jet particles with energy $E_k$ that are destroyed via reaction $i$.
12. $\sigma_{ik}$: The cross section for reaction $i$ and projectile energy $E_k$, is the average cross section for $E_{k-1} < E < E_k$.
13. $\phi_{pq}^{ik}$: Energy distribution tensor is defined to be the fraction of products p with energy $E_q$ from a reaction $i$ with a projectile that has energy $E_k$. For constant reaction type ($i$), projectile energy ($E_k$), and products ($p$), this tensor gives the energy distribution of product $p$ and is normalized to unity, i.e., $\sum_q \phi_{pq}^{ik} = 1$.
14. $y_{pq}^{in}$: Yield of particles $p$ with discrete energy $E_q$ per incident projectile with energy $E_n$ in reaction $i$ between jet and cloud.


## Cloud Geometry, Evolution, and Composition

There are three cloud models used:

1. Constant volume: $d\rho/dt = \dot{M}/V_0$
2. Constant density: $dV/dt = \dot{M}/\rho_0$
3. Variable volume and density:

$\frac{d(\rho V)}{dt} = \dot{M} \implies \rho V = \rho_0 V_0 (1 + \frac{\dot{M}}{M_0})$

where, $\rho = \rho_0 (1 + \dot{M}/M)^{0.5}$ and $V = V_0 (1 + \dot{M}/M)^{0.5}$

\textbf{Modeling notes:} User should have ability to choose a model. Need to update geometry after each timestep!

\textbf{Assumptions:} Most simulations assumed the cloud contained primordial initial number density fraction of elements. However, a few simulations seeded the cloud with an initial number of heavier isotopes.

## Jet Energy, Evolution, and Composition