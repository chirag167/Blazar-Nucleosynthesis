# Famiano Model API and Code Specification

## Table of Contents

1. [Definitions and Variables](#definitions-and-variables)
2. [Cloud Geometry, Evolution, and Composition](#cloud-geometry-evolution-and-composition)
3. [Jet Energy, Evolution, and Composition](#jet-energy-evolution-and-composition)
4. [Jet Normalization](#jet-normalization)
5. [Jet-Cloud Reaction Mechanisms](#jet-cloud-reaction-mechanisms)
6. [Reaction Network](#reaction-network)
7. [Time-Step Scheme](#time-step-scheme)
8. [Survival Fractions](#survival-fractions)
9. [Yields](#yields)
10. [Data Inputs](#data-inputs)
11. [Output Quantities](#output-quantities)
12. [Code Module Reference](#code-module-reference)
13. [Implementation Status](#implementation-status)

---

## Definitions and Variables

Summary of variables used in Famiano et al. (2002), their definitions, and units.

| Symbol | Description | Units |
|--------|-------------|-------|
| $Y_i$ | Abundance per baryon of species $i$ | dimensionless |
| $\boldsymbol{Y}$ | Vector of all species abundances | dimensionless |
| $f(\boldsymbol{Y})$ | Time rate of change of each species abundance | s$^{-1}$ |
| $h$ | Discrete timestep | s |
| $\epsilon$ | Safety factor for adaptive timestep ($\ll 1$; $\epsilon = 0.01$) | dimensionless |
| $E_0$ | Initial jet particle energy | MeV |
| $S(E, E_0)$ | Survival fraction: fraction of jet particles surviving to energy $E$ | dimensionless |
| $N_m$ | Abundance of cloud species participating in reaction $m$ | dimensionless |
| $\epsilon_i$ | Stopping power of incident jet particles in the cloud medium | MeV cm$^{-1}$ |
| $y_k$ | Yield of product particle $k$ from jet–cloud reactions | dimensionless |
| $\zeta_{ik}$ | Destruction fraction: fraction of jet particles at energy $E_k$ destroyed via reaction $i$ | dimensionless |
| $\sigma_{ik}$ | Cross section for reaction $i$ at projectile energy $E_k$ | mb |
| $\phi_{pq}^{ik}$ | Energy distribution tensor: fraction of products $p$ with energy $E_q$ from reaction $i$ at energy $E_k$ | dimensionless |
| $y_{pq}^{in}$ | Yield of particles $p$ with energy $E_q$ per incident projectile at energy $E_n$ in reaction $i$ | dimensionless |
| $\dot{M}$ | Jet mass deposition rate into the cloud | g s$^{-1}$ |
| $M_0$ | Total BLR mass | g |
| $\Delta M / M_0$ | Cumulative fractional mass increase of the BLR (Famiano's x-axis) | dimensionless |
| $f_\mathrm{norm}$ | Jet-to-cloud number density ratio $n_\mathrm{jet}/n_\mathrm{cloud}$ | dimensionless |

---

## Cloud Geometry, Evolution, and Composition

Three cloud evolution models are available:

1. **Constant volume:** $d\rho/dt = \dot{M}/V_0$
2. **Constant density:** $dV/dt = \dot{M}/\rho_0$
3. **Variable volume and density:**

$$\frac{d(\rho V)}{dt} = \dot{M} \quad \Rightarrow \quad \rho V = \rho_0 V_0 \left(1 + \frac{\dot{M}}{M_0}\right)$$

where $\rho = \rho_0 (1 + \dot{M}/M)^{0.5}$ and $V = V_0 (1 + \dot{M}/M)^{0.5}$.

### Cloud Model Parameters

| Parameter | Model B (BLR) | Model A (Knot) |
|-----------|--------------|----------------|
| $n$ (cm$^{-3}$) | $10^{11}$ | $10^{18}$ |
| $T$ (K) | $10^4$ | $6 \times 10^8$ |
| Ionization fraction | 0.001 | 1.0 |
| Length (cm) | $5 \times 10^{11}$ | $10^{11}$ |
| Diameter (cm) | $5 \times 10^{11}$ | $10^{11}$ |

### Initial Composition (Primordial)

| Species | $Y_i$ |
|---------|-------|
| p ($^1$H) | 0.750 |
| d ($^2$H) | $1.7 \times 10^{-5}$ |
| $^3$He | $1.0 \times 10^{-5}$ |
| $^4$He | 0.0625 |
| $^7$Li | $4.7 \times 10^{-10}$ |

Some models seed the cloud with additional heavier isotopes (labeled model variants "S").

### BLR Context

About $10^5$ clouds exist in the BLR at $T \approx 10^4$ K and $n \sim 10^{11}$–$10^{13}$ cm$^{-3}$. The average cloud size is ${\sim}400\;R_\odot$ ($\approx 2.8 \times 10^{13}$ cm), with a mass of ${\sim}10^{-5}\;M_\odot$. The total BLR mass is $M_0 \sim 10^6\;M_\odot$. Dense central knots can reach $T \sim 10^9$ K and $n \sim 10^{18}$ cm$^{-3}$ (Model A).

---

## Jet Energy, Evolution, and Composition

Jets are modeled with primordial composition:

- **Simple:** $^1$H [93%], $^4$He [7%] by number
- **Primordial:** $^1$H, $^2$H, $^3$He, $^4$He, $^6$Li, $^7$Li

Particles are injected at a fixed energy per nucleon (monoenergetic spectrum). The current default is **100 MeV/nucleon**, giving:

- Proton injection energy: 100 MeV
- $^4$He injection energy: 400 MeV
- $\beta = 0.428$, $\gamma = 1.107$ at 100 MeV/nucleon

Energy loss follows the fast-ion and slow-ion stopping power formulae from Ginzburg & Syrovatskii (1964), implemented in `core/stopping.py`.

Only species with $A < 8$ are treated as energetic (non-thermal) projectiles, per Famiano Section 3: p, n, d, t, $^3$He, $^4$He, $^6$Li, $^7$Li.

---

## Jet Normalization

The spectrum normalization converts the monoenergetic injection spectrum from arbitrary units to physical number densities relative to the cloud.

### Derivation

The steady-state number density of jet particles in the cloud is:

$$n_\mathrm{jet} = \frac{\Phi}{v_\mathrm{jet}}$$

where the particle flux through the cloud face is:

$$\Phi = \frac{\dot{M}/m_\mathrm{avg}}{A_\mathrm{cloud}}, \quad A_\mathrm{cloud} = \pi (d/2)^2$$

with $m_\mathrm{avg}$ the mean jet particle mass and $v_\mathrm{jet} = \beta c$.

The dimensionless normalization factor is:

$$f_\mathrm{norm} = \frac{n_\mathrm{jet}}{n_\mathrm{cloud}}$$

Each species' injection bin value is set to $f_\mathrm{norm} \times \chi_\mathrm{sp}$, where $\chi_\mathrm{sp}$ is the species number fraction.

### Model B Values

| Quantity | Value |
|----------|-------|
| $\dot{M}$ | $10^{-6}\;M_\odot\;\mathrm{yr}^{-1} = 6.30 \times 10^{19}\;\mathrm{g\;s}^{-1}$ |
| $\Phi$ | $1.60 \times 10^{20}\;\mathrm{cm}^{-2}\;\mathrm{s}^{-1}$ |
| $n_\mathrm{jet}$ | $1.24 \times 10^{10}\;\mathrm{cm}^{-3}$ |
| $n_\mathrm{cloud}$ | $10^{11}\;\mathrm{cm}^{-3}$ |
| $f_\mathrm{norm}$ | $0.1245$ |

### $\Delta M / M_0$ Tracking

Famiano's x-axis is the **cumulative fractional mass** of jet material deposited into the entire BLR:

$$\frac{\Delta M}{M_0}(t) = \frac{\dot{M} \cdot t}{M_0}$$

with $M_0 = 10^6\;M_\odot$ (total BLR mass). This gives $\Delta M / M_0 \sim 10^{-8}$ at $t \sim 10^4$ s for Model B. Note: this is **not** relative to a single clump mass.

---

## Jet-Cloud Reaction Mechanisms

Jet material introduced into the cloud is assumed to be well-mixed following thermalization. Cloud particles are homogeneously distributed and have the same energy per nucleon initially.

$$\dot{M} = \dot{m}_\mathrm{jet} \times \Omega$$

where $\dot{M}$ is the cloud mass rate of change, $\dot{m}_\mathrm{jet}$ is the jet mass outflow rate, and $\Omega$ is the jet cross-sectional area intersecting with the cloud. $\dot{M}$ ranges from ${\sim}10^{-6}\;M_\odot\;\mathrm{yr}^{-1}$ (BLR) to ${\sim}10\;M_\odot\;\mathrm{yr}^{-1}$ (knot).

Reactions within the jet and between jet and cloud are treated separately because they do not follow the Boltzmann energy distribution.

### Reaction Groups

- **Group 1:** Jet particles reacting with dominant cloud species (p, $^4$He targets). Includes p+p, p+$^4$He, and $^4$He+$^4$He channels. Product energy distributions require DWBA calculations. *Currently disabled pending DWBA output.*
- **Group 2:** Jet particles reacting with heavier cloud species (Li, Be, B, C, N, O targets). Binary kinematics allow Q-value based cross sections. *Currently enabled; 46 channels loaded.*

### Secondary Cascade

Products with $A < 8$ (n, d, t, $^3$He, $^4$He, $^6$Li, $^7$Li) can in principle re-enter the cascade as secondary non-thermal projectiles. Currently, `update_spectra=False` — the jet spectrum is reset to steady-state injection values each step, so secondaries do not re-enter. This is a planned enhancement:

- Group-2 secondaries: implementable via two-body kinematics (no DWBA needed)
- Group-1 secondaries: requires DWBA product energy distributions

---

## Reaction Network

The implicit Euler update for the abundance vector is:

$$\left(\frac{\tilde{I}}{h} - \tilde{J}\right) \cdot \Delta = f[\boldsymbol{Y}(t)]$$

where $\tilde{J}$ is the Jacobian matrix:

$$\tilde{J} = J_{i,j} = \frac{\partial f(Y_i)}{\partial Y_j}$$

The current implementation uses **explicit Euler** (adequate for Model B, non-stiff). The Jacobian module (`core/jacobian.py`) supports implicit integration for Model A (stiff thermonuclear network).

### Reaction Threshold

For endothermic reactions ($Q < 0$), the lab-frame threshold is:

$$E_\mathrm{thr} \approx -Q \left(1 + \frac{m_a}{m_A}\right)$$

computed at load time. Reactions are suppressed below threshold.

### CM-Frame Energy Conversion

Cross section data files with CM-frame energies are automatically converted to lab-frame at load time using the relativistic invariant-mass relation:

$$T_\mathrm{lab} = \frac{s - m_a^2 - m_A^2}{2 m_A} - m_a, \quad s = (T_\mathrm{cm} + m_a + m_A)^2$$

Total cross sections are Lorentz invariant — only the energy axis transforms.

---

## Time-Step Scheme

The adaptive timestep (Famiano eq. 4) is:

$$h_{n+1} = \epsilon \cdot h_n \cdot \min_i \left(\frac{Y_i^{n+1}}{\Delta_i}\right)$$

where $\Delta_i = Y_i^{n+1} - Y_i^n$ is the abundance change per step and $\epsilon = 0.01$.

This simplifies to $h_{n+1} = \epsilon \cdot \tau_\mathrm{min}$ where $\tau_\mathrm{min} = \min_i(Y_i / |\dot{Y}_i|)$ is the shortest depletion timescale. A `max_growth = 5` cap prevents sudden jumps.

### Current Run Parameters (Model B)

| Parameter | Value |
|-----------|-------|
| $t_0$ | 0 s |
| $t_\mathrm{max}$ | $10^5$ s |
| $h_\mathrm{min}$ | $10^{-2}$ s |
| $h_\mathrm{max}$ | $10^4$ s |
| $\epsilon$ | 0.01 |
| max steps | 100,000 |

---

## Survival Fractions

For species $i$ in the jet with initial energy $E_0$, the fraction of particles surviving to energy $E$ is:

$$S_i(E, E_0) = 1 - \int_E^{E_0} S(E') \frac{\sum_m N_m \sigma_m(E')}{\epsilon_i(E')} \, dE'$$

If a particle has initial energy $E_1 < E_0$, its survival fraction to $E_2 < E_1$ is:

$$S(E_2, E_1) = \frac{S_i(E_2, E_0)}{S_i(E_1, E_0)}$$

---

## Yields

The yield of particle $k$ from all jet–cloud interactions for projectiles slowing from $E_0$ through the range $E_1 < E < E_2$ is:

$$y_k = N_i(E_0) \int_{E_1}^{E_2} S(E, E_0) \frac{\sum_m \sigma_m(E)}{\epsilon_i(E)} \, dE$$

The discretized survival fraction increment is:

$$\delta S_{ji} = S_j(E_i, E_0) - S_j(E_{i-1}, E_0), \quad \delta S_{j1} = S_j(E_1, E_0)$$

The destruction fraction is:

$$\zeta_{ik} = \frac{\sigma_{ik} N_i}{\sum_m \sigma_{mk} N_m}$$

The discretized yield tensor is:

$$y_{pq}^{in} = \phi_{pq}^{ik} \, \zeta_{ik} \, \frac{\delta S_{ik}}{S_{in}}$$

---

## Data Inputs

### Cross-Section Data — `data/CrossSections/`

Cross sections are stored as CSV files with the naming convention:

```
{target}_{projectile}{products}_{dataset_index}.csv
```

Example: `7Li_pa_4He_4.csv` = $^7$Li(p,$\alpha$)$^4$He, dataset 4.

| Directory | Contents |
|-----------|----------|
| `Group1/` | p+p, p+$^4$He, $^4$He+$^4$He channels (DWBA required) |
| `Group2/` | Heavy-target channels: Li, Be, B, C, N, O (46 channels loaded) |

Supported energy column names: `E_MeV`, `E_keV`, `E_cm`, `E_cm_kev`, `E_min`/`E_max` (bin edges → midpoints used).

Supported cross-section column names and units: `sigma_mb` (mb), `sigma_b` (b), `sigma_ub` (μb), `sigma_nb` (nb), `sigma_cm_B` (barn, CM-frame label, treated as mb×10³ → converted).

CM-frame energy files are automatically converted to lab frame on load if projectile and target species are parseable from the filename.

### Mass Table — `data/mass_table.json`

Atomic masses in unified atomic mass units (u), AME 2020. Used for Q-value computation and CM→lab energy conversion. The built-in table covers 32 nuclides (n through $^{21}$Ne); an external JSON file can extend it.

### Configuration Files — `config/`

| File | Contents |
|------|----------|
| `cloud.json` | Cloud target properties (density, temperature, composition, geometry, BLR mass) |
| `jet.json` | Jet species, spectrum type, injection energy, mass rate |
| `species.json` | Nuclear data ($A$, $Z$) for all species in the network |
| `run.json` | Solver settings (timestep bounds, $t_\mathrm{max}$), energy grid, output paths |

---

## Output Quantities

| File | Contents |
|------|----------|
| `outputs/abundance_history.csv` | Step, $t$ (s), $\Delta M/M_0$, $Y_i(t)$ for all species at every `write_every_n_steps` interval |
| `outputs/final_state.json` | Final $t$, step count, $\Delta M/M_0$, all $Y_i$, mass fractions, stop reason |
| `outputs/famiano_abundance_evolution.png` | Log–log abundance evolution plot; bottom x-axis = time [s], top x-axis = $\Delta M/M_0$, y-axis = $Y_i$ |

---

## Code Module Reference

The simulation is organized into the following directories and files.

### `core/` — Physics Engine

| File | Responsibility |
|------|----------------|
| `state.py` | Data containers: `CloudState`, `CascadeState`, `SolverState`, `NetworkState`, `SpeciesData`, `ProjectileSpectrum` |
| `reactions.py` | `CrossSectionTable`, `Reaction`, `ReactionLibrary`: load, parse, and interpolate cross-section data; CM→lab conversion; threshold computation |
| `cascade.py` | `run_cascade_step()`: computes $dY/dt$ for all cloud species from jet-cloud reactions using the loaded reaction library |
| `timestep.py` | `compute_next_dt()`, `estimate_initial_dt()`, `euler_increment()`: Famiano eq. (4) adaptive timestep |
| `stopping.py` | Stopping power $\epsilon_i(E)$ via fast/slow ion formulae (Ginzburg & Syrovatskii 1964) |
| `survival.py` | Survival fraction $S_i(E, E_0)$ integration |
| `grids.py` | `make_energy_grid()`: linear or logarithmic energy bin construction |
| `jacobian.py` | Jacobian matrix $J_{ij}$ for implicit Euler integration (used for Model A) |
| `io.py` | Output helpers: CSV row writing, JSON serialization |

### `scripts/` — Entry Points

| File | Responsibility |
|------|----------------|
| `run_famiano.py` | Main driver: load config, build initial state, expand cloud with reaction products, compute jet normalization, run adaptive Euler loop, write CSV and JSON outputs |
| `plot_famiano.py` | Post-processing: read `abundance_history.csv`, produce single-panel log–log Famiano-style abundance evolution figure |

### `utils/` — Shared Utilities

**`utils/utils.py`** provides:

- `beta(E, A)`, `lorentz_factor(beta)`: relativistic kinematics helpers
- `canonical_species_name(species)`: normalizes species name strings (e.g., `"he4"` → `"4He"`)
- `load_mass_table(path)`: loads AME 2020 atomic masses from JSON; falls back to 32-nuclide built-in table
- `q_value_mev(reactants_stoich, products_stoich)`: Q-value from atomic masses
- `reaction_threshold_lab_mev(projectile, target, q_mev)`: non-relativistic lab-frame threshold for endothermic reactions
- `cm_energy_to_lab_mev(t_cm_mev, m_projectile_mev, m_target_mev)`: relativistic CM-to-lab energy conversion using Mandelstam $s$
- `cm_to_lab_energy_mev(t_cm_mev, projectile, target)`: species-name wrapper for the above

### `nonthermal/` — Legacy Survival Fraction Code

`nonthermal/survival.py`: earlier standalone implementation of the survival fraction integral. Superseded by `core/survival.py` but retained for reference.

### `config/` — Simulation Configuration

See [Data Inputs — Configuration Files](#configuration-files----config) above.

### `data/` — Cross Sections and Mass Data

See [Data Inputs — Cross-Section Data](#cross-section-data----datacrosssections) above.

### `outputs/` — Generated Outputs

Runtime output directory (not tracked in version control except `.gitkeep`). See [Output Quantities](#output-quantities) above.

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Group-2 non-thermal reactions (heavy targets) | **Done** | 46 channels; CM→lab conversion; threshold enforcement |
| Adaptive explicit Euler timestep | **Done** | Famiano eq. (4); $h_\mathrm{min}$ / $h_\mathrm{max}$ bounds; max growth cap |
| Jet normalization ($f_\mathrm{norm}$) | **Done** | Physical $n_\mathrm{jet}/n_\mathrm{cloud}$ from mass rate and cloud geometry |
| $\Delta M/M_0$ tracking | **Done** | Relative to total BLR mass $M_0 = 10^6\;M_\odot$ |
| Cloud species expansion | **Done** | All reaction products pre-seeded at $Y=0$ before evolution |
| CM→lab energy conversion | **Done** | Relativistic; auto-applied on load for CM-frame data files |
| Reaction threshold enforcement | **Done** | Endothermic reactions suppressed below $E_\mathrm{thr}$ |
| Q-value computation | **Done** | AME 2020 masses; 32 nuclides built-in |
| Famiano-style abundance plot | **Done** | Log–log; time [s] bottom axis; $\Delta M/M_0$ top axis |
| Group-1 reactions (p+p, p+$^4$He) | **Pending** | Requires DWBA product energy distributions |
| Secondary non-thermal cascade | **Pending** | Group-2 secondaries: two-body kinematics sufficient; Group-1: needs DWBA |
| Implicit Euler integration | **Pending** | Jacobian implemented; needed for stiff Model A thermonuclear network |
| Thermonuclear reactions (Model A) | **Pending** | `SimpleThermoNuclearOperator` stub exists; rates not connected |
| Cloud geometry update per timestep | **Pending** | Cloud volume/density assumed constant (adequate for Model B) |
| Missing/incomplete cross-section files | **Pending** | Several files lack energy columns or have arbitrary-unit $\sigma$; listed in loader warnings |
