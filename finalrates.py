import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
cross_section_data = pd.read_csv('crosssections.csv')     # sigma_m2
probability_data = pd.read_csv('Th Probability.csv')       # results = P(E)
spectrum_data = pd.read_csv('lambda.csv')                  # lambda = flux(E)

# Merge on common 'energy' column (in MeV)
df = cross_section_data.merge(probability_data, on='energy') \
                       .merge(spectrum_data, on='energy')

flux_total = 5.05e6  # cm^-2 s^-1
det_mass_kton = 22.5
N_A = 6.022e23
electrons_per_molecule = 10
mol_weight_water = 18.015  # g/mol
# Number of electrons per kton
n_e_per_kton = (1e6 / mol_weight_water) * electrons_per_molecule * N_A  # per kton
N_e = n_e_per_kton * det_mass_kton

seconds_per_day = 86400
bin_width = 0.5  # MeV

# Rate = P(E) × sigma(E) × λ(E) × total_flux × N_e × time
df['rate'] = df['results'] * df['sigma_m2'] * df['lambda'] * N_e * seconds_per_day / bin_width

# Treat energy as recoil kinetic energy
df['T_e'] = df['energy']  # recoil electron energy ≈ neutrino energy

# Save computed rates
df.to_csv('rates.csv', index=False)

# Plot in Super-K style
plt.figure(figsize=(8, 5))
plt.plot(df['T_e'], df['rate'], label='Expected Rate', color='blue')
plt.yscale('log')
plt.xlim(6, 17)
plt.ylim(1e-4, 1e2)
plt.xlabel('Recoil electron kinetic energy [MeV]')
plt.ylabel('Event/day/22.5kton/0.5MeV')
plt.title('Expected Solar Neutrino Event Rate')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
