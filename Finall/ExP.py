import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV file
data = pd.read_csv("Finall/plot-data.csv")
energy = data['Recoil energy(MeV)'].values
R_evt = data['rate'].values.astype(float)  # Note the space before 'rate'
sigma_evt = np.full(R_evt.shape, 0.25) 

np.savetxt("Finall/experimental_rate_SK.csv", np.column_stack((energy, R_evt, sigma_evt)),  # type: ignore
           header="Energy_bin_center(MeV),Event_rate(day^-1 22.5kt^-1 0.5MeV^-1),Statistical_error(day^-1 22.5kt^-1 0.5MeV^-1)", 
           delimiter=",")
plt.figure(figsize=(8,5))
plt.errorbar(energy, R_evt, xerr=sigma_evt, fmt='o', label='CSV data', alpha=0.8) # type: ignore
plt.xlabel('Energy (MeV)')
plt.ylabel('Events/day/22.5kt/0.5MeV')
plt.title('Experimental Spectrum from CSV Data')
plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Finall/experimental_spectrum.png", dpi=300)
plt.show()
