import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

E_nu_vals = np.linspace(0.1, 20, 100)
hep_df = pd.read_csv('lambda.csv')
hep_E = np.array(hep_df['energy'].values, dtype=float)
lambda_val_hep = np.array(hep_df['lambda'].values, dtype=float)

lambda_interp_hep_on_grid = np.interp(E_nu_vals, hep_E, lambda_val_hep, left=0.0, right=1e-6)
# Plot
plt.figure(figsize=(15, 8))
plt.plot(E_nu_vals, lambda_interp_hep_on_grid)
plt.xlabel("Energy (MeV)", fontsize=14)
plt.ylabel(r"$\lambda$", fontsize=14)
plt.title(r"Energy Spectrum", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
