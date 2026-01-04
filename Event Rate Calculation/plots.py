import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rate_df = pd.read_csv('Finall/Theory Probability[0.0].csv')
E = np.array(rate_df['energy'].values, dtype=float)
rate = np.array(rate_df['results'].values, dtype=float)

# Plot
plt.figure(figsize=(15, 8))
plt.scatter(E, rate)
plt.xlabel("Recoil Energy (MeV)", fontsize=14)
plt.ylabel(r"Event Rate (per day per 22.5kt per bin)", fontsize=14)
plt.title(r"Energy Spectrum", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.xscale('log')
plt.show()
