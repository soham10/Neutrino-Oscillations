import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv('Theory Probability[0.0].csv')
data2 = pd.read_csv('Theory Probability[0.03].csv')
data3 = pd.read_csv('Theory Probability[0.05].csv')
data4 = pd.read_csv('Theory Probability[0.1].csv')

# Plot
plt.figure(figsize=(15, 8))
plt.plot(data1['energy'], data1['results'], label='Beta = 0.0')
plt.plot(data2['energy'], data2['results'], label='Beta = 0.03')
plt.plot(data3['energy'], data3['results'], label='Beta = 0.05')
plt.plot(data4['energy'], data4['results'], label='Beta = 0.1')
plt.xlabel("Energy (MeV)", fontsize=14)
plt.ylabel(r"$P_{ee}$", fontsize=14)
plt.title(r"Electron Neutrino Survival for Varying $\beta$ with Stochastic Fluctuations", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.show()
