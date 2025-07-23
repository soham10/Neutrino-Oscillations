import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('prob.csv')
energy = data['E'].values
probs = data['Day'].values

# Plot
plt.figure(figsize=(10, 6), dpi=120)
plt.plot(energy, probs, label='$P_{ee}$', color='darkblue', linewidth=2)
plt.scatter(energy, probs, s=10, color='blue', alpha=0.5)

# Styling
plt.xlabel('Neutrino Energy (MeV)', fontsize=12)
plt.ylabel(r'Survival Probability $P_{ee}$', fontsize=12)
plt.title('Daytime Neutrino Survival Probability vs Energy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

