import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('theoretical_rate_vs_Te_SK.csv')
data_MC = pd.read_csv('theoretical_rate_vs_Te_SK_MC.csv')
rate_exp = pd.read_csv('MC.csv')
rate_MSW = data['EventRate_per_day_per_22.5kt_per_bin']
rate_MC = data_MC['EventRate_per_day_per_22.5kt_per_bin']
ratio = rate_MSW / rate_MC
plt.figure(figsize=(10, 6))
plt.scatter(data['Te_MeV'], ratio, label='Data/MC', color='green')
plt.scatter(rate_exp['Recoil energy(MeV)'], rate_exp['rate'], label='SK Data', color='red')
plt.xlabel('Electron Recoil Energy Te (MeV)')
plt.ylabel('Ratio of Event Rates')
plt.title('Ratio of Theoretical Rates with and without MSW Effect')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.ylim(0,0.8)
plt.show()