import numpy as np
import matplotlib.pyplot as plt

# Data setup
sk_energy_bins = np.array([
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0, 16.0
])

R_exp = np.array([
    72.1, 64.8, 60.2, 54.2, 49.2, 44.8, 35.7, 26.6, 25.4, 20.7,
    15.7, 10.9, 9.65, 7.14, 5.05, 3.96, 2.56, 1.95, 1.60, 0.750
])

sigma_exp_stat = np.array([
    (9.5+9.4)/2, (4.7+4.6)/2, (3.4+3.3)/2, (2.4+2.4)/2, (2.2+2.1)/2,
    (2.0+1.9)/2, (1.2+1.2)/2, (1.0+1.0)/2, (0.9+0.9)/2, (0.8+0.8)/2,
    (0.7+0.7)/2, (0.6+0.5)/2, (0.51+0.49)/2, (0.42+0.40)/2, (0.36+0.33)/2,
    (0.30+0.28)/2, (0.25+0.23)/2, (0.22+0.20)/2, (0.21+0.19)/2, (0.157+0.133)/2
])

bin_widths = sk_energy_bins[1:] - sk_energy_bins[:-1]
bin_centers = (sk_energy_bins[:-1] + sk_energy_bins[1:]) / 2

# Conversion
R_evt = R_exp * 21.5 / 365
sigma_evt = sigma_exp_stat * 21.5 / 365

np.savetxt("Finall/experimental_rate_SK.csv", np.column_stack((bin_centers, R_evt, sigma_evt)), 
           header="Energy_bin_center(MeV),Event_rate(day^-1 21.5kt^-1 0.5MeV^-1),Statistical_error(day^-1 21.5kt^-1 0.5MeV^-1)", 
           delimiter=",")
plt.figure(figsize=(8,5))
plt.errorbar(bin_centers, R_evt, yerr=sigma_evt, fmt='o', label='SK data', alpha=0.8)
plt.xlabel('Energy (MeV)')
plt.ylabel('Events/day/21.5kt/0.5MeV')
plt.title('Super-Kamiokande Experimental Spectrum')
plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Finall/experimental_spectrum.png", dpi=300)
plt.show()
