import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Constants
G_F = 1.1663787e-23  # eV^2
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (1.973*1e-5)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1

# SK settings -
fiducial_mass = 22.5e9 # g (22.5 kton)
N_e_tgt = 7.521e33  # Target electron count as specified
phi_B0 = 5.25e6  # cm^-2 s^-1
phi_hep = 7.88e3  # cm^-2 s^-1

sigmas = np.loadtxt('sigma.csv', delimiter=',', skiprows=1)
E_nu_vals_grid = sigmas[:, 0]   # Neutrino energy (MeV)
Te_vals_grid = sigmas[:, 1]      # Te bin centers (MeV)
sigma_e_grid = sigmas[:, 2]      # ν_e cross section
sigma_x_grid = sigmas[:, 3]      # ν_x cross section 


# Load lambda values
lambda_df = pd.read_csv('lambda.csv')
lambda_E = np.array(lambda_df['energy'].values, dtype=float)
lambda_val_B = np.array(lambda_df['lambda'].values, dtype=float)

hep_df = pd.read_csv('hep.csv')
hep_E = np.array(hep_df['energy'].values, dtype=float)
lambda_val_hep = np.array(hep_df['lambda'].values, dtype=float)

# Get unique energy and Te values
E_nu_vals = np.unique(E_nu_vals_grid)
Te_bin_centers = np.unique(Te_vals_grid)

print(f"Loaded cross sections for {len(E_nu_vals)} energies and {len(Te_bin_centers)} Te bins")

# Create combined spectrum
lambda_interp_B_on_grid = np.interp(E_nu_vals, lambda_E, lambda_val_B, left=0.0, right=1e-7)
lambda_interp_hep_on_grid = np.interp(E_nu_vals, hep_E, lambda_val_hep, left=0.0, right=1e-4)

# Flux-weighted spectrum (not normalized)
flux_spectrum = phi_B0 * lambda_interp_B_on_grid + phi_hep * lambda_interp_hep_on_grid

# Save combined spectrum for reference
comb_df = pd.DataFrame({
    'E': E_nu_vals,
    'flux_spectrum': flux_spectrum,
    'lambda_B': lambda_interp_B_on_grid,
    'lambda_hep': lambda_interp_hep_on_grid,
})
comb_df.to_csv('combined_spectrum.csv', index=False)
def avg_Pee():
    return 1.0

# Load Te values from plot-data.csv
plot_data = pd.read_csv("plot-data.csv")
Te_values = plot_data['energy'].values

# Pre-compute oscillation probabilities for all energies
print("Pre-computing oscillation probabilities...")
Pee_values = np.zeros(len(E_nu_vals))
for i, E_nu in enumerate(tqdm(E_nu_vals, desc="Computing Pee")):
    Pee_values[i] = avg_Pee()

print("Computing rates for each Te bin...")

# Calculate energy bin width for Riemann sum integration
dE_nu = 0.02  # MeV

# Calculate rate for each Te bin
results = []

# For each T_e center, perform Riemann sum over all E_nu
for Te_center in tqdm(Te_bin_centers, desc="Computing Te rates"):
    # Initialize sum for this T_e bin
    riemann_sum = 0.0
    
    # Sum over all E_nu associated with this T_e center
    for i_E, E_nu in enumerate(E_nu_vals):
        # Find cross sections for this (E_nu, T_e) pair
        mask = (E_nu_vals_grid == E_nu) & (Te_vals_grid == Te_center)
        
        if np.any(mask):
            # Extract cross sections
            sigma_e = sigma_e_grid[mask][0]
            sigma_x = sigma_x_grid[mask][0]
            
            # Get oscillation probability
            Pee_val = Pee_values[i_E]
            
            # Get total flux * lambda at this energy
            flux_at_E = flux_spectrum[i_E]
            
            # Effective cross section: sigma_e*P_ee + sigma_x*(1-P_ee)
            sigma_eff = sigma_e * Pee_val + sigma_x * (1 - Pee_val)
            
            # Riemann sum element: flux * lambda * sigma_eff * dE
            riemann_sum += flux_at_E * sigma_eff * dE_nu
    
    # Multiply by target electrons to get rate
    total_rate_for_Te = riemann_sum * N_e_tgt
    
    # Convert s^-1 -> day^-1
    total_rate_for_Te *= 24 * 3600
    
    results.append([Te_center, total_rate_for_Te])

# Convert to array
rates_vs_Te = np.array(results)

# Save results
np.savetxt(f'theoretical_rate_vs_Te_SK_MC.csv', rates_vs_Te, 
           delimiter=',', header='Te_MeV,EventRate_per_day_per_22.5kt_per_bin', comments='')

# Load experimental data for comparison if available
exp_data = np.loadtxt('plot-data.csv', delimiter=',', skiprows=1)
exp_te = exp_data[:, 0]
exp_rate = exp_data[:, 1]

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(rates_vs_Te[:, 0], rates_vs_Te[:, 1], label='Theoretical Rate', color='blue', s=30)    
plt.scatter(exp_te, exp_rate, label='SK Experimental Data', color='red', s=30)

plt.xlabel('Electron Recoil Energy Te (MeV)')
plt.ylabel('Events/day/22.5kt/0.5MeV')
plt.title('Event Rate vs Electron Recoil Energy')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.yscale('log')
plt.savefig(f"event_rate_vs_Te_final.png", dpi=300)
plt.show()