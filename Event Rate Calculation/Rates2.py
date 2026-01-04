import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.linalg import expm
from numba import njit

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (1.973*1e-5)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1
m_e = 0.511  # electron mass in MeV

# SK settings
fiducial_mass = 22.5e9 # g
N_e_tgt = 10*fiducial_mass*N_A/18  # number of target electrons
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
lambda_val_B = np.array(lambda_df['lambda'].values, dtype=float)/1000

hep_df = pd.read_csv('hep.csv')
hep_E = np.array(hep_df['energy'].values, dtype=float)
lambda_val_hep = np.array(hep_df['lambda'].values, dtype=float)

# Get unique energy and Te values
E_nu_vals = np.unique(E_nu_vals_grid)
Te_bin_centers = np.unique(Te_vals_grid)

print(f"Loaded cross sections for {len(E_nu_vals)} energies and {len(Te_bin_centers)} Te bins")

# Create combined spectrum
lambda_interp_B_on_grid = np.interp(E_nu_vals, lambda_E, lambda_val_B, left=0.0, right=0.0)
lambda_interp_hep_on_grid = np.interp(E_nu_vals, hep_E, lambda_val_hep, left=0.0, right=0.0)

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

# Vacuum Parameters
del_m2 = 7.5e-5  # eV2
theta = np.arcsin(np.sqrt(0.307))

# Load solar electron density profile from CSV
solar_data = pd.read_csv('solar.csv')
r_over_rsol = solar_data['R/R_s'].values.astype(float)
log10_n_over_NA = solar_data['log10(n/N_A)'].values.astype(float)

# Convert to actual radius in eV^-1 and number density in eV^3
r_data = r_over_rsol * R_sol # type: ignore
n_e_data = (10**log10_n_over_NA) * N_A * one_by_cm3_to_eV3 # type: ignore

def avg_Pee():
    return 1.0

# Oscillation parameters
beta = 0.0 # Turbulence Parameter
tau = 10*eV_to_1_by_m*1000

# Load Te values from plot-data.csv
plot_data = pd.read_csv("plot-data.csv")
Te_values = plot_data['energy'].values
bin_widths = plot_data['bin_width'].values

# Pre-compute oscillation probabilities for all energies
print("Pre-computing oscillation probabilities...")
Pee_values = np.zeros(len(E_nu_vals))
for i, E_nu in enumerate(tqdm(E_nu_vals, desc="Computing Pee")):
    Pee_values[i] = avg_Pee()

print("Computing rates for each Te bin...")

# Calculate rate for each Te bin
results = []

# For each T_e center, perform Riemann sum over all E_nu
for Te_center, bin_width in zip(tqdm(Te_bin_centers, desc="Computing Te rates"), bin_widths):

    E_min = 4.0  # MeV, approximate threshold for SK

    # Filter for E_nu >= E_min
    valid_mask = E_nu_vals >= E_min
    E_nu_valid = E_nu_vals[valid_mask]
    
    # Build arrays for integration
    integrand = np.zeros(len(E_nu_valid))
    
    for i, E_nu in enumerate(E_nu_valid):
        # Find cross sections for this (E_nu, T_e) pair
        mask = (E_nu_vals_grid == E_nu) & (Te_vals_grid == Te_center)
        if np.any(mask):
            # Extract cross sections
            sigma_e = sigma_e_grid[mask][0]
            sigma_x = sigma_x_grid[mask][0]
            
            # Get oscillation probability
            i_E = np.where(E_nu_vals == E_nu)[0][0]
            Pee_val = Pee_values[i_E]
            
            # Get total flux * lambda at this energy
            flux_at_E = flux_spectrum[i_E]
            
            # Effective cross section: sigma_e*P_ee + sigma_x*(1-P_ee)
            sigma_eff = sigma_e * Pee_val + sigma_x * (1 - Pee_val)
            
            # Integrand: flux * lambda * sigma_eff
            integrand[i] = flux_at_E * sigma_eff
    
    # Perform trapezoidal integration
    integral = np.trapz(integrand, E_nu_valid)
    
    # Multiply by target electrons to get rate
    total_rate_for_Te = integral * N_e_tgt
    total_rate_for_Te *= 24 * 3600*2970
    results.append([Te_center, total_rate_for_Te])

# Convert to array
rates_vs_Te = np.array(results)

# Save results
np.savetxt(f'theoretical_rate_vs_Te_SK_MC.csv', rates_vs_Te, 
           delimiter=',', header='Te_MeV,EventRate_per_bin', comments='')

# Load experimental data for comparison if available
exp_data = np.loadtxt('plot-data.csv', delimiter=',', skiprows=1)
exp_te = exp_data[:, 0]
exp_rate = exp_data[:, 1]*22.5*2970/365  # Scale to total events in SK
exp_error = exp_data[:,2]/2
yerr = (exp_data[:,3] + exp_data[:,4])/2
yerr = yerr * 22.5*2970/365
# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(rates_vs_Te[:, 0], rates_vs_Te[:, 1], label='Theoretical Rate', color='blue', s=30)    
plt.scatter(exp_te, exp_rate, label='SK Experimental Data', color='red', s=30)
plt.errorbar(exp_te, exp_rate,
        xerr=exp_error, yerr=yerr,
        fmt='o', color='red', ecolor='red', capsize=3,markersize=6, alpha=0.7
    )
plt.xlabel('Electron Recoil Energy Te (MeV)')
plt.ylabel('Events/bin')
plt.title('Event Rate vs Electron Recoil Energy')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.yscale('log')
plt.savefig(f"event_rate_vs_Te_final.png", dpi=300)
plt.show()