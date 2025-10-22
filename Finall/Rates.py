import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import expm
from numba import njit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 806554.39
one_by_cm3_to_eV3 = (1.23981e-4)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1

# SK settings
fiducial_mass = 1e9 # g
N_e_tgt = 10*fiducial_mass*N_A/18  # number of target electrons
phi_B0 = 2.32e6  # cm^-2 s^-1, https://arxiv.org/pdf/hep-ex/0508053
phi_hep = 7.88e3  # cm^-2 s^-1

# Load detector-corrected cross sections σ(E_ν, Te)
sigmas = np.loadtxt('Finall/sigma_vs_Enu_Te_bins_SK.csv', delimiter=',', skiprows=1)
E_nu_vals_grid = sigmas[:, 0]   # Neutrino energy (MeV)
Te_vals_grid = sigmas[:, 1]      # Te bin centers (MeV)
sigma_e_grid = sigmas[:, 2]      # ν_e cross section (10^-46 cm^2)
sigma_x_grid = sigmas[:, 3]      # ν_x cross section (10^-46 cm^2)

# Convert back to cm^2
sigma_e_grid *= 1e-46
sigma_x_grid *= 1e-46

# Load lambda values
lambda_df = pd.read_csv('Finall/lambda.csv')
lambda_E = np.array(lambda_df['energy'].values, dtype=float)
lambda_val_B = np.array(lambda_df['lambda'].values, dtype=float)

hep_df = pd.read_csv('Finall/hep.csv')
hep_E = np.array(hep_df['energy'].values, dtype=float)
lambda_val_hep = np.array(hep_df['lambda'].values, dtype=float)

# Get unique energy and Te values
E_nu_vals = np.unique(E_nu_vals_grid)
Te_bin_centers = np.unique(Te_vals_grid)

print(f"Loaded cross sections for {len(E_nu_vals)} energies and {len(Te_bin_centers)} Te bins")

# Create combined spectrum
lambda_interp_hep_on_grid = np.interp(E_nu_vals, hep_E, lambda_val_hep, left=0.0, right=0.0)
lambda_combined = (phi_B0 * lambda_val_B + phi_hep * lambda_interp_hep_on_grid) / (phi_B0 + phi_hep)

total_flux = phi_B0 + phi_hep

# Save combined spectrum for reference
comb_df = pd.DataFrame({
    'E': E_nu_vals,
    'lambda_eff': lambda_combined,
    'lambda_B': lambda_val_B,
    'lambda_hep': lambda_interp_hep_on_grid,
})
comb_df.to_csv('Finall/lambda_combined.csv', index=False)
nz = comb_df[comb_df['lambda_hep'] > 0]

# Couplings parameters for computing oscillations and matter potential
@njit
def N_e(r):
    if r <= R_sol:
        return 245 * N_A * np.exp(-r * 10.45 / R_sol) * one_by_cm3_to_eV3
    else:
        return 0.0

@njit
def A_cc(n_e, E):
    return 2 * np.sqrt(2) * G_F * n_e * E

@njit
def del_m2_eff(del_m2, theta, A_cc_val):
    return np.sqrt((del_m2 * np.cos(2 * theta) - A_cc_val) ** 2 + (del_m2 * np.sin(2 * theta)) ** 2)

@njit
def theta_eff(del_m2, theta, A_cc_val):
    num = del_m2 * np.sin(2 * theta)
    den = del_m2 * np.cos(2 * theta) - A_cc_val
    return 0.5 * np.arctan2(num, den)

@njit
def k(N, beta, tau):
    return tau * (G_F * beta * N) ** 2

@njit
def A(E, N, del_m2_m, theta_m):
    return -del_m2_m * np.cos(2 * theta_m) / (4 * E) + G_F * N / np.sqrt(2)

@njit
def B(E, del_m2_m, theta_m):
    return del_m2_m * np.sin(2 * theta_m) / (4 * E)

def solar_solver(E, beta, tau, del_m2, theta, n_slabs=10000, r_i=0.0, r_f=1.0):
    E = E * 1e6  # MeV to eV
    dx = (r_f - r_i) * R_earth / n_slabs
    r_vals = np.linspace(r_i, r_f, n_slabs)
    psi = np.array([1.0, 0.0, 0.0])
    Pee_profile = np.empty(n_slabs)

    for i in range(n_slabs):
        r = r_vals[i] * R_earth
        N = N_e(r)
        A_m_val = A_cc(N, E)
        del_m2_m = del_m2_eff(del_m2, theta, A_m_val)
        theta_m = theta_eff(del_m2, theta, A_m_val)
        k_r = k(N, beta, tau)
        A_r = A(E, N, del_m2_m, theta_m)
        B_E = B(E, del_m2_m, theta_m)
        M = np.array([[0.0, 0.0, B_E],
                      [0.0, k_r, -A_r],
                      [-B_E, A_r, k_r]])
        U = expm(-2 * M * dx)
        psi = U @ psi
        Pee_profile[i] = (psi[0] + 1) * 0.5
    return r_vals, Pee_profile

def avg_Pee(E, beta, tau, del_m2, theta):
    r_frac, Pee_profile = solar_solver(E, beta, tau, del_m2, theta)
    mask = (r_frac >= 0.9)  # Reduced from 0.95 to 0.9 for fewer points
    return np.mean(Pee_profile[mask])

# Oscillation parameters
beta = 0.05 # Turbulence Parameter
tau = 10*eV_to_1_by_m*1000
del_m2 = 7.1e-5  # eV2
theta = np.arctan(np.sqrt(0.46))    # radians

# Load Te values from plot-data.csv
plot_data = pd.read_csv("Finall/plot-data.csv")
Te_values = plot_data['Recoil energy(MeV)'].values

# Pre-compute oscillation probabilities for all energies
print("Pre-computing oscillation probabilities...")
Pee_values = np.zeros(len(E_nu_vals))
for i, E_nu in enumerate(tqdm(E_nu_vals, desc="Computing Pee")):
    Pee_values[i] = avg_Pee(E_nu, beta, tau, del_m2, theta)

print("Computing rates for each Te bin...")

# Calculate rate for each Te bin
results = []

for Te_center in tqdm(Te_bin_centers, desc="Computing Te rates"):
    total_rate_for_Te = 0.0
    
    # Sum over all neutrino energies
    for i_E, E_nu in enumerate(E_nu_vals):
        # Find the cross section for this (E_nu, Te_center) pair
        mask = (E_nu_vals_grid == E_nu) & (Te_vals_grid == Te_center)
        
        if np.any(mask):
            sigma_e = sigma_e_grid[mask][0]
            sigma_x = sigma_x_grid[mask][0]
            
            # Get oscillation probability and spectrum weight
            Pee_val = Pee_values[i_E]
            lambda_val = lambda_combined[i_E]
            rate_contrib = total_flux * lambda_val * (sigma_e * Pee_val + sigma_x * (1 - Pee_val))
            rate_contrib *= N_e_tgt * 24 * 3600  # Events/day/22.5kt
            
            total_rate_for_Te += rate_contrib
    
    results.append([Te_center, total_rate_for_Te])

# Convert to array
rates_vs_Te = np.array(results)

# Save results
np.savetxt(f'Finall/theoretical_rate_vs_Te_SK ({beta}).csv', rates_vs_Te, 
           delimiter=',', header='Te_MeV,EventRate_per_day_per_21.5kt_per_0.5MeV', comments='')

# Load experimental data for comparison if available
try:
    exp_data = np.loadtxt('Finall/plot-data.csv', delimiter=',', skiprows=1)
    exp_te = exp_data[:, 0]
    exp_rate = exp_data[:, 1]
    exp_err = exp_data[:, 2] if exp_data.shape[1] > 2 else None
    has_exp_data = True
except:
    print("No experimental data found for comparison")
    has_exp_data = False

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(rates_vs_Te[:, 0], rates_vs_Te[:, 1], label='Theoretical Rate', color='blue', s=30)    
if has_exp_data:
    if exp_err is not None:
        plt.errorbar(exp_te, exp_rate, xerr=exp_err, fmt='o', label='SK Experimental Data', 
                    color='red', alpha=0.7, markersize=5)
    else:
        plt.plot(exp_te, exp_rate, 'o', label='SK Experimental Data', color='red', markersize=5)

plt.xlabel('Electron Recoil Energy Te (MeV)')
plt.ylabel('Events/day/21.5kt/0.5MeV')
plt.title('Event Rate vs Electron Recoil Energy')

# Add textual information about parameters
plt.text(0.02, 0.98, 
         f"Oscillation parameters: β={beta}, τ={tau/eV_to_1_by_m/1000:.1f} km, Δm²={del_m2:.1e} eV², tan²θ={np.tan(theta)**2:.2f}", 
         transform=plt.gca().transAxes, 
         verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.7))

plt.yscale('log')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"Finall/event_rate_vs_Te_final ({beta}).png", dpi=300)
plt.show()