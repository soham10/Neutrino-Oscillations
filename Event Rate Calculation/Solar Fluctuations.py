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

# Vacuum Parameters
del_m2 = 7.5e-5  # eV2
theta = np.arcsin(np.sqrt(0.32))

# Load solar electron density profile from CSV
solar_data = pd.read_csv('solar.csv')
r_over_rsol = solar_data['R/R_s'].values.astype(float)
log10_n_over_NA = solar_data['log10(n/N_A)'].values.astype(float)

# Convert to actual radius in eV^-1 and number density in eV^3
r_data = r_over_rsol * R_sol # type: ignore
n_e_data = (10**log10_n_over_NA) * N_A * one_by_cm3_to_eV3 # type: ignore

# Couplings parameters for computing oscillations and matter potential
@njit
def k(N, beta, tau):
    return tau*(G_F*beta*N)** 2

@njit
def A(E, N, del_m2, theta):
    return -del_m2*np.cos(2*theta)/(4*E) + G_F*N/np.sqrt(2)

@njit
def B(E, del_m2, theta):
    return del_m2*np.sin(2*theta)/(4*E)

def solar_solver(E, beta, tau, del_m2, theta, r_i=0.0, r_f=1.0):
    """
    Propagate neutrino through the Sun using density profile from CSV.
    Uses actual data points from solar.csv instead of interpolation.
    """
    E = E * 1e6  # MeV to eV
    
    # Find indices for the radial range
    r_min = r_i * R_sol
    r_max = r_f * R_sol
    
    # Select data points within the range
    mask = (r_data >= r_min) & (r_data <= r_max)
    r_vals = r_data[mask]
    N_vals = n_e_data[mask]
    
    n_slabs = len(r_vals) - 1
    psi = np.array([1.0, 0.0, 0.0])
    
    for i in range(n_slabs):
        dx = r_vals[i+1] - r_vals[i]
        N = N_vals[i]
        
        k_r = k(N, beta, tau)
        A_r = A(E, N, del_m2, theta)
        B_E = B(E, del_m2, theta)
        M = np.array([[0.0, 0.0, B_E],
                      [0.0, k_r, -A_r],
                      [-B_E, A_r, k_r]])
        U = expm(-2 * M * dx)
        psi = U @ psi
    return psi

def vacuum_propagation(E, psi_initial, del_m2, theta, n_slabs=100000):
    """Propagate neutrino from solar surface (R_sol) to Earth (R_earth) in vacuum"""
    E = E * 1e6  # MeV to eV
    dx = (R_earth - R_sol) / n_slabs
    psi = psi_initial
    Pee_profile = np.zeros(n_slabs)
    
    # In vacuum, N_e = 0
    B_E = B(E, del_m2, theta)
    A_vac = A(E, 0.0, del_m2, theta)  # N=0 for vacuum
    
    M_vac = np.array([[0.0, 0.0, B_E],
                      [0.0, 0.0, -A_vac],
                      [-B_E, A_vac, 0.0]])
    U_vac = expm(-2 * M_vac * dx)
    
    for i in range(n_slabs):
        psi = U_vac @ psi
        Pee_profile[i] = (psi[0] + 1)*0.5
    return Pee_profile

def avg_Pee(E, beta, tau, del_m2, theta):
    # Solar propagation
    psi = solar_solver(E, beta, tau, del_m2, theta)
    
    # Vacuum propagation from R_sol to R_earth
    Pee_profile_vac = vacuum_propagation(E, psi, del_m2, theta)
    return np.mean(Pee_profile_vac)

# Energy range for plotting (MeV)
E_range = np.linspace(0.1, 20, 100)  # From 0.1 to 20 MeV

# Beta values to compare
beta_values = [0.0, 0.02, 0.05, 0.1]
tau = 10 * eV_to_1_by_m * 1000

# Colors for different beta values
colors = ['blue', 'green', 'orange', 'red']

# Compute Pee for each beta value
plt.figure(figsize=(12, 7))

for idx, beta in enumerate(beta_values):
    print(f"Computing Pee for beta = {beta}...")
    Pee_vals = np.zeros(len(E_range))
    
    for i, E in enumerate(tqdm(E_range, desc=f"β={beta}")):
        Pee_vals[i] = avg_Pee(E, beta, tau, del_m2, theta)
    
    # Plot
    plt.plot(E_range, Pee_vals, label=f'β = {beta}', 
             color=colors[idx], linewidth=2)
    
    # Save data
    df = pd.DataFrame({'Energy_MeV': E_range, 'Pee': Pee_vals})
    df.to_csv(f'Pee_vs_E_beta_{beta}.csv', index=False)
    print(f"Saved Pee_vs_E_beta_{beta}.csv")

plt.xlabel('Neutrino Energy (MeV)', fontsize=12)
plt.ylabel('Survival Probability $P_{ee}$', fontsize=12)
plt.title('Electron Neutrino Survival Probability vs Energy', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Pee_vs_Energy_different_betas.png', dpi=300)
plt.show()

print("\nPlot saved as 'Pee_vs_Energy_different_betas.png'")