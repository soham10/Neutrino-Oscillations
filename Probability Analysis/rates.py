import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.constants import physical_constants as pc
from scipy.ndimage import gaussian_filter1d
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (1.973*1e-5)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1

# Vacuum Parameters
del_m2_v = 7.5e-5  # eV2
theta_v = np.arcsin(np.sqrt(0.32))                                 

# Load solar electron density profile from CSV
solar_data = pd.read_csv('solar.csv')
r_over_rsol = solar_data['R/R_s'].values.astype(float)
log10_n_over_NA = solar_data['log10(n/N_A)'].values.astype(float)

# Convert to actual radius in eV^-1 and number density in eV^3
r_data = r_over_rsol * R_sol
n_e_data = (10**log10_n_over_NA) * N_A * one_by_cm3_to_eV3

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

def avg_Pee(beta, E_vals, tau, del_m2, theta, n_jobs=-1):
    def probability(E):
        # Solar propagation
        psi = solar_solver(E, beta, tau, del_m2, theta)
        Pee_profile_vac = vacuum_propagation(E, psi, del_m2, theta)
        return np.mean(Pee_profile_vac)
    
    avg_Pee = Parallel(n_jobs=n_jobs)(delayed(probability)(E) for E in tqdm(E_vals, desc=f'β = {beta}'))
    return np.array(avg_Pee)

# Implementation
E_vals = np.linspace(4, 16, 400)  # MeV
beta_values = [0.0,0.01,0.03,0.05,0.1]  # Multiple beta values to plot
tau = 10*eV_to_1_by_m*1000

exp_Pee = pd.read_csv('SK_rate.csv')
exp_eng = exp_Pee['energy']
exp_rate = exp_Pee['Pee_rate']
exp_rate_up = exp_Pee['rate_up']
exp_rate_down = exp_Pee['rate_down']
exp_left = exp_Pee['energy_left']
exp_right = exp_Pee['energy_right']

# Vertical error bars (rate)
yerr_lower = exp_rate - exp_rate_down
yerr_upper = exp_rate_up - exp_rate
yerr = [yerr_lower, yerr_upper]
# Horizontal error bars (energy)
xerr_left = exp_eng - exp_left
xerr_right = exp_right - exp_eng
xerr = [xerr_left, xerr_right]

plt.figure(figsize=(10, 6))

# Plot theory curves for multiple beta values
for beta in beta_values:
    Pee = avg_Pee(beta, E_vals, tau, del_m2_v, theta_v, n_jobs=-1)
    deg = 4
    coeffs = np.polyfit(E_vals, Pee, deg)
    Pee_smooth = np.polyval(coeffs, E_vals)
    # Calculate rate using smoothed Pee
    rate = 0.16 + 0.84*Pee_smooth
    plt.plot(E_vals, rate, label=f'β={beta}', linewidth=2)
    
    # Save theory rates for beta=0 only
    if beta == 0.0:
        np.savetxt('theory_rates.csv', np.column_stack((E_vals, Pee_smooth)), 
                   delimiter=',', header='energy,Pee_rate', comments='')

# Plot experimental data with error bars
plt.errorbar(
    exp_eng, exp_rate,
    xerr=xerr, yerr=yerr,
    fmt='o', color='black', ecolor='red', capsize=3, label='Experiment', markersize=6
)

plt.xlabel('Neutrino Energy E (MeV)')
plt.ylabel('Data/MC Rate')
plt.title('Solar Neutrino Data/MC Rate vs Energy')
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(0.30, 0.70)
plt.tight_layout()
plt.savefig('rates-SKIV.png', dpi=300)
plt.show()