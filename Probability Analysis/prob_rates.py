import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
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
theta_v = np.arcsin(np.sqrt(0.307))                                 

# Load solar electron density profile from CSV
solar_data = pd.read_csv('solar.csv')
r_over_rsol = solar_data['R/R_s'].values.astype(float)
log10_n_over_NA = solar_data['log10(n/N_A)'].values.astype(float)

# Convert to actual radius in eV^-1 and number density in eV^3
r_data = r_over_rsol * R_sol # type: ignore
n_e_data = (10**log10_n_over_NA) * N_A * one_by_cm3_to_eV3 # type: ignore

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
E_vals = np.linspace(3, 20, 500)  # MeV
beta_values = [0.0,0.03,0.05,0.1,1]  # Multiple beta values to plot
tau = 10*eV_to_1_by_m*1000

# Define colors for different beta values
beta_colors = ['black', 'orange', 'cyan', 'magenta', 'brown']

# Load experimental data from all SK phases
sk1_data = pd.read_csv('SK I.csv')
sk2_data = pd.read_csv('SK II.csv')
sk3_data = pd.read_csv('SK III.csv')
sk4_data = pd.read_csv('SK IV.csv')

sk_datasets = [
    ('SK I', sk1_data, 'o', 'red'),
    ('SK II', sk2_data, 's', 'blue'),
    ('SK III', sk3_data, '^', 'green'),
    ('SK IV', sk4_data, 'D', 'purple')
]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = [ax1, ax2, ax3, ax4]

# Plot theory curves and experimental data for each SK phase
for beta, beta_color in zip(beta_values, beta_colors):
    Pee = avg_Pee(beta, E_vals, tau, del_m2_v, theta_v, n_jobs=-1)
    deg = 4
    coeffs = np.polyfit(E_vals, Pee, deg)
    Pee_smooth = np.polyval(coeffs, E_vals)
    # Calculate rate using smoothed Pee
    rate = 0.16 + 0.84*Pee_smooth
    
    # Label beta=0 as "Standard MSW"
    label = 'Standard MSW' if beta == 0.0 else f'β={beta}'
    
    # Plot theory on all subplots
    for ax in axes:
        ax.plot(E_vals, rate, label=label, linewidth=2, color=beta_color)
    
    # Save theory rates for beta=0 only
    if beta == 0.0:
        np.savetxt('theory_rates.csv', np.column_stack((E_vals, Pee_smooth)), 
                   delimiter=',', header='energy,Pee_rate', comments='')

# Plot each SK dataset on its corresponding subplot
for (label, data, marker, color), ax in zip(sk_datasets, axes):
    exp_eng = data['energy']
    exp_rate = data['Pee_rate']
    exp_rate_up = data['error_up']
    exp_rate_down = data['error_down']
    exp_left = data['energy_left']
    exp_right = data['energy_right']
    
    # Vertical error bars (rate)
    yerr_lower = exp_rate_down
    yerr_upper = exp_rate_up
    yerr = [yerr_lower, yerr_upper]
    # Horizontal error bars (energy)
    xerr_left = exp_eng - exp_left
    xerr_right = exp_right - exp_eng
    xerr = [xerr_left, xerr_right]
    
    ax.errorbar(
        exp_eng, exp_rate,
        xerr=xerr, yerr=yerr,
        fmt=marker, color=color, ecolor=color, capsize=3, 
        label=label, markersize=6, alpha=0.7
    )
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(0.30, 0.70)

# Set labels
ax1.set_ylabel('Data/MC Rate')
ax3.set_ylabel('Data/MC Rate')
ax3.set_xlabel('Neutrino Energy E (MeV)')
ax4.set_xlabel('Neutrino Energy E (MeV)')

plt.tight_layout()
plt.savefig('Data_per_MC_plot_for_different_phases.png', dpi=300)
plt.show()