import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from scipy.optimize import minimize

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (1.973*1e-5)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1                                

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
        
        # Vacuum propagation from R_sol to R_earth
        Pee_profile_vac = vacuum_propagation(E, psi, del_m2, theta)
        return np.mean(Pee_profile_vac)
    
    avg_Pee = Parallel(n_jobs=n_jobs)(delayed(probability)(E) for E in E_vals)
    return np.array(avg_Pee)


def chi_sq(exp, th, sigma):
    """
    Calculate chi-squared: χ² = Σ[(O_i - E_i)² / σ_i²]
    where O_i = observed (experimental), E_i = expected (theoretical), σ_i = uncertainty
    """
    return np.sum(((exp - th) / sigma) ** 2)

def calculate_rate(beta, tau, exp_eng, del_m2, theta, n_jobs=-1):
    """Calculate theoretical rate for given parameters"""
    Pee = avg_Pee(beta, exp_eng, tau, del_m2, theta, n_jobs=n_jobs)
    return 0.16 + 0.84*Pee

def chi_square_scan(beta, tau, exp_eng, exp_rate, sigma, 
                    del_m2_range, sin2_theta_range, n_jobs=-1):
    """
    Perform chi-squared scan over del_m2 and sin^2(theta) parameter space
    """
    chi2_grid = np.zeros((len(del_m2_range), len(sin2_theta_range)))
    total_pairs = len(del_m2_range) * len(sin2_theta_range) 
    with tqdm(total=total_pairs, desc='Δm²-sin²θ scan') as pbar:
        for i, del_m2 in enumerate(del_m2_range):
            for j, sin2_theta in enumerate(sin2_theta_range):
                theta = np.arcsin(np.sqrt(sin2_theta))
                rate = calculate_rate(beta, tau, exp_eng, del_m2, theta, n_jobs=n_jobs)
                chi2_grid[i, j] = chi_sq(exp_rate, rate, sigma)
                pbar.update(1)
    
    return chi2_grid

def chi_square_beta_scan(beta_range, tau, exp_eng, exp_rate, sigma, 
                         del_m2_fixed, sin2_theta_fixed, n_jobs=-1):
    """
    Perform chi-squared scan over beta parameter
    with fixed del_m2 and sin^2(theta)
    """
    chi2_beta = np.zeros(len(beta_range))
    theta = np.arcsin(np.sqrt(sin2_theta_fixed))
    
    for i, beta in enumerate(tqdm(beta_range, desc='β scan')):
        rate = calculate_rate(beta, tau, exp_eng, del_m2_fixed, theta, n_jobs=n_jobs)
        chi2_beta[i] = chi_sq(exp_rate, rate, sigma)
    
    return chi2_beta


# Implementation
exp_Pee = pd.read_csv('SK_rate.csv')
exp_eng = exp_Pee['energy']
exp_rate = exp_Pee['Pee_rate']
exp_rate_up = exp_Pee['rate_up']
exp_rate_down = exp_Pee['rate_down']

error1 = exp_rate - exp_rate_down
error2 = exp_rate_up - exp_rate
# Calculate uncertainty (sigma) as average of upper and lower errors
sigma = (error1 + error2) / 2

tau = 10*eV_to_1_by_m*1000

# Define parameter ranges for initial chi-squared scan with beta=0
del_m2_range = np.logspace(-5+np.log10(7),-5+np.log10(8), 30)
sin2_theta_range = np.linspace(0.1, 0.5, 30)

print("Step 1: Chi-squared scan with β=0 to find best-fit parameters...")
chi2_grid = chi_square_scan(0.0, tau, exp_eng, exp_rate, sigma, 
                            del_m2_range, sin2_theta_range, n_jobs=-1)

# Find minimum chi-squared
min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
best_del_m2 = del_m2_range[min_idx[0]]
best_sin2_theta = sin2_theta_range[min_idx[1]]
min_chi2_at_beta0 = chi2_grid[min_idx]

print(f"\nBest fit parameters at β=0:")
print(f"Δm² = {best_del_m2:.3e} eV²")
print(f"sin²θ = {best_sin2_theta:.4f}")
print(f"θ = {np.arcsin(np.sqrt(best_sin2_theta)):.4f} rad ({np.degrees(np.arcsin(np.sqrt(best_sin2_theta))):.2f}°)")
print(f"Minimum χ² = {min_chi2_at_beta0:.2f}")

# Step 2: Scan beta with fixed best-fit parameters
print("\nStep 2: Chi-squared scan varying β...")
beta_range = np.linspace(0, 0.2, 100)
chi2_beta = chi_square_beta_scan(beta_range, tau, exp_eng, exp_rate, sigma,
                                 best_del_m2, best_sin2_theta, n_jobs=-1)

# Find minimum chi-squared for beta scan
min_idx_beta = np.argmin(chi2_beta)
best_beta = beta_range[min_idx_beta]
min_chi2_beta = chi2_beta[min_idx_beta]

print(f"\nBest fit β:")
print(f"β = {best_beta:.4f}")
print(f"Minimum χ² = {min_chi2_beta:.2f}")

# Step 3: Plot chi-squared vs beta with confidence intervals
plt.figure(figsize=(10, 6))

# Calculate Delta chi-squared relative to minimum at beta=0 (Standard Model)
# This is the reference point from Step 1
delta_chi2_beta = chi2_beta - min_chi2_at_beta0

plt.plot(beta_range, delta_chi2_beta, 'b-', linewidth=2, label='Δχ²(β)')

# Confidence intervals for 1 DOF (now relative to 0)
delta_chi2_1sigma = 1.0  # Δχ² = 1 for 68% CL
delta_chi2_2sigma = 4.0  # Δχ² = 4 for 95% CL

plt.axhline(y=delta_chi2_1sigma, color='g', linestyle='--', linewidth=1.5, label='1σ (Δχ²=1)')
plt.axhline(y=delta_chi2_2sigma, color='r', linestyle='--', linewidth=1.5, label='2σ (Δχ²=4)')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.axvline(x=0, color='k', linestyle=':', linewidth=1, label='SM (β=0)')

plt.xlabel('β', fontsize=12)
plt.ylabel('Δχ² = χ²(β) - χ²_min(β=0)', fontsize=12)
plt.title(f'Δχ² vs β (Fixed: Δm²={best_del_m2:.3e} eV², sin²θ={best_sin2_theta:.4f})', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chi2_vs_beta.png', dpi=300)
plt.show()

print(f"\n1σ confidence interval: Δχ² = {delta_chi2_1sigma:.2f}")
print(f"2σ confidence interval: Δχ² = {delta_chi2_2sigma:.2f}")
print(f"Δχ²(β=0) = {delta_chi2_beta[0]:.2f} (should be 0)")
print(f"Reference χ²_min(β=0) = {min_chi2_at_beta0:.2f}")

# Step 4: Plot rate vs energy for different beta values
print("\nStep 3: Generating rate vs energy comparison for different beta values...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot chi-squared vs beta on left panel
delta_chi2_beta_replot = chi2_beta - min_chi2_at_beta0
ax1.plot(beta_range, delta_chi2_beta_replot, 'b-', linewidth=2, label='Δχ²(β)')
ax1.axhline(y=delta_chi2_1sigma, color='g', linestyle='--', linewidth=1.5, label='1σ (Δχ²=1)')
ax1.axhline(y=delta_chi2_2sigma, color='r', linestyle='--', linewidth=1.5, label='2σ (Δχ²=4)')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax1.axvline(x=0, color='k', linestyle=':', linewidth=1, label='SM (β=0)')
ax1.set_xlabel('β', fontsize=12)
ax1.set_ylabel('Δχ² = χ²(β) - χ²_min(β=0)', fontsize=12)
ax1.set_title(f'Δχ² vs β (Fixed: Δm²={best_del_m2:.3e} eV², sin²θ={best_sin2_theta:.4f})', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot rate vs energy for different beta values on right panel
error_upper = exp_rate_up - exp_rate
error_lower = exp_rate - exp_rate_down
ax2.errorbar(exp_eng, exp_rate, yerr=[error_lower, error_upper], 
             fmt='o', label='Experimental Data', color='#E74C3C', capsize=5, 
             markersize=8, linewidth=2, elinewidth=2, capthick=2, zorder=3, alpha=0.8)

# Select representative beta values to plot
beta_values_to_plot = [0.0, 0.01,0.03,0.05,0.1]
colors = ['#2ECC71', '#3498DB', '#9B59B6', '#F39C12', "#341A03"]
theta = np.arcsin(np.sqrt(best_sin2_theta))

for beta_val, color in zip(beta_values_to_plot, colors):
    rate = calculate_rate(beta_val, tau, exp_eng, best_del_m2, theta, n_jobs=-1)
    label = f'β={beta_val:.2f}' + (' (Best fit)' if beta_val == best_beta else ' (β=0)' if beta_val == 0.0 else '')
    ax2.plot(exp_eng, rate, 'o-', linewidth=2.5, markersize=6, color=color,
             label=label, zorder=2, alpha=0.9)

ax2.set_xlabel('Energy (MeV)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Pₑₑ Rate', fontsize=12, fontweight='bold')
ax2.set_title(f'Rate vs Energy Comparison (Δm²={best_del_m2:.3e}, sin²θ={best_sin2_theta:.3f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.9, loc='best')
ax2.grid(alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=10)
ax2.set_ylim(0.3, 0.7)

plt.tight_layout()
plt.savefig('chi2_and_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.show()