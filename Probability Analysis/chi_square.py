import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.constants import physical_constants as pc
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
    """Propagate neutrino through the Sun using density profile from CSV."""
    E = E * 1e6  # MeV to eV
    
    r_min = r_i * R_sol
    r_max = r_f * R_sol
    
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
    """Propagate neutrino from solar surface to Earth in vacuum"""
    E = E * 1e6  # MeV to eV
    dx = (R_earth - R_sol) / n_slabs
    psi = psi_initial
    Pee_profile = np.zeros(n_slabs)
    
    B_E = B(E, del_m2, theta)
    A_vac = A(E, 0.0, del_m2, theta)
    
    M_vac = np.array([[0.0, 0.0, B_E],
                      [0.0, 0.0, -A_vac],
                      [-B_E, A_vac, 0.0]])
    U_vac = expm(-2 * M_vac * dx)
    
    for i in range(n_slabs):
        psi = U_vac @ psi
        Pee_profile[i] = (psi[0] + 1)*0.5
    return Pee_profile

def avg_Pee(E, beta, tau, del_m2, theta):
    """Calculate average oscillation probability"""
    psi = solar_solver(E, beta, tau, del_m2, theta)
    Pee_profile_vac = vacuum_propagation(E, psi, del_m2, theta)
    return np.mean(Pee_profile_vac)

def chi_sq(exp, th, sigma):
    """Calculate chi-squared"""
    return np.sum(((exp - th) / sigma) ** 2)

def calculate_rate(beta, tau, exp_eng, del_m2, theta):
    """Calculate theoretical rate for given parameters"""
    Pee = np.zeros(len(exp_eng))
    for i, E in enumerate(exp_eng):
        Pee[i] = avg_Pee(E, beta, tau, del_m2, theta)
    return 0.16 + 0.84*Pee

def find_chi2_min_beta0(del_m2_range, sin2_theta_range, tau, exp_eng, exp_rate, sigma):
    """Find minimum chi-squared for beta=0 by scanning over del_m2 and sin2_theta"""
    print("\nScanning parameter space for β=0...")
    chi2_grid = np.zeros((len(del_m2_range), len(sin2_theta_range)))
    
    total_pairs = len(del_m2_range) * len(sin2_theta_range)
    pair_count = 0
    
    for i, del_m2 in enumerate(del_m2_range):
        for j, sin2_theta in enumerate(sin2_theta_range):
            theta = np.arcsin(np.sqrt(sin2_theta))
            rate_theory = calculate_rate(0.0, tau, exp_eng, del_m2, theta)
            chi2_grid[i, j] = chi_sq(exp_rate, rate_theory, sigma)
            pair_count += 1
            print(f'\rProgress: {pair_count}/{total_pairs} ({100*pair_count/total_pairs:.1f}%) | Δm²={del_m2:.2e}, sin²θ={sin2_theta:.3f}, χ²={chi2_grid[i,j]:.2f}', end='', flush=True)
    
    print()
    min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    chi2_min = chi2_grid[min_idx]
    best_del_m2 = del_m2_range[min_idx[0]]
    best_sin2_theta = sin2_theta_range[min_idx[1]]
    
    print(f"\nβ=0 Results:")
    print(f"χ²_min = {chi2_min:.4f}")
    print(f"Best Δm² = {best_del_m2:.2e} eV²")
    print(f"Best sin²θ = {best_sin2_theta:.4f}")
    
    return chi2_min, best_del_m2, best_sin2_theta, chi2_grid

def calculate_delta_chi2_vs_beta(beta_range, chi2_min_beta0, best_del_m2_beta0, 
                                  best_sin2_theta_beta0, tau, exp_eng, exp_rate, sigma):
    """Calculate Δχ² as a function of beta with Δm² and sin²θ FIXED at their β=0 best-fit values"""
    theta_beta0 = np.arcsin(np.sqrt(best_sin2_theta_beta0))
    delta_chi2_values = []
    
    print(f"\nCalculating Δχ² vs β with FIXED parameters:")
    print(f"  Δm² = {best_del_m2_beta0:.2e} eV²")
    print(f"  sin²θ = {best_sin2_theta_beta0:.4f}\n")
    
    for i, beta in enumerate(beta_range):
        print(f"\nEvaluating β = {beta:.4f} ({i+1}/{len(beta_range)})...")
        
        rate_theory = calculate_rate(beta, tau, exp_eng, best_del_m2_beta0, theta_beta0)
        
        chi2_beta = chi_sq(exp_rate, rate_theory, sigma)
        delta_chi2 = chi2_beta - chi2_min_beta0
        delta_chi2_values.append(delta_chi2)
        
        print(f"  χ²(β={beta:.4f}) = {chi2_beta:.4f}")
        print(f"  Δχ² = {delta_chi2:.4f}")
    
    return np.array(delta_chi2_values)

def plot_delta_chi2_vs_beta(beta_range, delta_chi2_values, save_path='delta_chi2_vs_beta.png'):
    """Plot Δχ² as a function of beta with 1 DOF confidence level lines"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(beta_range, delta_chi2_values, 'o-', color='darkblue', 
            linewidth=2.5, markersize=8, label='Δχ²(β)', zorder=3)
    
    # Confidence level lines for 1 DOF
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, 
              label='1σ (Δχ²=1.0, 1 DOF)', zorder=2)
    ax.axhline(y=4.0, color='green', linestyle='--', linewidth=2, 
              label='2σ (Δχ²=4.0, 1 DOF)', zorder=2)
    ax.axhline(y=9.0, color='red', linestyle='--', linewidth=2, 
              label='3σ (Δχ²=9.0, 1 DOF)', zorder=2)
    
    # Mark crossing points
    for level, color, name in [(1.0, 'blue', '1σ'), (4.0, 'green', '2σ'), (9.0, 'red', '3σ')]:
        crossing_indices = np.where(np.diff(np.sign(delta_chi2_values - level)))[0]
        for idx in crossing_indices:
            if idx + 1 < len(beta_range):
                beta_cross = np.interp(level, 
                                       [delta_chi2_values[idx], delta_chi2_values[idx+1]], 
                                       [beta_range[idx], beta_range[idx+1]])
                ax.plot(beta_cross, level, 'o', color=color, markersize=10, 
                       markeredgecolor='black', markeredgewidth=1.5, zorder=4)
    
    ax.set_xlabel('β', fontsize=15, fontweight='bold')
    ax.set_ylabel('Δχ²', fontsize=15, fontweight='bold')
    ax.set_title('Δχ² vs β (Δm² and sin²θ fixed at β=0 best-fit values)\n1 Degree of Freedom', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Best-fit info box
    textstr = f'Δχ²_min = {np.min(delta_chi2_values):.3f}\nat β = {beta_range[np.argmin(delta_chi2_values)]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nΔχ² vs β plot saved to {save_path}")
    plt.show()


# Implementation
exp_Pee = pd.read_csv('SK III.csv')
exp_eng = np.array(exp_Pee['energy'].values)
exp_rate = np.array(exp_Pee['Pee_rate'].values)
error_up = np.array(exp_Pee['error_up'].values)
error_down = np.array(exp_Pee['error_down'].values)

# Calculate uncertainty (sigma) as average of upper and lower errors
sigma = (error_up + error_down) / 2 # type: ignore

tau = 10*eV_to_1_by_m*1000

# Define parameter ranges for chi-squared scan for β=0
del_m2_range = np.logspace(-5+np.log10(7), -5+np.log10(8), 20)
sin2_theta_range = np.linspace(0.20, 0.40,20)

# Step 1: Find chi-squared minimum for beta=0
chi2_min_beta0, best_del_m2_beta0, best_sin2_theta_beta0, chi2_grid = find_chi2_min_beta0(
    del_m2_range, sin2_theta_range, tau, exp_eng, exp_rate, sigma
)

# Calculate reduced chi-squared (χ²/dof)
n_data_points = len(exp_eng)
n_parameters = 2  # del_m2 and sin2_theta
dof = n_data_points - n_parameters

print("\n" + "="*80)
print("BEST-FIT PARAMETERS FROM β=0 ANALYSIS")
print("="*80)
print(f"χ²_min(β=0) = {chi2_min_beta0:.4f}")
print(f"Reduced χ²/dof = {chi2_min_beta0/dof:.4f}")
print(f"Δm²₂₁ = {best_del_m2_beta0:.2e} eV²")
print(f"sin²θ₁₂ = {best_sin2_theta_beta0:.4f}")
print(f"θ₁₂ = {np.arcsin(np.sqrt(best_sin2_theta_beta0)):.4f} rad ({np.degrees(np.arcsin(np.sqrt(best_sin2_theta_beta0))):.2f}°)")
print("="*80 + "\n")

# Step 2: Calculate Δχ² vs β with FIXED Δm² and sin²θ
beta_range = np.linspace(0.0, 0.15, 30)
delta_chi2_values = calculate_delta_chi2_vs_beta(
    beta_range, chi2_min_beta0, best_del_m2_beta0, best_sin2_theta_beta0,
    tau, exp_eng, exp_rate, sigma
)

# Step 3: Plot Δχ² vs β
plot_delta_chi2_vs_beta(beta_range, delta_chi2_values, 
                       save_path='delta_chi2_vs_beta_SK3.png')

# Create contour plot for β=0 analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

X, Y = np.meshgrid(sin2_theta_range, del_m2_range)

# Calculate confidence levels (for 2 DOF: Δχ² = 2.30, 6.18, 11.83 for 1σ, 2σ, 3σ)
levels_conf = [chi2_min_beta0 + 2.30, chi2_min_beta0 + 6.18, chi2_min_beta0 + 11.83]
level_labels = ['1σ (68%)', '2σ (95%)', '3σ (99.7%)']

# Plot filled contours
contourf = ax1.contourf(X, Y, chi2_grid, levels=20, cmap='viridis')
cbar = plt.colorbar(contourf, ax=ax1, label='χ²')
cbar.set_label('χ²', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Plot confidence level contour lines
contour_colors = ['red', 'orange', 'cyan']
CS = ax1.contour(X, Y, chi2_grid, levels=levels_conf, colors=contour_colors, linewidths=3)

# Manual legend for contour levels
for level, label, color in zip(levels_conf, level_labels, contour_colors):
    ax1.plot([], [], '-', linewidth=3, color=color, label=f'{label}: χ²={level:.1f}')

# Mark best fit point
ax1.plot(best_sin2_theta_beta0, best_del_m2_beta0, 'r*', markersize=25, 
        label=f'Best fit: χ²={chi2_min_beta0:.2f}, χ²/dof={chi2_min_beta0/dof:.2f}',
        markeredgecolor='white', markeredgewidth=1.5, zorder=5)

ax1.set_xlabel('sin²θ', fontsize=14, fontweight='bold')
ax1.set_ylabel('Δm² (eV²)', fontsize=14, fontweight='bold')
ax1.set_title(f'χ² Contours for β=0 (SK III)', fontsize=16, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=11)

# Best fit comparison
best_rate = calculate_rate(0.0, tau, exp_eng, best_del_m2_beta0, 
                          np.arcsin(np.sqrt(best_sin2_theta_beta0)))

# Use polyfit to smooth the best fit curve
deg = 4
coeffs = np.polyfit(exp_eng, best_rate, deg)
exp_eng_smooth = np.linspace(exp_eng.min(), exp_eng.max(), 200)
best_rate_smooth = np.polyval(coeffs, exp_eng_smooth)

ax2.errorbar(exp_eng, exp_rate, yerr=[error_down, error_up], 
             fmt='o', label='Experimental Data (SK III)', color='#E74C3C', capsize=5, 
             markersize=8, linewidth=2, elinewidth=2, capthick=2, zorder=3)
ax2.plot(exp_eng_smooth, best_rate_smooth, '-', linewidth=3, color='#3498DB',
         label=f'Best Fit (Δm²={best_del_m2_beta0:.2e}, sin²θ={best_sin2_theta_beta0:.3f})', zorder=2)
ax2.set_xlabel('Energy (MeV)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Pₑₑ Rate', fontsize=14, fontweight='bold')
ax2.set_title('Best Fit vs Experimental Data (β=0)', fontsize=16, fontweight='bold', pad=15)
ax2.legend(fontsize=11, framealpha=0.9)
ax2.grid(alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=11)
ax2.set_ylim(0.3, 0.6)

plt.tight_layout()
plt.savefig('chi_square_analysis_SK3.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
results_df = pd.DataFrame({
    'beta': beta_range,
    'delta_chi2': delta_chi2_values,
    'chi2': delta_chi2_values + chi2_min_beta0
})
results_df.to_csv('delta_chi2_vs_beta_results_SK3.csv', index=False)