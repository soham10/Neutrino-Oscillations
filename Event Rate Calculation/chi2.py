import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import expm
from tqdm import tqdm
from numba import njit

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (1.973*1e-5)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1

# SK settings
fiducial_mass = 22.5e9 # g
N_e_tgt = 10*fiducial_mass*N_A/18  # number of target electrons
phi_B0 = 5.25e6  # cm^-2 s^-1
phi_hep = 7.88e3  # cm^-2 s^-1

# Load data
sigmas = np.loadtxt('SKIV/sigma_SKIV.csv', delimiter=',', skiprows=1)
E_nu_vals_grid = sigmas[:, 0]
Te_vals_grid = sigmas[:, 1]
sigma_e_grid = sigmas[:, 2]
sigma_x_grid = sigmas[:, 3]

lambda_df = pd.read_csv('lambda.csv')
lambda_E = np.array(lambda_df['energy'].values, dtype=float)
lambda_val_B = np.array(lambda_df['lambda'].values, dtype=float)/1000

hep_df = pd.read_csv('hep.csv')
hep_E = np.array(hep_df['energy'].values, dtype=float)
lambda_val_hep = np.array(hep_df['lambda'].values, dtype=float)

E_nu_vals = np.unique(E_nu_vals_grid)
Te_bin_centers = np.unique(Te_vals_grid)

# Create combined flux spectrum
lambda_interp_B_on_grid = np.interp(E_nu_vals, lambda_E, lambda_val_B, left=0.0, right=0.0)
lambda_interp_hep_on_grid = np.interp(E_nu_vals, hep_E, lambda_val_hep, left=0.0, right=0.0)
flux_spectrum = phi_B0 * lambda_interp_B_on_grid + phi_hep * lambda_interp_hep_on_grid

# Load solar electron density profile from CSV
solar_data = pd.read_csv('solar.csv')
r_over_rsol = solar_data['R/R_s'].values.astype(float)
log10_n_over_NA = solar_data['log10(n/N_A)'].values.astype(float)

# Convert to actual radius in eV^-1 and number density in eV^3
r_data = np.array(r_over_rsol) * R_sol
n_e_data = (10**np.array(log10_n_over_NA)) * N_A * one_by_cm3_to_eV3

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

def vacuum_propagation(E, psi_initial, del_m2, theta, n_slabs=10000):
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

def chi_sq_with_nuisance(exp, th, sigma, alpha, sigma_alpha):
    """Calculate chi-squared with nuisance parameter alpha"""
    return np.sum(((exp - alpha*th)/sigma)**2) + ((alpha - 1)/sigma_alpha)**2

def best_alpha(exp, th, sigma, sigma_alpha):
    """Find best-fit nuisance parameter alpha that minimizes chi-squared"""
    num = np.sum(exp*th/sigma**2) + 1/sigma_alpha**2
    den = np.sum(th**2/sigma**2) + 1/sigma_alpha**2
    return num/den

def calculate_theoretical_rates(beta, tau, del_m2, theta, exp_te):
    """Calculate theoretical rates for given parameters at experimental Te values"""
    Pee_values = np.zeros(len(E_nu_vals))
    for i, E_nu in enumerate(E_nu_vals):
        Pee_values[i] = avg_Pee(E_nu, beta, tau, del_m2, theta)
    
    theo_rates = np.zeros_like(exp_te)
    
    for k, Te_exp in enumerate(exp_te):
        integrand = np.zeros(len(E_nu_vals))
        
        for i, E_nu in enumerate(E_nu_vals):
            # Find cross sections for this (E_nu, T_e) pair
            mask = (E_nu_vals_grid == E_nu) & (np.abs(Te_vals_grid - Te_exp) < 1e-6)
            
            if np.any(mask):
                sigma_e = sigma_e_grid[mask][0]
                sigma_x = sigma_x_grid[mask][0]
                Pee_val = Pee_values[i]
                flux_at_E = flux_spectrum[i]
                sigma_eff = sigma_e * Pee_val + sigma_x * (1 - Pee_val)
                integrand[i] = flux_at_E * sigma_eff
        
        integral = np.trapz(integrand, E_nu_vals)
        theo_rates[k] = integral * N_e_tgt * 24 * 3600 * 2970
    
    return theo_rates

def profile_chi2_for_beta(beta, del_m2_range, sin2_theta_range, tau, exp_te, exp_rate, sigma, sigma_alpha):
    """Minimize chi^2 over (del_m2, sin^2 theta, alpha) for fixed beta."""
    chi2_min = np.inf
    best_del_m2 = None
    best_sin2_theta = None

    for del_m2 in del_m2_range:
        for sin2_theta in sin2_theta_range:
            theta = np.arcsin(np.sqrt(sin2_theta))
            rate_theory = calculate_theoretical_rates(beta, tau, del_m2, theta, exp_te)
            
            # Find best-fit alpha and calculate chi-squared with nuisance parameter
            alpha = best_alpha(exp_rate, rate_theory, sigma, sigma_alpha)
            chi2_val = chi_sq_with_nuisance(exp_rate, rate_theory, sigma, alpha, sigma_alpha)

            if chi2_val < chi2_min:
                chi2_min = chi2_val
                best_del_m2 = del_m2
                best_sin2_theta = sin2_theta

    return chi2_min, best_del_m2, best_sin2_theta

def calculate_delta_chi2_vs_beta_profiled(beta_range, del_m2_range, sin2_theta_range,
                                          tau, exp_te, exp_rate, sigma, sigma_alpha):
    """Compute profiled Δχ²(β) by minimizing over Δm², sin²θ, and α for each β."""
    profiled_chi2 = []
    best_params = []

    print("\nProfiling over Δm², sin²θ, and α for each β...\n")

    for i, beta in enumerate(beta_range):
        print(f"β = {beta:.4f} ({i+1}/{len(beta_range)})")
        chi2_min_beta, best_del_m2, best_sin2_theta = profile_chi2_for_beta(
            beta, del_m2_range, sin2_theta_range, tau, exp_te, exp_rate, sigma, sigma_alpha
        )
        print(f"χ²_min = {chi2_min_beta:.4f} at Δm²={best_del_m2:.2e}, sin²θ={best_sin2_theta:.4f}")

        profiled_chi2.append(chi2_min_beta)
        best_params.append((best_del_m2, best_sin2_theta))

    profiled_chi2 = np.array(profiled_chi2)
    delta_chi2 = profiled_chi2 - np.min(profiled_chi2)

    return delta_chi2, profiled_chi2, best_params

def plot_delta_chi2_vs_beta(beta_range, delta_chi2_values, save_path=None):
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
    ax.set_title('Δχ² vs β', 
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nΔχ² vs β plot saved to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    # Load experimental data
    plot_data = pd.read_csv('SKIV/plot-data_SKIV.csv')
    plot_data = plot_data.to_numpy(dtype=float)
    exp_te = plot_data[:, 0]
    exp_rate = plot_data[:, 1]*2970*22.5/365  # Scale to total events in SK
    exp_sigma = (plot_data[:,3] + plot_data[:,4])/2
    exp_sigma = exp_sigma * 22.5*2970/365
    
    # Parameters
    tau = 10*eV_to_1_by_m*1000
    sigma_alpha = 0.1  # Nuisance parameter uncertainty (10% normalization uncertainty)
    
    # Define parameter grids
    del_m2_range = np.logspace(-5+np.log10(7),-5+np.log10(8), 5)
    sin2_theta_range = np.linspace(0.25, 0.35, 5)
    beta_range = np.linspace(0.0, 0.15, 30)
    
    # Perform profiled likelihood analysis
    delta_chi2_values, profiled_chi2, best_params = calculate_delta_chi2_vs_beta_profiled(
        beta_range, del_m2_range, sin2_theta_range, tau, exp_te, exp_rate, exp_sigma, sigma_alpha
    )
    
    # Find global minimum
    min_idx = np.argmin(profiled_chi2)
    chi2_global_min = profiled_chi2[min_idx]
    best_beta = beta_range[min_idx]
    best_del_m2, best_sin2_theta = best_params[min_idx]

    print(f"χ²_min = {chi2_global_min:.4f}")
    print(f"β = {best_beta:.4f}")
    print(f"Δm²₂₁ = {best_del_m2:.2e} eV²")
    print(f"sin²θ₁₂ = {best_sin2_theta:.4f}")
    print(f"σ_α = {sigma_alpha:.2f} (normalization uncertainty)")
    
    # Plot Δχ² vs β
    plot_delta_chi2_vs_beta(beta_range, delta_chi2_values, 
                           save_path='SKIV/delta_chi2_vs_beta_profiled.png')
    
    # Save results
    results_df = pd.DataFrame({
        'beta': beta_range,
        'delta_chi2': delta_chi2_values,
        'chi2': profiled_chi2,
        'best_del_m2': [p[0] for p in best_params],
        'best_sin2_theta': [p[1] for p in best_params]
    })
    results_df.to_csv('SKIV/delta_chi2_vs_beta_results_SKIV.csv', index=False)