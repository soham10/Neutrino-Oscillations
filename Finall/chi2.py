import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import expm
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit

# Constants copied from Rates.py
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
total_flux = phi_B0 + phi_hep

# Load data - match Rates.py file paths
sigmas = np.loadtxt('sigma.csv', delimiter=',', skiprows=1)
E_nu_vals_grid = sigmas[:, 0]
Te_vals_grid = sigmas[:, 1]
sigma_e_grid = sigmas[:, 2] * 1e-46
sigma_x_grid = sigmas[:, 3] * 1e-46

lambda_df = pd.read_csv('lambda.csv')
lambda_E = np.array(lambda_df['energy'].values, dtype=float)
lambda_val_B = np.array(lambda_df['lambda'].values, dtype=float)

hep_df = pd.read_csv('hep.csv')
hep_E = np.array(hep_df['energy'].values, dtype=float)
lambda_val_hep = np.array(hep_df['lambda'].values, dtype=float)

E_nu_vals = np.unique(E_nu_vals_grid)
Te_bin_centers = np.unique(Te_vals_grid)

lambda_interp_hep_on_grid = np.interp(E_nu_vals, hep_E, lambda_val_hep, left=0.0, right=0.0)
lambda_combined = (phi_B0 * lambda_val_B + phi_hep * lambda_interp_hep_on_grid) / (phi_B0 + phi_hep)

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
    mask = (r_frac >= 0.9)
    return np.mean(Pee_profile[mask])

def chi_sq(exp, th, sigma):
    return np.sum(((exp - th) / sigma) ** 2)

def compute_chi2_for_rates(dm2_grid, tan2theta_grid, beta=0.0, tau=10*eV_to_1_by_m*1000):
    """
    Compute chi-squared over parameter grid using experimental vs theoretical rates
    """
    # Load experimental data
    exp_data = np.loadtxt('experiment.csv', delimiter=',', skiprows=1)
    exp_te = exp_data[:, 0]
    exp_rate = exp_data[:, 1] 
    
    # Create parameter list
    param_list = [(dm2, tan2theta) for dm2 in dm2_grid for tan2theta in tan2theta_grid]
    
    # Sequential computation with main progress bar
    print(f"Starting chi-squared calculation for {len(param_list)} parameter combinations...")
    results = []
    
    # Main progress bar for parameter combinations
    for i, (dm2, tan2theta) in enumerate(tqdm(param_list, 
                                            desc="Parameter Grid", 
                                            unit="combination")):
        theta = np.arctan(np.sqrt(tan2theta))
        
        # Pre-compute oscillation probabilities for all energies for this parameter set
        print(f"\nPre-computing Pee for dm²={dm2:.2e}, tan²θ={tan2theta:.3f}...")
        Pee_values = np.zeros(len(E_nu_vals))
        for j, E_nu in enumerate(tqdm(E_nu_vals, desc="Computing Pee", leave=False)):
            Pee_values[j] = avg_Pee(E_nu, beta, tau, dm2, theta)
        
        # Calculate theoretical rates for experimental Te bins using Rates.py logic
        theo_rates = np.zeros_like(exp_te)
        
        for k, Te_exp in enumerate(tqdm(exp_te, desc="Computing rates for Te bins", leave=False)):
            total_rate_for_Te = 0.0
            
            # Sum over all neutrino energies for this experimental Te bin
            for j, E_nu in enumerate(E_nu_vals):
                # Find the cross section for this (E_nu, Te_exp) pair
                mask = (E_nu_vals_grid == E_nu) & (Te_vals_grid == Te_exp)
                
                if np.any(mask):
                    sigma_e = sigma_e_grid[mask][0]
                    sigma_x = sigma_x_grid[mask][0]
                    
                    # Get pre-computed oscillation probability and spectrum weight
                    Pee_val = Pee_values[j]
                    lambda_val = lambda_combined[j]
                    
                    # Calculate rate contribution directly
                    rate_contrib = total_flux * lambda_val * (sigma_e * Pee_val + sigma_x * (1 - Pee_val))
                    rate_contrib *= N_e_tgt * 24 * 3600  # Events/day
                    
                    total_rate_for_Te += rate_contrib
            
            theo_rates[k] = total_rate_for_Te
        
        # If no errors provided, assume 10% relative error
        chi2_val = chi_sq(exp_rate, theo_rates, 0.1 * exp_rate)
        
        results.append(chi2_val)
        
        # Print progress summary
        print(f"Completed {i+1}/{len(param_list)} - dm²={dm2:.2e}, tan²θ={tan2theta:.3f}, χ²={chi2_val:.2f}")
    
    return np.array(results).reshape(len(dm2_grid), len(tan2theta_grid))

if __name__ == '__main__':
    # Define parameter grids
    dm2_grid = np.logspace(-5, -4, 30)
    tan2theta_grid = np.logspace(-1, 0, 30)  
    
    # Physics parameters
    beta = 0.0    
    print("Computing chi-squared over parameter grid...")
    chi2_grid = compute_chi2_for_rates(dm2_grid, tan2theta_grid, beta)
    
    # Find best-fit parameters
    min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    best_dm2 = dm2_grid[min_idx[0]]
    best_tan2theta = tan2theta_grid[min_idx[1]]
    min_chi2 = chi2_grid[min_idx]
    
    print(f"Best fit: Δm² = {best_dm2:.2e} eV², tan²θ = {best_tan2theta:.3f}")
    print(f"Minimum χ² = {min_chi2:.2f}")
    
    # Save results
    np.savetxt('chi2_results.csv', 
               np.column_stack([dm2_grid.repeat(len(tan2theta_grid)), 
                               np.tile(tan2theta_grid, len(dm2_grid)), 
                               chi2_grid.flatten()]),
               delimiter=',', header='dm2,tan2theta,chi2', comments='')
    
    # Create contour plot
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(tan2theta_grid, dm2_grid)
    
    # Plot contours at confidence levels
    levels = [min_chi2 + 2.30, min_chi2 + 5.99, min_chi2 + 9.21]  # 68%, 95%, 99% CL
    contours = plt.contour(X, Y, chi2_grid, levels=levels, colors=['red', 'blue', 'green'])
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    # Mark best fit point
    plt.plot(best_tan2theta, best_dm2, 'k*', markersize=15, label=f'Best fit: χ²={min_chi2:.1f}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('tan²θ')
    plt.ylabel('Δm² (eV²)')
    plt.title('χ² Contours for Solar Neutrino Oscillation Parameters')
    plt.colorbar(label='χ²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('chi2_contours.png', dpi=300)
    
    print("Results saved to 'chi2_results.csv' and plot saved to 'chi2_contours.png'")