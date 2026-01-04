import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numba import njit
from tqdm import tqdm
import pandas as pd

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (100/eV_to_1_by_m)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1                                

@njit
def N_e(r):
    if r <= R_sol:
        return 245*N_A*np.exp(-r*10.45/R_sol)*one_by_cm3_to_eV3
    else:
        return 0.0

@njit
def k(N, beta, tau):
    return tau*(G_F*beta*N)** 2

@njit
def A(E, N, del_m2, theta):
    return -del_m2*np.cos(2*theta)/(4*E) + G_F*N/np.sqrt(2)

@njit
def B(E, del_m2, theta):
    return del_m2*np.sin(2*theta)/(4*E)

def solar_solver(E, beta, tau, del_m2, theta, n_slabs=100000, r_i=0.0, r_f=1.0):
    E = E * 1e6  # MeV to eV
    dx = (r_f - r_i) * R_earth / n_slabs
    r_vals = np.linspace(r_i, r_f, n_slabs)
    psi = np.array([1.0, 0.0, 0.0])
    for i in range(n_slabs):
        r = r_vals[i]*R_earth
        N = N_e(r)
        k_r = k(N, beta, tau)
        A_r = A(E, N, del_m2, theta)
        B_E = B(E, del_m2, theta)
        M = np.array([[0.0, 0.0, B_E],
                      [0.0, k_r, -A_r],
                      [-B_E, A_r, k_r]])
        U = expm(-2 * M * dx)
        psi = U @ psi
    # Return final probability
    return (psi[0] + 1) * 0.5

def chi_sq(exp, th, sigma):
    return np.sum(((exp - th) / sigma) ** 2)

def compute_chi2_for_probabilities(dm2_grid, tan2theta_grid, beta=0.0, tau=10*eV_to_1_by_m*1000):
    """
    Compute chi-squared over parameter grid using experimental vs theoretical probabilities
    """
    # Load null hypothesis data (experimental probabilities)
    null_data = pd.read_csv('Prob[0].csv')
    E_vals = null_data['energy'].values
    exp_probabilities = np.array(null_data['results'].values)
    
    # Create parameter list
    param_list = [(dm2, tan2theta) for dm2 in dm2_grid for tan2theta in tan2theta_grid]
    
    print(f"Starting chi-squared calculation for {len(param_list)} parameter combinations...")
    results = []
    
    # Main progress bar for parameter combinations
    for i, (dm2, tan2theta) in enumerate(tqdm(param_list, desc="Parameter Grid")):
        theta = np.arctan(np.sqrt(tan2theta))
        
        # Compute theoretical probabilities for this parameter set
        theo_probabilities = np.zeros(len(E_vals))
        for j, E in enumerate(E_vals):
            theo_probabilities[j] = solar_solver(E, beta, tau, dm2, theta)
        
        # Calculate chi-squared using 10% relative error assumption
        sigma = 0.1 * exp_probabilities
        chi2_val = chi_sq(exp_probabilities, theo_probabilities, sigma)
        
        results.append(chi2_val)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{len(param_list)} - dm²={dm2:.2e}, tan²θ={tan2theta:.3f}, χ²={chi2_val:.2f}")
    
    return np.array(results).reshape(len(dm2_grid), len(tan2theta_grid))

if __name__ == '__main__':
    # Define parameter grids
    dm2_grid = np.logspace(-5, -4, 10)
    tan2theta_grid = np.linspace(0.3, 0.6, 10)
    
    # Physics parameters
    beta_test = 0.05
    
    print("Computing chi-squared over parameter grid...")
    chi2_grid = compute_chi2_for_probabilities(dm2_grid, tan2theta_grid, beta_test)
    
    # Find best-fit parameters
    min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    best_dm2 = dm2_grid[min_idx[0]]
    best_tan2theta = tan2theta_grid[min_idx[1]]
    min_chi2 = chi2_grid[min_idx]
    
    print(f"\nBest fit: Δm² = {best_dm2:.2e} eV², tan²θ = {best_tan2theta:.3f}")
    print(f"Minimum χ² = {min_chi2:.2f}")
    
    # Save results
    results_df = pd.DataFrame({
        'dm2': np.repeat(dm2_grid, len(tan2theta_grid)),
        'tan2theta': np.tile(tan2theta_grid, len(dm2_grid)),
        'chi2': chi2_grid.flatten()
    })
    results_df.to_csv(f'chi2_results (beta = {beta_test}).csv', index=False)
    
    print(f"Results saved to chi2_results (beta = {beta_test}).csv")