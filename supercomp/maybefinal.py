"""
Full analysis script: grid χ² scan in (Δm², tan²θ) including
- full 8B spectrum convolution,
- detector energy resolution (SK),
- smeared cross sections computed per reconstructed-energy bin,
- correlated systematic via covariance matrix,
- overlayed contours for multiple β values (same plot),
- save best-fit table and produce spectrum comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import inv, expm
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Constants ------------------------- #
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # mol^-1
m_e = 0.510998950e6  # eV
# conversion factors
eV_to_1_by_m = 8.065e5
eV_to_1_by_km = 8.065e8
R_sol = 6.9634e8 * eV_to_1_by_m   # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1
one_by_cm3_to_eV3 = (1.234e-4) ** 3
eV_to_cm = 1.97e-5  # 1 eV^-1 = 1.97e-5 cm

# ------------------------- Experimental data (Super-K) ------------------------- #
sk_energy_bins = np.array([
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0, 16.0, 20.0
])

R_exp = np.array([
    74.7, 65.0, 61.5, 54.1, 49.4, 44.3, 36.3, 28.7, 25.0, 20.8,
    16.2, 11.2, 9.85, 6.79, 5.13, 3.65, 2.46, 2.02, 1.72, 0.949, 0.341
])

sigma_exp_stat = np.array([
    (6.6+6.5)/2, (3.3+3.2)/2, (2.4+2.3)/2, (1.7+1.7)/2, (1.5+1.5)/2,
    (1.4+1.4)/2, (1.2+1.2)/2, (1.0+1.0)/2, (0.9+0.9)/2, (0.8+0.8)/2,
    (0.7+0.7)/2, (0.6+0.5)/2, (0.51+0.49)/2, (0.42+0.40)/2, (0.36+0.33)/2,
    (0.30+0.28)/2, (0.25+0.23)/2, (0.22+0.20)/2, (0.21+0.19)/2, (0.157+0.133)/2, (0.103+0.077)/2
])

n_bins = len(sk_energy_bins)-1
bin_centers = (sk_energy_bins[:-1] + sk_energy_bins[1:]) / 2
bins_low = sk_energy_bins[:-1]
bins_high = sk_energy_bins[1:]

# ------------------------- Detector response (SK) ------------------------- #
@njit
def s_SK(Tp_MeV: float) -> float:
    """SK energy resolution width (MeV)"""
    return 0.47 * np.sqrt(max(Tp_MeV, 0.0))

@njit
def R_SK(T_rec_MeV: float, Tp_MeV: float) -> float:
    s = s_SK(Tp_MeV)
    if s <= 0:
        return 0.0
    return np.exp(-0.5*((T_rec_MeV - Tp_MeV)/s)**2) / (np.sqrt(2*np.pi)*s)

# ------------------------- Weak cross sections (SM) ------------------------- #
sin2_theta_v = 0.2317
rho = 1.0126
k_e = 0.9791
k_mu = 0.9970

@njit
def couplings(is_nu_e: bool):
    if is_nu_e:
        gL = rho * (0.5 - k_e * sin2_theta_v) - 1
        gR = -rho * k_e * sin2_theta_v
    else:
        gL = rho * (0.5 - k_mu * sin2_theta_v)
        gR = -rho * k_mu * sin2_theta_v
    return gL, gR

@njit
def dsigma_dT_SM(Tp_eV: float, Ev_eV: float, is_nu_e: bool) -> float:
    z = Tp_eV / Ev_eV
    gL, gR = couplings(is_nu_e)
    term1 = gL**2
    term2 = gR**2 * (1 - z)**2
    term3 = -gL * gR * (m_e / Ev_eV) * z
    factor = 2 * (G_F**2) * m_e / np.pi
    dsig = factor * (term1 + term2 + term3)
    return max(dsig, 0.0)

# ------------------------- Smeared cross-section in a reconstructed bin -------------- #
@njit
def sigma_smeared_in_bin_single(Ev_MeV: float, is_nu_e: bool, Trec_low_MeV: float, Trec_high_MeV: float,
                               n_Tp=60, n_Trec=40):
    """Compute smeared cross section for a single bin"""
    Ev_eV = Ev_MeV * 1e6
    Tp_max_MeV = Ev_MeV / (1.0 + (m_e/1e6)/(2.0*Ev_MeV))
    if Tp_max_MeV <= 0:
        return 0.0

    Tp_vals = np.linspace(0.0, Tp_max_MeV, n_Tp)
    sigma_sum = 0.0
    
    for Tp in Tp_vals:
        s_val = s_SK(Tp)
        # Define integration range for Trec
        Trec_min = max(0.0, Trec_low_MeV - 5*s_val)
        Trec_max = min(Tp_max_MeV, Trec_high_MeV + 5*s_val)
        
        if Trec_max <= Trec_min:
            continue
            
        Trec_vals = np.linspace(Trec_min, Trec_max, n_Trec)
        R_vals = np.zeros(n_Trec)
        for k in range(n_Trec):
            R_vals[k] = R_SK(Trec_vals[k], Tp)
        
        P_rec_in_bin = np.trapz(R_vals, Trec_vals)
        dsig = dsigma_dT_SM(Tp*1e6, Ev_eV, is_nu_e)
        sigma_sum += dsig * P_rec_in_bin
    
    dTp = Tp_vals[1] - Tp_vals[0] if len(Tp_vals) > 1 else 0.0
    sigma_sum *= dTp
    return sigma_sum * (eV_to_cm**2)

# Vectorized version for parallel computation
def compute_sigma_for_bin(args):
    i, E, b, bins_low, bins_high, n_Tp, n_Trec = args
    sigma_e = sigma_smeared_in_bin_single(E, True, bins_low[b], bins_high[b], n_Tp, n_Trec)
    sigma_x = sigma_smeared_in_bin_single(E, False, bins_low[b], bins_high[b], n_Tp, n_Trec)
    return i, b, sigma_e, sigma_x

def precompute_sigma_bins_parallel(E_nu_vals, bins_low, bins_high, n_Tp=60, n_Trec=40):
    """Parallel computation of smeared cross sections"""
    nE = len(E_nu_vals)
    nB = len(bins_low)
    sigma_e_bin = np.zeros((nE, nB))
    sigma_x_bin = np.zeros((nE, nB))
    
    # Prepare arguments for parallel computation
    args_list = []
    for i, E in enumerate(E_nu_vals):
        for b in range(nB):
            args_list.append((i, E, b, bins_low, bins_high, n_Tp, n_Trec))
    
    # Parallel computation
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_sigma_for_bin)(args) for args in tqdm(args_list, desc="Computing cross sections")
    )
    
    # Organize results
    for i, b, sigma_e, sigma_x in results: # type: ignore
        sigma_e_bin[i, b] = sigma_e
        sigma_x_bin[i, b] = sigma_x
    
    return sigma_e_bin, sigma_x_bin

# ------------------------- Load 8B spectrum ------------------------- #
def load_8B_spectrum_data(csv_path='lambda.csv'):
    df = pd.read_csv(csv_path)
    return df['energy'].values, df['lambda'].values, df['error1'].values, df['error2'].values

# ------------------------- Solar propagation with decoherence ------------------------- #
@njit
def N_e(r):
    """Solar electron density profile (Bahcall model)"""
    if r <= R_sol:
        return 245 * N_A * np.exp(-r * 10.45 / R_sol) * one_by_cm3_to_eV3
    else:
        return 0.0

@njit
def A_cc(n_e, E):
    """Matter potential"""
    return 2 * np.sqrt(2) * G_F * n_e * E

@njit
def del_m2_eff(del_m2, theta, A_cc_val):
    """Effective mass difference in matter"""
    return np.sqrt((del_m2 * np.cos(2*theta) - A_cc_val)**2 + (del_m2 * np.sin(2*theta))**2)

@njit
def theta_eff(del_m2, theta, A_cc_val):
    """Effective mixing angle in matter"""
    num = del_m2 * np.sin(2*theta)
    den = del_m2 * np.cos(2*theta) - A_cc_val
    return 0.5 * np.arctan2(num, den)

@njit
def k(N, beta, tau):
    """Decoherence parameter"""
    return tau * (G_F * beta * N)**2

@njit
def A(E, N, del_m2_m, theta_m):
    """Hamiltonian component A"""
    return -del_m2_m * np.cos(2*theta_m) / (4*E) + G_F * N / np.sqrt(2)

@njit
def B(E, del_m2_m, theta_m):
    """Hamiltonian component B"""
    return del_m2_m * np.sin(2*theta_m) / (4*E)

def solar_solver(E, beta, tau, del_m2, theta, n_slabs=100000, r_i=0.0, r_f=1.0):
    """Solve neutrino propagation through Sun using matrix exponential method"""
    E_ev = E * 1e6  # Convert MeV to eV
    dx = (r_f - r_i) * R_earth / n_slabs
    r_vals = np.linspace(r_i + dx/(2*R_earth), r_f - dx/(2*R_earth), n_slabs)
    psi = np.array([1.0, 0.0, 0.0])  # Initial state: pure electron neutrino
    Pee = np.zeros(n_slabs)

    for i in range(n_slabs):
        r = r_vals[i] * R_earth
        N = N_e(r)
        A_m = A_cc(N, E_ev)
        del_m2_m = del_m2_eff(del_m2, theta, A_m)
        theta_m = theta_eff(del_m2, theta, A_m)
        k_r = k(N, beta, tau)
        
        # Build Hamiltonian
        A_r = A(E_ev, N, del_m2_m, theta_m)
        B_E = B(E_ev, del_m2_m, theta_m)
        M = np.array([
            [0.0,   0.0,   B_E],
            [0.0,   k_r,   -A_r],
            [-B_E,  A_r,   k_r]
        ])
        
        # Matrix exponential propagation
        U = expm(-2 * M * dx)
        psi = U @ psi
        Pee[i] = (psi[0] + 1) * 0.5  # Survival probability
    
    # Return survival probability at surface
    return Pee[-1]

# ------------------------- Build covariance matrix ------------------------- #
def build_covariance_matrix(sigma_stat, R_ref, s_sys_frac=0.03):
    V_stat = np.diag(sigma_stat**2)
    sys_vec = s_sys_frac * R_ref
    V_sys = np.outer(sys_vec, sys_vec)
    V = V_stat + V_sys
    return V

# ------------------------- Theoretical rate calculation ------------------------- #
def calculate_theoretical_rates_full(dm2, tan2theta, beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin):
    theta = np.arctan(np.sqrt(tan2theta))
    lam = lambda_vals.copy()
    if np.trapz(lam, E_nu_vals) > 0:
        lam = lam / np.trapz(lam, E_nu_vals)

    R_pred = np.zeros(n_bins)
    
    # Precompute survival probabilities for all neutrino energies
    Pee_vals = np.zeros(len(E_nu_vals))
    for i, E_nu in enumerate(tqdm(E_nu_vals, desc="Computing survival probabilities")):
        Pee_vals[i] = solar_solver(E_nu, beta, tau, dm2, theta, n_slabs=50000)  # Reduced for speed

    # Compute rates for each bin
    for b in range(n_bins):
        for i, E_nu in enumerate(E_nu_vals):
            sigma_eff = sigma_e_bin[i, b] * Pee_vals[i] + sigma_x_bin[i, b] * (1.0 - Pee_vals[i])
            R_pred[b] += lam[i] * sigma_eff

    dE = E_nu_vals[1] - E_nu_vals[0]
    R_pred *= dE

    norm = np.sum(R_exp) / np.sum(R_pred) if np.sum(R_pred) > 0 else 1.0
    return R_pred * norm

# ------------------------- χ² function with covariance ------------------------- #
def chi2_cov(R_pred, R_obs, V_inv):
    diff = R_pred - R_obs
    return float(diff.T @ V_inv @ diff)

# ------------------------- Grid scan function ------------------------- #
def run_grid_scan_full(beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin,
                       dm2_vals, tan2_vals, V_inv):
    chi2_grid = np.zeros((len(dm2_vals), len(tan2_vals)))
    
    def compute_chi2(i, j):
        dm2 = dm2_vals[i]
        tan2 = tan2_vals[j]
        R_pred = calculate_theoretical_rates_full(dm2, tan2, beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin)
        return i, j, chi2_cov(R_pred, R_exp, V_inv)
    
    # Parallel grid scan
    args_list = [(i, j) for i in range(len(dm2_vals)) for j in range(len(tan2_vals))]
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_chi2)(i, j) for i, j in tqdm(args_list, desc=f"Grid scan β={beta}")
    )
    
    # Fill chi2 grid
    for i, j, chi2_val in results: # type: ignore
        chi2_grid[i, j] = chi2_val
    
    chi2_min = np.min(chi2_grid)
    delta_chi2 = chi2_grid - chi2_min
    best_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    best_dm2 = dm2_vals[best_idx[0]]
    best_tan2 = tan2_vals[best_idx[1]]
    return chi2_grid, delta_chi2, (best_dm2, best_tan2, chi2_min)

# ------------------------- Plotting functions ------------------------- #
def plot_overlaid_contours_full(results, dm2_vals, tan2_vals):
    levels = [4.61, 5.99, 9.21, 11.83]
    plt.figure(figsize=(9,7))
    
    for beta, delta_chi2, best_fit, color in results:
        X, Y = np.meshgrid(tan2_vals, dm2_vals)
        cs = plt.contour(X, Y, delta_chi2, levels=levels, colors=color, alpha=0.8, linewidths=1.2)
        best_dm2, best_tan2, chi2min = best_fit
        plt.plot(best_tan2, best_dm2, marker='o', color=color, label=f'β={beta}, χ²={chi2min:.1f}')
    
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'$\tan^2\theta_{12}$')
    plt.ylabel(r'$\Delta m^2_{21}\,(eV^2)$')
    plt.title('Overlaid χ² Contours for different β (covariance χ²)')
    plt.legend()
    plt.grid(which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('chi2_contours.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectrum_comparison(best_params, beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin):
    dm2, tan2 = best_params
    R_pred = calculate_theoretical_rates_full(dm2, tan2, beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin)
    
    plt.figure(figsize=(8,5))
    plt.errorbar(bin_centers, R_exp, yerr=sigma_exp_stat, fmt='o', label='SK data', alpha=0.8)
    plt.step(bin_centers, R_pred, where='mid', label=f'Model β={beta}', linewidth=2)
    plt.xlabel('Recoil electron energy (MeV)')
    plt.ylabel('Events/kton/year')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f'Spectrum Comparison for β={beta}')
    plt.tight_layout()
    plt.savefig(f'spectrum_comparison_beta_{beta}.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_bestfit_table(table, filename='bestfit_table.csv'):
    df = pd.DataFrame(table)
    df.to_csv(filename, index=False)
    print(f"Saved best-fit table to {filename}")

# ------------------------- Main driver ------------------------- #
if __name__ == '__main__':
    print("Starting full neutrino oscillation analysis...")
    
    # Load 8B spectrum
    print("Loading 8B spectrum data...")
    E_lambda, lambda_vals_raw, err1, err2 = load_8B_spectrum_data('lambda.csv')
    
    # Interpolate lambda onto our E_nu grid
    n_nu = 60  # Reduced for speed
    E_nu_vals = np.linspace(0.1, 20.0, n_nu)
    lambda_interp = interp1d(E_lambda, lambda_vals_raw, bounds_error=False, fill_value=0.0)
    lambda_vals = lambda_interp(E_nu_vals)

    # Precompute smeared cross-sections
    print('Precomputing smeared cross sections (parallel computation)...')
    sigma_e_bin, sigma_x_bin = precompute_sigma_bins_parallel(E_nu_vals, bins_low, bins_high, n_Tp=30, n_Trec=20)

    # Build covariance matrix
    s_sys = 0.03
    V = build_covariance_matrix(sigma_exp_stat, R_exp, s_sys_frac=s_sys)
    V_inv = inv(V)

    # Grid ranges (reduced for speed)
    dm2_vals = np.logspace(-8, -3, 40)
    tan2_vals = np.logspace(-3, 1, 40)

    # Scan for several beta values
    tau_value = 10 * eV_to_1_by_km
    betas = [0.0, 0.05, 0.1]
    colors = ['b', 'g', 'r']

    results = []
    bestfit_table = []

    for beta, color in zip(betas, colors):
        print(f"\nRunning full grid scan for β={beta}...")
        chi2_grid, delta_chi2, best_fit = run_grid_scan_full(
            beta, tau_value, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin, 
            dm2_vals, tan2_vals, V_inv
        )
        
        results.append((beta, delta_chi2, best_fit, color))
        best_dm2, best_tan2, chi2min = best_fit
        bestfit_table.append({
            'beta': beta, 
            'dm2_best': best_dm2, 
            'tan2theta_best': best_tan2, 
            'chi2_min': chi2min
        })
        
        print(f"Best fit for β={beta}: Δm²={best_dm2:.3e}, tan²θ={best_tan2:.3f}, χ²={chi2min:.2f}")
        
        # Plot spectrum comparison
        plot_spectrum_comparison((best_dm2, best_tan2), beta, tau_value, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin)

    # Overlay contours
    plot_overlaid_contours_full(results, dm2_vals, tan2_vals)
    
    # Save best-fit table
    save_bestfit_table(bestfit_table)

    print('\nAnalysis complete!')