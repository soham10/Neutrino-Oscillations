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
fiducial_mass = 2.15e10 # g
N_e_tgt = 10*fiducial_mass*N_A/18  # number of target electrons
phi_B0 = 2.32e6  # 8B neutrino flux in cm^-2 s^-1
phi_hep = 2.1e3  # cm^-2 s^-1

# Load detector-corrected cross sections σ(E_ν)
sigmas = np.loadtxt('Finall/sigma_vs_E_nu_SK.csv', delimiter=',', skiprows=1)
E_nu_vals = sigmas[:, 0]   # Neutrino energy (MeV)
Te = sigmas[:, 1]      # Recoil electron energy (MeV)
sigma_e = sigmas[:, 2]    # ν_e cross section (cm^2)
sigma_x = sigmas[:, 3]    # ν_x cross section (cm^2)

# Load 8B spectrum λ_B(E)
lambda_df = pd.read_csv('Finall/lambda.csv')
lambda_E = np.array(lambda_df['energy'].values, dtype=float)
lambda_val_B = np.array(lambda_df['lambda'].values, dtype=float)

hep_df = pd.read_csv('Finall/hep.csv')
hep_E = np.array(hep_df['energy'].values, dtype=float)
lambda_val_hep = np.array(hep_df['lambda'].values, dtype=float)

E_unique = np.unique(E_nu_vals)
E_unique.sort()
if len(E_unique) > 1:
    dE = np.zeros_like(E_unique)
    dE[0] = 0.5 * (E_unique[1] - E_unique[0])
    dE[-1] = 0.5 * (E_unique[-1] - E_unique[-2])
    for i in range(1, len(E_unique) - 1):
        dE[i] = 0.5 * (E_unique[i+1] - E_unique[i-1])

# Interpolate both spectra on the cross-section energy grid
lambda_interp_B = np.interp(E_unique, lambda_E, lambda_val_B, left=0.0, right=0.0)
lambda_interp_hep = np.interp(E_unique, hep_E, lambda_val_hep, left=0.0, right=0.0)
lambda_interp = (phi_B0 * lambda_interp_B + phi_hep * lambda_interp_hep) / (phi_B0 + phi_hep)
total_flux = phi_B0 + phi_hep
weights = lambda_interp * dE 
weight_map = {float(E_unique[i]): float(weights[i]) for i in range(len(E_unique))}

phi_B = total_flux

try:
    combined_lambda_path = 'Finall/lambda_combined.csv'
    comb_df = pd.DataFrame({
        'E': E_unique,
        'lambda_eff': lambda_interp,
        'lambda_B': lambda_interp_B,
        'lambda_hep': np.interp(E_unique, hep_E, lambda_val_hep, left=0.0, right=0.0) if hep_df is not None else np.zeros_like(E_unique)
    })
    comb_df.to_csv(combined_lambda_path, index=False)
    print(f"Saved combined effective spectrum to {combined_lambda_path}")
except Exception as e:
    print(f"Could not save combined lambda file: {e}")

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
    mask = (r_frac >= 0.95)
    return np.mean(Pee_profile[mask])

# Oscillation parameters
beta = 0.0 # Turbulence Parameter
tau = 10*eV_to_1_by_m*1000  # Collective oscillation parameter
del_m2 = 7.1e-5  # eV2
theta = np.arctan(np.sqrt(0.46))    # radians

# ---------------------------
# Parallelized rate section
# ---------------------------
def compute_rate_for_tuple(args):
    """
    args: tuple (E, Te, se, sx, w)
    returns: [Te, rate_times_weight, weight]
    """
    E, Te, se, sx, w = args
    Pee_val = avg_Pee(E, beta, tau, del_m2, theta)
    rate = phi_B * (se * Pee_val + sx * (1 - Pee_val))
    rate *= N_e_tgt * 24 * 3600/0.5  # Events/day/21.5kt/0.5 MeV for this (E, Te) pair
    return [Te, rate * w, w]

if __name__ == '__main__':
    n_jobs = cpu_count()
    print(f"Using {n_jobs} worker processes for parallel rate computation.")
    
    # Create a dictionary to store cross sections by Te value
    sigma_dict = {}
    
    # Group all data by Te bin
    for i in range(len(Te)):
        te_val = Te[i]
        e_nu = E_nu_vals[i]
        se = sigma_e[i]
        sx = sigma_x[i]
        
        if te_val not in sigma_dict:
            sigma_dict[te_val] = []
        
        sigma_dict[te_val].append((e_nu, se, sx))
    
    # For each Te bin, prepare tasks across all contributing neutrino energies
    tasks = []
    for te in sorted(sigma_dict.keys()):
        for e_nu, se, sx in sigma_dict[te]:
            w = weight_map.get(float(e_nu), 0.0)
            tasks.append((e_nu, te, se, sx, w))
    
    results = []
    # Use multiprocessing.Pool + tqdm to show progress
    with Pool(processes=n_jobs) as pool:
        for out in tqdm(pool.imap_unordered(compute_rate_for_tuple, tasks), total=len(tasks), desc="Rates"):
            results.append(out)

    # Process results: compute λ(E)*ΔE-weighted average per Te
    te_weighted_sum = {}
    te_weight_sum = {}
    for te, rate_w, w in results:
        if te not in te_weighted_sum:
            te_weighted_sum[te] = 0.0
            te_weight_sum[te] = 0.0
        if w > 0:
            te_weighted_sum[te] += rate_w
            te_weight_sum[te] += w

    te_values = sorted(te_weighted_sum.keys())
    rates = np.array([
        [te, (te_weighted_sum[te] / te_weight_sum[te]) if te_weight_sum[te] > 0 else 0.0]
        for te in te_values
    ])
    
    np.savetxt('Finall/theoretical_rate_SK.csv', rates, delimiter=',', header='Te,EventRate/day/21.5kt/0.5MeV', comments='')
    print("Saved theoretical rate to 'theoretical_rate_SK.csv'")
    
    # Load experimental data for comparison
    exp_data = np.loadtxt('Finall/experimental_rate_SK.csv', delimiter=',', skiprows=1)
    exp_te = exp_data[:, 0]
    exp_rate = exp_data[:, 1]
    exp_err = exp_data[:, 2]
    
    plt.figure(figsize=(10,6))
    plt.scatter(rates[:,0], rates[:,1], label='Theoretical Rate', color='blue', s=40)
    plt.errorbar(exp_te, exp_rate, yerr=exp_err, fmt='o', label='SK Experimental Data', color='red', alpha=0.7, markersize=5)
    plt.xlabel('Electron Recoil Energy Te (MeV)')
    plt.ylabel('Events/day/21.5kt/0.5MeV')
    plt.title('Event Rate vs Electron Recoil Energy')
    
    # Add textual information about parameters
    plt.text(0.02, 0.02, 
             f"Oscillation parameters:\nβ={beta}, τ={tau/eV_to_1_by_m/1000:.1f} km, Δm²={del_m2:.1e} eV², tan²θ={np.tan(theta)**2:.2f}", 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("Finall/event_rate_vs_Te.png", dpi=300)
    plt.show()