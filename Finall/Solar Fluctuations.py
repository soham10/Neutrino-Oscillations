import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.constants import physical_constants as pc
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (100/eV_to_1_by_m)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1

# Vacuum Parameters
del_m2_v = 7.1e-5  # eV2
theta_v = np.arctan(np.sqrt(0.46))                                 

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

def solar_solver(E, beta, tau, del_m2, theta, n_slabs=40000, r_i=0.0, r_f=1.0):
    E = E * 1e6  # MeV to eV
    dx = (r_f - r_i) * R_earth / n_slabs
    r_vals = np.linspace(r_i, r_f, n_slabs)
    psi = np.array([1.0, 0.0, 0.0])
    Pee_profile = np.zeros(n_slabs)
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
        Pee_profile[i] = (psi[0] + 1)*0.5
    return r_vals, Pee_profile

def avg_Pee(beta, E_vals, tau, n_jobs=-1):
    def probability(E):
        r_frac, Pee_profile = solar_solver(E,beta,tau, del_m2_v, theta_v)
        mask = (r_frac>=0.95)
        return np.mean(Pee_profile[mask])
    
    avg_Pee = Parallel(n_jobs=n_jobs)(delayed(probability)(E) for E in tqdm(E_vals, desc=f'β = {beta}'))
    return np.array(avg_Pee)

# Implementation
beta_values = [0.0, 0.03, 0.05, 0.1]
E_vals = np.logspace(np.log10(0.01), np.log10(100), 100)
tau = 10*eV_to_1_by_m*1000
mass = del_m2_v/E_vals
# Plot
plt.figure(figsize=(15, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']

for idx, beta in enumerate(beta_values):
    results = avg_Pee(beta, E_vals, tau)
    results_df = pd.DataFrame({
        'energy': E_vals,
        'results': results
    })
    results_df.to_csv(f'Theory Probability[{beta}].csv', index = False)
    
    plt.plot(mass, results, label=f'β = {beta}', color=colors[idx])

plt.xlabel("Energy (MeV)", fontsize=14)
plt.ylabel(r"$P_{ee}$", fontsize=14)
plt.title(r"Electron Neutrino Survival for Varying $\beta$ with Stochastic Fluctuations", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.show()