import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.constants import physical_constants as pc
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

# Constants
G_F = pc['Fermi coupling constant'][0] * 1e-18  # eV^2
N_A = pc['Avogadro constant'][0]  # mol^-1
eV_to_1_by_m = pc['electron volt-inverse meter relationship'][0]
eV_to_1_by_km = eV_to_1_by_m * 1e3
one_by_cm3_to_eV3 = (1.973e-5) ** 3
R_sol = 6.9634e8 * eV_to_1_by_m   # solar radius in eV^-1

# Vacuum Parameters
del_m2_v = 7.1e-5     
theta_v = np.arcsin(0.5)                                

@njit
# Matter Potential
def A_cc(n_e, E):
    return 2*np.sqrt(2)*G_F*n_e*E

@njit
# Effective mass squared difference in matter
def del_m2_eff(del_m2, theta, A_cc):
    x = np.sqrt((del_m2*np.cos(2*theta) - A_cc)**2 + (del_m2*np.sin(2*theta))** 2)
    return x

@njit
# Effective mixing angle
def theta_eff(del_m2, theta, A_cc):
    num = del_m2*np.sin(2*theta)
    den = del_m2*np.cos(2*theta) - A_cc
    return 0.5*np.arctan2(num,den)

@njit
def N_e(r):
    return 245*N_A*np.exp(-r*10.45/R_sol)*one_by_cm3_to_eV3

@njit
def k(N, beta, tau):
    return tau*(G_F*beta*N)** 2

@njit
def A(E, N, del_m2_m, theta_m):
    return -del_m2_m*np.cos(2*theta_m)/(4*E) + G_F*N/np.sqrt(2)

@njit
def B(E, del_m2_m, theta_m):
    return del_m2_m*np.sin(2*theta_m)/(4*E)

def solar_solver(E, beta, tau, del_m2, theta, n_slabs=100000, r_i=0.0, r_f=1.0):
    E = E*1e6    
    dx = (r_f - r_i) * R_sol / n_slabs
    r_vals = np.linspace(r_i + dx/(2*R_sol), r_f - dx/(2*R_sol), n_slabs)
    psi = np.array([0.5, 0.0, 0.0])
    Pee = np.zeros(n_slabs)

    for i in range(n_slabs):
        r = r_vals[i]*R_sol
        N = N_e(r)
        A_m = A_cc(N, E)
        del_m2_m = del_m2_eff(del_m2, theta, A_m)
        theta_m = theta_eff(del_m2, theta, A_m)
        k_r = k(N, beta, tau)
        A_r = A(E, N, del_m2_m, theta_m)
        B_E = B(E, del_m2_m, theta_m)
        M = np.array([
            [0.0,   0.0,   B_E],
            [0.0,   k_r,   -A_r],
            [-B_E,  A_r,   k_r]])
        U = expm(-2 * M * dx)
        psi = U @ psi
        Pee[i] = psi[0] + 0.5
    return r_vals, Pee

def avg_Pee(beta, E_vals, tau, n_jobs=-1):
    def probability(E):
        r_frac, Pee_profile = solar_solver(E,beta,tau, del_m2_v, theta_v)
        mask = (r_frac>=0.9)
        return np.mean(Pee_profile[mask])
    
    avg_Pee = Parallel(n_jobs=n_jobs)(delayed(probability)(E) for E in tqdm(E_vals, desc=f'β = {beta}'))
    return np.array(avg_Pee)

def final_prob(beta_values, E_vals, tau, n_jobs=-1):
    results = {}
    for beta in beta_values:
        results[beta] = avg_Pee(beta, E_vals, tau, n_jobs)
    return results

# Implementation
beta_vals = [0,0.02,0.04,0.06]
E_vals = np.logspace(-2, np.log10(30), 200)  # 0.01 to 30 MeV
tau = 10*eV_to_1_by_km
results = final_prob(beta_vals, E_vals, tau, n_jobs=-1)

# Plot
colors = plt.cm.seismic(np.linspace(0, 1, len(results)))
linestyles = ['-', '--', ':', '-.']
plt.figure(figsize=(15, 8))
for i, (beta, avg_Pee) in enumerate(results.items()):
    color = colors[i]
    linestyle = linestyles[i % len(linestyles)]
    plt.plot(E_vals, avg_Pee, label=f'β = {beta}', color=color, linestyle=linestyle)

plt.xscale('log')
plt.xlabel("Energy (MeV)", fontsize=14)
plt.ylabel(r"$P_{ee}$", fontsize=14)
plt.title(r"Electron Neutrino Survival for Varying $\beta$ with Stochastic Fluctuations", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()