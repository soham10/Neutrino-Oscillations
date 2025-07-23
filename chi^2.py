import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.constants import physical_constants as pc
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

# ------------------------- Constants ------------------------- #
G_F = pc['Fermi coupling constant'][0] * 1e-18  # eV^2
N_A = pc['Avogadro constant'][0]                # mol^-1
eV_to_1_by_m = pc['electron volt-inverse meter relationship'][0]
eV_to_1_by_km = eV_to_1_by_m * 1e3
one_by_cm3_to_eV3 = (1.973e-5) ** 3
R_sol = 6.9634e8 * eV_to_1_by_m   # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1

# ------------------------- Vacuum Parameters ------------------------- #
del_m2_v = 7.1e-5     
theta_v = np.arcsin(0.5)                                

@njit
def A_cc(n_e, E):
    return 2*np.sqrt(2)*G_F*n_e*E

@njit
def del_m2_eff(del_m2, theta, A_cc_val):
    return np.sqrt((del_m2*np.cos(2*theta) - A_cc_val)**2 + (del_m2*np.sin(2*theta))**2)

@njit
def theta_eff(del_m2, theta, A_cc_val):
    num = del_m2*np.sin(2*theta)
    den = del_m2*np.cos(2*theta) - A_cc_val
    return 0.5*np.arctan2(num, den)

@njit
def N_e(r):
    if r <= R_sol:
        return 245*N_A*np.exp(-r*10.45/R_sol)*one_by_cm3_to_eV3
    else:
        return 0.0

@njit
def k(N, beta, tau):
    return tau*(G_F*beta*N)**2

@njit
def A(E, N, del_m2_m, theta_m):
    return -del_m2_m*np.cos(2*theta_m)/(4*E) + G_F*N/np.sqrt(2)

@njit
def B(E, del_m2_m, theta_m):
    return del_m2_m*np.sin(2*theta_m)/(4*E)

def solar_solver(E, beta, tau, del_m2, theta, n_slabs=10000, r_i=0.0, r_f=None):
    if r_f is None:
        r_f = R_earth / R_sol
    E = E*1e6
    dx = (r_f - r_i) * R_sol / n_slabs
    r_vals = np.linspace(r_i + dx/(2*R_sol), r_f - dx/(2*R_sol), n_slabs)
    psi = np.array([1.0, 0.0, 0.0])
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
        Pee[i] = (psi[0] + 1)*0.5
    return r_vals, Pee


def solar_solver_point(E, beta, tau, del_m2, theta):
    r_frac, Pee_profile = solar_solver(E, beta, tau, del_m2, theta)
    mask = (r_frac >= 0.9)
    return np.mean(Pee_profile[mask])

def chi_sq(exp, th, sigma):
    return np.sum(((exp - th) / sigma) ** 2)

def compute_chi2_grid_parallel(dm2_grid, tan2theta_grid, E_vals, data_probability, data_sigma, beta, tau):
    def calc_chi2_point(dm2, tan2theta):
        theta = np.arctan(np.sqrt(tan2theta))
        probs = [solar_solver_point(E, beta, tau, dm2, theta) for E in E_vals]
        return chi_sq(data_probability, probs, data_sigma)

    chi2_grid = Parallel(n_jobs=-1)(
        delayed(calc_chi2_point)(dm2, tan2theta) 
        for dm2 in tqdm(dm2_grid, desc="Δm² Grid") 
        for tan2theta in tan2theta_grid
    )

    chi2_grid_arr = np.array(chi2_grid).reshape(len(dm2_grid), len(tan2theta_grid))
    return chi2_grid_arr

# ------------------------- Load Experimental Data ------------------------- #
data = pd.read_csv('prob.csv')
E_vals = data['E'].values
probs = data['E'].values
tau = 10 * eV_to_1_by_km
beta = 0

data['sigma'] = np.std(probs) + 0.02*data['Day']


# ------------------------- Chi2 Grid Search ------------------------- #
dm2_grid = np.logspace(-8, -3, 10)
tan2theta_grid = np.logspace(-4, 1, 10)

chi2_grid = compute_chi2_grid_parallel(
    dm2_grid,
    tan2theta_grid,
    E_vals,
    data['Day'].values,
    data['sigma'].values,
    beta,
    tau
)
# ------------------------- Best-Fit Parameters ------------------------- #
min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
min_chi2 = chi2_grid[min_idx]
best_dm2 = dm2_grid[min_idx[0]]
best_tan2theta = tan2theta_grid[min_idx[1]]

print(f"\nMinimum χ²: {min_chi2:.3f}")
print(f"Best-fit Δm²: {best_dm2:.3e} eV²")
print(f"Best-fit tan²θ: {best_tan2theta:.3f}")

# ------------------------- Plot Contour with Confidence Levels ------------------------- #
plt.figure(figsize=(8, 6))

# Filled chi2 contour
contour = plt.contourf(tan2theta_grid, dm2_grid, chi2_grid.T, levels=50, cmap='viridis')

# Confidence level contour lines
delta_chi2 = [2.30, 6.18, 11.83]  # 1σ, 2σ, 3σ
confidence_levels = [min_chi2 + d for d in delta_chi2]
lines = plt.contour(tan2theta_grid, dm2_grid, chi2_grid.T,
                    levels=confidence_levels,
                    colors=['white', 'cyan', 'red'],
                    linestyles=['--', '-.', ':'])

# Annotate contours
fmt = {}
for l, s in zip(lines.levels, ['1σ', '2σ', '3σ']):
    fmt[l] = s
plt.clabel(lines, inline=True, fontsize=10, fmt=fmt)

# Mark best-fit point
plt.scatter(best_tan2theta, best_dm2, color='black', marker='x', label='Best Fit')

# Log scales and labels
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\tan^2 \theta$')
plt.ylabel(r'$\Delta m^2$ (eV$^2$)')
plt.title(r'$\chi^2$ Contour Map with Confidence Intervals')
plt.colorbar(contour, label=r'$\chi^2$')
plt.legend()
plt.tight_layout()
plt.show()
