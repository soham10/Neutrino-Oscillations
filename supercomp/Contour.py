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

def solar_solver(E, beta, tau, del_m2, theta, n_slabs=100000, r_i=0.0, r_f=1.0):
    E = E*1e6
    dx = (r_f - r_i) * R_earth / n_slabs
    r_vals = np.linspace(r_i + dx/(2*R_earth), r_f - dx/(2*R_earth), n_slabs)
    psi = np.array([1.0, 0.0, 0.0])
    Pee = np.zeros(n_slabs)

    for i in range(n_slabs):
        r = r_vals[i]*R_earth
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


def avg_Pee(E, beta, tau, del_m2, theta):
    r_frac, Pee_profile = solar_solver(E, beta, tau, del_m2, theta)
    mask = (r_frac >= 0.95)
    return np.mean(Pee_profile[mask])

def chi_sq(exp, th, sigma):
    return np.sum(((exp - th) / sigma) ** 2)

def compute_chi2(dm2_grid, tan2theta_grid, E_vals, data_probability, data_sigma, beta, tau):
    param_list = [(dm2, tan2theta) for dm2 in dm2_grid for tan2theta in tan2theta_grid]
    def calc_chi2_point(dm2, tan2theta):
        theta = np.arctan(np.sqrt(tan2theta))
        probs = [avg_Pee(E, beta, tau, dm2, theta) for E in E_vals]
        return chi_sq(data_probability, probs, data_sigma)
    results = Parallel(n_jobs=-1)(
        delayed(calc_chi2_point)(dm2, tan2theta) for dm2, tan2theta in tqdm(param_list, desc="Calculating Grid", dynamic_ncols=True, unit='pt')
    )
    return np.array(results).reshape(len(dm2_grid), len(tan2theta_grid))

# ------------------------- Load Experimental Data ------------------------- #
data = pd.read_csv('supercomp/prob.csv')
E_vals = data['E'].values
probs = data['Day'].values
tau = 10*eV_to_1_by_km
sigma = np.std(np.asarray(probs, dtype=float))
data['sigma'] = sigma* np.ones_like(probs)

# ------------------------- Chi2 Grid Search ------------------------- #
dm2_grid = np.logspace(-5, -3, 10)
tan2theta_grid = np.logspace(-1, 0, 10)

# ------------------------- Beta Variation Setup ------------------------- #
beta_list = [0, 0.04,0.08]  # Add more β values as desired
colors = ['cyan', 'green', 'blue', 'orange', 'purple']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot', 'dotted']

min_info = []  
plt.figure(figsize=(10, 7))

for i, beta in enumerate(beta_list):
    print(f"\nProcessing for β = {beta}")
    chi2_grid = compute_chi2(
        dm2_grid,
        tan2theta_grid,
        E_vals,
        data['Day'].values,
        data['sigma'].values,
        beta,
        tau
    )

    # Best-fit extraction
    min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    min_chi2 = chi2_grid[min_idx]
    best_dm2 = dm2_grid[min_idx[0]]
    best_tan2theta = tan2theta_grid[min_idx[1]]

    min_info.append((beta, min_chi2, best_dm2, best_tan2theta))

    # Filled χ² contours: fill up to 1σ (Δχ² = 2.30) with the color for this beta
    cf = plt.contourf(
        tan2theta_grid, dm2_grid, chi2_grid,
        levels=[min_chi2, min_chi2 + 2.30],
        colors=[colors[i % len(colors)]],
        alpha=0.4
    )

    # Plot confidence contours: Δχ² = 2.30 (1σ), 6.18 (2σ), 11.83 (3σ)
    confidence_levels = [min_chi2 + 2.30, min_chi2 + 6.18, min_chi2 + 11.83]
    for j, delta in enumerate(confidence_levels):
        plt.contour(
            tan2theta_grid, dm2_grid, chi2_grid,
            levels=[delta],
            colors=[colors[i % len(colors)]],
            linestyles=[linestyles[j % len(linestyles)]],
            linewidths=2
        )

    # Mark best-fit point
    plt.scatter(best_tan2theta, best_dm2, marker='x', s=70, color=colors[i % len(colors)],
                label=f'Best Fit β={beta:.3f}')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\tan^2 \theta$')
plt.ylabel(r'$\Delta m^2$ (eV$^2$)')
plt.title(r'$\chi^2$ minimization contours for varying $\beta$')
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.colorbar(cf, label=r'$\chi^2$')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('chi2_contours.png')

# Save the results
results_df = pd.DataFrame(min_info, columns=['beta', 'min_chi2', 'best_dm2', 'best_tan2theta'])
results_df.to_csv('chi2_results.csv', index=False)
print("Chi-squared results saved to 'chi2_results.csv'.")