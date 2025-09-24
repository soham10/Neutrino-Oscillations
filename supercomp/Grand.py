"""
Full analysis script: grid χ² scan in (Δm², tan²θ) including
- full 8B spectrum convolution,
- detector energy resolution (SK),
- smeared cross sections computed per reconstructed-energy bin,
- correlated systematic via covariance matrix,
- overlayed contours for multiple β values (same plot),
- save best-fit table and produce spectrum comparison plots.

ASSUMPTIONS / REQUIREMENTS:
- A file `lambda.csv` must exist in the working directory with columns:
  energy (MeV), lambda (arbitrary units), error1, error2
- This script is moderately heavy numerically. For speed-ups you can:
  * reduce n_nu_points or grid resolution, or
  * precompute and cache sigma_in_bin values.

Caveats:
- I use a single fully-correlated fractional systematic `s_sys` (default 0.03 = 3%).
- Absolute normalization is treated as a free scaling (I normalize predicted total to data total) —
  if you prefer a different treatment (float normalization as pull parameter), we can implement that.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import inv
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
one_by_cm3_to_eV3 = (1.234e-4) ** 3

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

# ------------------------- Detector response (SK) ------------------------- #
def s_SK(Tp_MeV: float) -> float:
    """SK energy resolution width (MeV)"""
    return 0.47 * np.sqrt(max(Tp_MeV, 0.0))

def R_SK(T_rec_MeV: float, Tp_MeV: float) -> float:
    s = s_SK(Tp_MeV)
    if s <= 0:
        return 0.0
    return np.exp(-0.5*((T_rec_MeV - Tp_MeV)/s)**2) / (np.sqrt(2*np.pi)*s)

# ------------------------- Weak cross sections (SM) ------------------------- #
# Couplings (use your earlier values)
sin2_theta_v = 0.2317
rho = 1.0126
k_e = 0.9791
k_mu = 0.9970

def couplings(is_nu_e: bool):
    if is_nu_e:
        gL = rho * (0.5 - k_e * sin2_theta_v) - 1
        gR = -rho * k_e * sin2_theta_v
    else:
        gL = rho * (0.5 - k_mu * sin2_theta_v)
        gR = -rho * k_mu * sin2_theta_v
    return gL, gR

def dsigma_dT_SM(Tp_eV: float, Ev_eV: float, is_nu_e: bool) -> float:
    # Returns differential cross section in eV^-2
    z = Tp_eV / Ev_eV
    gL, gR = couplings(is_nu_e)
    term1 = gL**2
    term2 = gR**2 * (1 - z)**2
    term3 = -gL * gR * (m_e / Ev_eV) * z
    factor = 2 * (G_F**2) * m_e / np.pi
    dsig = factor * (term1 + term2 + term3)
    return max(dsig, 0.0)

# Convert eV^-2 to cm^2: (1 eV^-1 = 1.97e-5 cm)
eV_to_cm = 1.97e-5

# ------------------------- Smeared cross-section in a reconstructed bin -------------- #
def sigma_smeared_in_bin(Ev_MeV: float, is_nu_e: bool, Trec_low_MeV: float, Trec_high_MeV: float,
                          n_Tp=60, n_Trec=40):
    Ev_eV = Ev_MeV * 1e6
    # kinematic max Tp (MeV) (approx): Tp_max = 2 E_nu^2 / (m_e + 2 E_nu) in MeV
    # but using non-relativistic expression: Tp_max = Ev / (1 + m_e/(2 Ev)) where Ev in MeV, m_e in MeV
    Tp_max_MeV = Ev_MeV / (1.0 + (m_e/1e6)/(2.0*Ev_MeV))  # approx, since m_e is in eV, convert
    if Tp_max_MeV <= 0:
        return 0.0

    Tp_vals = np.linspace(0.0, Tp_max_MeV, n_Tp)
    sigma_sum = 0.0
    for Tp in Tp_vals:
        # probability that reconstructed T falls in [Trec_low, Trec_high]
        Trec_vals = np.linspace(max(0.0, Trec_low_MeV - 5*s_SK(Tp)), min(Tp_max_MeV, Trec_high_MeV + 5*s_SK(Tp)), n_Trec)
        if len(Trec_vals) == 0:
            continue
        R_vals = np.array([R_SK(Trec, Tp) for Trec in Trec_vals])
        P_rec_in_bin = np.trapz(R_vals, Trec_vals)
        dsig = dsigma_dT_SM(Tp*1e6, Ev_eV, is_nu_e)
        sigma_sum += dsig * P_rec_in_bin
    # integrate over Tp
    sigma_sum *= (Tp_vals[1]-Tp_vals[0]) if len(Tp_vals) > 1 else 0.0
    # convert eV^-2 to cm^2
    return sigma_sum * (eV_to_cm**2)

# ------------------------- Load 8B spectrum ------------------------- #
def load_8B_spectrum_data(csv_path='lambda.csv'):
    df = pd.read_csv(csv_path)
    # Expect columns: energy (MeV), lambda, error1, error2
    return df['energy'].values, df['lambda'].values, df['error1'].values, df['error2'].values

# ------------------------- Solar propagation with decoherence (full 3x3) ---------- #
# To keep code readable I re-use your earlier Hamiltonian evolution with RK4

def k_decoh(N, beta, tau):
    return tau * (G_F * beta * N)**2


def rk4_step(psi, M, dx):
    k1 = -2.0j * (M @ psi)
    k2 = -2.0j * (M @ (psi + 0.5*dx*k1))
    k3 = -2.0j * (M @ (psi + 0.5*dx*k2))
    k4 = -2.0j * (M @ (psi + dx*k3))
    return psi + (dx/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def solar_solver_full(E_MeV, beta, tau, del_m2, theta, n_slabs=100000):
    # returns survival probability P_ee for a neutrino produced near solar center that exits the Sun
    E_ev = E_MeV * 1e6
    dx = R_sol / n_slabs
    # radial sampling from center to surface
    r_vals = np.linspace(dx/2.0, R_sol - dx/2.0, n_slabs)
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])

    for r in r_vals:
        N = N_e(r)
        A_m = A_cc(N, E_ev)
        dm2_m = del_m2_eff(del_m2, theta, A_m)
        th_m = theta_eff(del_m2, theta, A_m)
        k_r = k_decoh(N, beta, tau)

        A_r = -dm2_m * np.cos(2*th_m) / (4*E_ev) + G_F * N / np.sqrt(2)
        B_E = dm2_m * np.sin(2*th_m) / (4*E_ev)

        M = np.array([
            [0.0,   0.0,   B_E],
            [0.0,   k_r,   -A_r],
            [-B_E,  A_r,   k_r]
        ], dtype=np.complex128)

        psi = rk4_step(psi, M, dx)

    return np.abs(psi[0])**2

# (re)define N_e, A_cc, del_m2_eff, theta_eff with consistent units

def N_e(r):
    # r is in eV^-1 (we pass R_sol-scaled values), use Bahcall-like profile implemented earlier
    if r <= R_sol:
        return 245 * N_A * np.exp(- (r / R_sol) * 10.45 ) * one_by_cm3_to_eV3
    else:
        return 0.0


def A_cc(n_e, E):
    return 2 * np.sqrt(2) * G_F * n_e * E


def del_m2_eff(del_m2, theta, A_cc_val):
    return np.sqrt((del_m2 * np.cos(2*theta) - A_cc_val)**2 + (del_m2 * np.sin(2*theta))**2)


def theta_eff(del_m2, theta, A_cc_val):
    num = del_m2 * np.sin(2*theta)
    den = del_m2 * np.cos(2*theta) - A_cc_val
    return 0.5 * np.arctan2(num, den)

# ------------------------- Build covariance matrix ------------------------- #
def build_covariance_matrix(sigma_stat, R_ref, s_sys_frac=0.03):
    # sigma_stat: array of statistical errors (same units as R_ref)
    V_stat = np.diag(sigma_stat**2)
    # fully correlated systematic proportional to rates (use experimental rates as reference)
    sys_vec = s_sys_frac * R_ref
    V_sys = np.outer(sys_vec, sys_vec)
    V = V_stat + V_sys
    return V

# ------------------------- Theoretical rate calculation (full) ------------------------- #
def calculate_theoretical_rates_full(dm2, tan2theta, beta, tau,
                                    E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin):
    # dm2 in eV^2, tan2theta dimensionless
    theta = np.arctan(np.sqrt(tan2theta))
    # normalize lambda to unity over E_nu grid
    lam = lambda_vals.copy()
    if np.trapz(lam, E_nu_vals) > 0:
        lam = lam / np.trapz(lam, E_nu_vals)

    R_pred = np.zeros(n_bins)

    for i, E_nu in enumerate(E_nu_vals):
        P_ee_bins = np.zeros(n_bins)
        # For speed, compute a single Pee at the neutrino energy and use it for all bins
        Pee = solar_solver_full(E_nu, beta, tau, dm2, theta)
        # effective cross section in each bin (cm^2)
        sigma_e_bins = sigma_e_bin[i, :]
        sigma_x_bins = sigma_x_bin[i, :]
        sigma_eff_bins = sigma_e_bins * Pee + sigma_x_bins * (1.0 - Pee)
        R_pred += lam[i] * sigma_eff_bins

    # multiply by dE_nu
    dE = E_nu_vals[1]-E_nu_vals[0]
    R_pred *= dE

    # At this stage R_pred has units proportional to (flux * cross section). We don't have
    # absolute flux normalization, so normalize to the total observed rate (same as earlier approach).
    # This effectively treats the overall 8B flux as a free normalization.
    norm = np.sum(R_exp) / np.sum(R_pred) if np.sum(R_pred) > 0 else 1.0
    return R_pred * norm

# ------------------------- Precompute smeared cross sections for each E_nu and bin ----- #

def precompute_sigma_bins(E_nu_vals, bins_low, bins_high, n_Tp=60, n_Trec=40):
    nE = len(E_nu_vals)
    nB = len(bins_low)
    sigma_e_bin = np.zeros((nE, nB))
    sigma_x_bin = np.zeros((nE, nB))
    t0 = time.time()
    for i, E in enumerate(E_nu_vals):
        for b in range(nB):
            sigma_e_bin[i, b] = sigma_smeared_in_bin(E, True, bins_low[b], bins_high[b], n_Tp=n_Tp, n_Trec=n_Trec)
            sigma_x_bin[i, b] = sigma_smeared_in_bin(E, False, bins_low[b], bins_high[b], n_Tp=n_Tp, n_Trec=n_Trec)
        if (i+1) % max(1, nE//10) == 0:
            print(f"Precomputed {i+1}/{nE} E_nu points... elapsed {time.time()-t0:.1f}s")
    return sigma_e_bin, sigma_x_bin

# ------------------------- χ² function with covariance ------------------------- #

def chi2_cov(R_pred, R_obs, V_inv):
    diff = R_pred - R_obs
    return float(diff.T @ V_inv @ diff)

# ------------------------- Grid scan over (dm2, tan2theta) for a fixed beta,tau ---- #

def run_grid_scan_full(beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin,
                       dm2_vals, tan2_vals, V_inv):
    chi2_grid = np.zeros((len(dm2_vals), len(tan2_vals)))
    best = None
    t0 = time.time()
    for i, dm2 in enumerate(dm2_vals):
        for j, tan2 in enumerate(tan2_vals):
            R_pred = calculate_theoretical_rates_full(dm2, tan2, beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin)
            chi2_grid[i, j] = chi2_cov(R_pred, R_exp, V_inv)
        if (i+1) % max(1, len(dm2_vals)//5) == 0:
            print(f"Finished {i+1}/{len(dm2_vals)} dm2 rows. elapsed {time.time()-t0:.1f}s")
    chi2_min = np.min(chi2_grid)
    delta_chi2 = chi2_grid - chi2_min
    best_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    best_dm2 = dm2_vals[best_idx[0]]
    best_tan2 = tan2_vals[best_idx[1]]
    return chi2_grid, delta_chi2, (best_dm2, best_tan2, chi2_min)

# ------------------------- Plotting and output helpers ------------------------- #

def plot_overlaid_contours_full(results, dm2_vals, tan2_vals):
    # results: list of (beta, delta_chi2, best_fit, color)
    levels = [4.61, 5.99, 9.21, 11.83]
    plt.figure(figsize=(9,7))
    for beta, delta_chi2, best_fit, color in results:
        X, Y = np.meshgrid(tan2_vals, dm2_vals)
        cs = plt.contour(X, Y, delta_chi2, levels=levels, colors=color, alpha=0.8, linewidths=1.2)
        best_dm2, best_tan2, chi2min = best_fit
        plt.plot(best_tan2, best_dm2, marker='o', color=color, label=f'β={beta}, best')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'$\tan^2\theta_{12}$')
    plt.ylabel(r'$\Delta m^2_{21}\,(eV^2)$')
    plt.title('Overlaid χ² Contours for different β (covariance χ²)')
    plt.legend()
    plt.grid(which='both', alpha=0.3)
    plt.show()


def save_bestfit_table(table, filename='bestfit_table.csv'):
    df = pd.DataFrame(table)
    df.to_csv(filename, index=False)
    print(f"Saved best-fit table to {filename}")


def plot_spectrum_comparison(best_params, beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin):
    dm2, tan2 = best_params
    R_pred = calculate_theoretical_rates_full(dm2, tan2, beta, tau, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin)
    plt.figure(figsize=(8,5))
    plt.errorbar(bin_centers, R_exp, yerr=sigma_exp_stat, fmt='o', label='SK data')
    plt.step(bin_centers, R_pred, where='mid', label=f'Model β={beta}')
    plt.xlabel('Recoil electron energy (MeV)')
    plt.ylabel('Events/kton/year')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# ------------------------- Main driver -------------------------------------------------- #
if __name__ == '__main__':
    # Load 8B spectrum
    E_lambda, lambda_vals_raw, err1, err2 = load_8B_spectrum_data('lambda.csv')
    # Interpolate lambda onto our E_nu grid used for convolution
    n_nu = 80
    E_nu_vals = np.linspace(0.1, 20.0, n_nu)
    lambda_interp = interp1d(E_lambda, lambda_vals_raw, bounds_error=False, fill_value=0.0)
    lambda_vals = lambda_interp(E_nu_vals)

    # Prepare bin edges
    bins_low = sk_energy_bins[:-1]
    bins_high = sk_energy_bins[1:]

    # Precompute smeared cross-sections per (E_nu, bin)
    print('Precomputing smeared cross sections (this can take time)...')
    sigma_e_bin, sigma_x_bin = precompute_sigma_bins(E_nu_vals, bins_low, bins_high, n_Tp=80, n_Trec=60)

    # Build covariance matrix (3% fully correlated syst by default)
    s_sys = 0.03
    V = build_covariance_matrix(sigma_exp_stat, R_exp, s_sys_frac=s_sys)
    V_inv = inv(V)

    # Grid ranges (fine enough for publication may be increased)
    dm2_vals = np.logspace(-8, -3, 20)
    tan2_vals = np.logspace(-3, 1, 20)

    # Scan for several beta values and overlay
    tau_value = 10 * eV_to_1_by_km  # keep your choice (units eV^-1)
    betas = [0.0, 0.05, 0.1]
    colors = ['b','g','r']

    results = []
    bestfit_table = []

    for beta, color in zip(betas, colors):
        print(f"Running full grid scan for beta={beta} ...")
        chi2_grid, delta_chi2, best_fit = run_grid_scan_full(beta, tau_value, E_nu_vals, lambda_vals,
                                                           sigma_e_bin, sigma_x_bin, dm2_vals, tan2_vals, V_inv)
        results.append((beta, delta_chi2, best_fit, color))
        best_dm2, best_tan2, chi2min = best_fit
        bestfit_table.append({'beta': beta, 'dm2_best': best_dm2, 'tan2theta_best': best_tan2, 'chi2_min': chi2min})
        # Plot spectrum comparison for this best fit
        plot_spectrum_comparison((best_dm2, best_tan2), beta, tau_value, E_nu_vals, lambda_vals, sigma_e_bin, sigma_x_bin)

    # Overlay contours
    plot_overlaid_contours_full(results, dm2_vals, tan2_vals)
    # Save best-fit table
    save_bestfit_table(bestfit_table)

    print('\nDone.')
