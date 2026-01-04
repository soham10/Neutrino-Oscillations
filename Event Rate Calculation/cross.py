import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import scipy.special as sp
from numba import njit
from multiprocessing import Pool, cpu_count

hep = pd.read_csv('hep.csv')
Enu_vals = hep['energy'].values  # Convert to numpy array for faster access

# ---- Constants and cross section functions ----
G_F = 1.1663787e-11  # MeV^-2
m_e = 0.510998950    # MeV
MeVsqtocmsq = (197.3*1e-13)**2  # Conversion factor MeV^-2 to cm^2
fsc = 1/137.036  # Fine structure constant (alpha)
rho_NC = 1.0126  # ±0.0016
sin_2_W = 0.2317  # sin^2(theta_W)


@njit
def I(T):
    """Radiative correction term I(T)"""
    x = np.sqrt(1 + 2*m_e/T)
    return (1/6) * (1/3 + (3 - x**2) * (0.5*x*np.log((x+1)/(x-1)) - 1))


def kappa(T, flavor):
    """Radiative correction term kappa"""
    if flavor == 'e':
        return 0.9791 + 0.0097 * I(T)
    elif flavor == 'mu' or flavor == 'mu/tau':
        return 0.9970 - 0.00037 * I(T)
    else:
        raise ValueError("flavor must be 'e' or 'mu'/'mu/tau'")


def g_L(T, flavor):
    """Left-handed coupling with radiative corrections"""
    if flavor == 'e':
        return rho_NC * (0.5 - kappa(T, flavor) * sin_2_W) - 1
    elif flavor == 'mu' or flavor == 'mu/tau':
        return rho_NC * (0.5 - kappa(T, flavor) * sin_2_W)
    else:
        raise ValueError("flavor must be 'e' or 'mu'/'mu/tau'")


def g_R(T, flavor):
    """Right-handed coupling with radiative corrections"""
    return -rho_NC * kappa(T, flavor) * sin_2_W


def fminus(z, q):
    """QED correction term f_-"""
    T = z * q
    E = T + m_e
    l = np.sqrt(E**2 - m_e**2)
    beta = l / E
    
    term1 = (E/l * np.log((E + l)/m_e) - 1) * (2 * np.log(1 - z - m_e/(E + l)) - np.log(1 - z) - 0.5*np.log(z) - 5/12)
    term2 = 0.5 * (-sp.spence(1 - z) + sp.spence(1 - beta)) - 0.5*np.log(1-z)**2 - (11/12 + z/2)*np.log(1-z)
    term3 = z*(np.log(z) + 0.5*np.log(2*q/m_e)) - (31/18 + 1/12 * np.log(z))*beta - 11*z/12 + z**2/24
    
    return term1 + term2 + term3


def fplus(z, q):
    """QED correction term f_+"""
    T = z * q
    E = T + m_e
    l = np.sqrt(E**2 - m_e**2)
    beta = l / E
    
    term1 = (E/l * np.log((E + l)/m_e) - 1) * ((1-z)**2 * (2*np.log(1 - z - m_e/(E + l)) - np.log(1 - z) - np.log(z)/2 - 2/3) - 0.5*(z**2 * np.log(z) + 1 - z))
    term2 = -0.5*(1-z)**2 * (np.log(1-z)**2 + beta * (-sp.spence(1-(1-z)) - np.log(z)*np.log(1-z)))
    term3 = np.log(1-z) * (0.5*z**2 * np.log(z) + (1-z)/3 * (2*z - 0.5)) + 0.5*z**2 * sp.spence(1-(1-z)) - z*(1-2*z)/3 * np.log(z) - z*(1-z)/6
    term4 = -beta/12 * (np.log(z) + (1-z)*(115 - 109*z)/6)
    
    return term1 + term2 + term3 + term4


def fpm(z, q):
    """QED correction term f_+-"""
    T = z * q
    E = T + m_e
    l = np.sqrt(E**2 - m_e**2)
    
    return 2 * (E/l * np.log((E+l)/m_e) - 1) * np.log(1 - z - m_e/(E+l))


def dSigma_dT_corrections(flavor, E_nu, T_e):
    """
    Differential cross section with QED and radiative corrections
    Based on arXiv:astro-ph/9502003
    
    Parameters:
        flavor: 'e' for electron neutrino, 'mu' for muon/tau neutrino
        E_nu: Incident neutrino energy (MeV)
        T_e: Electron recoil kinetic energy (MeV)
    
    Returns:
        Cross section in cm^2/MeV
    """
    z_val = T_e / E_nu    
    # Get couplings
    gL = g_L(T_e, flavor)
    gR = g_R(T_e, flavor)

    # Calculate QED corrections
    f_minus = fminus(z_val, E_nu)
    f_plus = fplus(z_val, E_nu)
    f_pm = fpm(z_val, E_nu)

    # Differential cross section with corrections
    prefactor = 2 * G_F**2 * m_e / np.pi

    term1 = gL**2 * (1 + fsc/np.pi * f_minus)
    term2 = gR**2 * (1 - z_val)**2 + gR**2*fsc/np.pi * f_plus
    term3 = -gR * gL * m_e * z_val / E_nu * (1 + fsc/np.pi * f_pm)

    xsc = prefactor * (term1 + term2 + term3) * MeVsqtocmsq

    return xsc


def dsigma_dT(Ev_MeV, T_MeV, is_nu_e): 
    """Differential cross section with radiative corrections (cm^2 / MeV)"""
    flavor = 'e' if is_nu_e else 'mu'
    return dSigma_dT_corrections(flavor, Ev_MeV, T_MeV)


@njit
def TMax(Enu):
    """Maximum kinetic energy the electron can have for a given neutrino energy"""
    return 2*Enu**2 / (2*Enu + m_e)


# SK detector response

@njit
def s_SK(T):
    """Width of the Gaussian response function in MeV"""
    return -0.123 + 0.376 * np.sqrt(T + m_e) + 0.0349 * (T + m_e)


@njit
def R_SK(Te_meas, T_true):
    """Detector response function: probability of measuring Te_meas given true energy T_true"""
    s = s_SK(T_true)
    norm = 1 / (np.sqrt(2 * np.pi) * s)
    return norm * np.exp(-0.5 * ((Te_meas - T_true) / s) ** 2)

def compute_convolved_cross_section(Enu, Te_meas_center, bin_width, is_nu_e):
    """
    Compute detector-convolved cross section with double integration
    Implements: σ(E_ν) = ∫[T_min to T_max] dT ∫[0 to T_max(E_ν)] R(T, T') × (dσ/dT')(E_ν, T') dT'
    
    Parameters:
        Enu: Neutrino energy (MeV)
        Te_meas_center: Center of measured Te bin (MeV)
        bin_width: Width of Te bin (MeV)
        is_nu_e: Boolean, True for electron neutrino, False for muon/tau
    
    Returns:
        Convolved cross section (cm^2)
    """
    T_max = TMax(Enu)
    
    # Integration bounds for T (measured energy bin)
    T_meas_min = Te_meas_center - bin_width/2
    T_meas_max = Te_meas_center + bin_width/2
    
    def outer_integrand(T_meas):
        """
        Integrand for outer integral over T (measured energy)
        Returns: ∫[0 to T_max] R(T, T') × (dσ/dT')(E_ν, T') dT'
        """
        def inner_integrand(Tprime):
            """
            Integrand for inner integral over T' (true energy)
            Returns: R(T, T') × (dσ/dT')(E_ν, T')
            """
            # Get the differential cross section at true energy T'
            dsigma = dsigma_dT(Enu, Tprime, is_nu_e)
            # Get detector response for measuring T given true energy T'
            response = R_SK(T_meas, Tprime)
            return response * dsigma
        
        # Inner integral: over true energy T' from 0 to T_max
        inner_result, _ = quad(inner_integrand, 0, T_max, epsabs=1e-10, epsrel=1e-8)
        return inner_result
    
    # Outer integral: over measured energy T from T_min to T_max
    result, _ = quad(outer_integrand, T_meas_min, T_meas_max, epsabs=1e-10, epsrel=1e-8)
    
    return result

def process_Te_bin(args):
    """Worker function for parallel processing of a single Te bin"""
    Te_center, bin_width, Enu_array = args
    print(f"Processing Te = {Te_center} MeV")
    
    results = []
    for Enu in Enu_array:
        sigma_e = compute_convolved_cross_section(Enu, Te_center, bin_width, is_nu_e=True)
        sigma_x = compute_convolved_cross_section(Enu, Te_center, bin_width, is_nu_e=False)
        results.append([Enu, Te_center, sigma_e, sigma_x])
    
    return results

# ---------- Calculate convolved cross sections ----------
output_total = []

# Load Te bin centers from plot-data.csv  
plot_data = pd.read_csv("plot-data.csv")
Te_bin_centers = plot_data['energy'].values
bin_widths = plot_data['bin_width'].values

print("Computing detector-convolved cross sections...")
print(f"Using {len(Enu_vals)} neutrino energies from hep.csv")
print(f"Using {len(Te_bin_centers)} Te bins from plot-data.csv")
print(f"Using {cpu_count()} CPU cores for parallel processing")

# Prepare arguments for parallel processing
args_list = [(Te_center, bin_width, Enu_vals) 
             for Te_center, bin_width in zip(Te_bin_centers, bin_widths)]

# Use multiprocessing to parallelize over Te bins
with Pool(processes=cpu_count()) as pool:
    results_nested = pool.map(process_Te_bin, args_list)

# Flatten results
output_total = []
for result_list in results_nested:
    output_total.extend(result_list)

output_total = np.array(output_total)

# Plot total cross section results
plt.figure(figsize=(8, 6))
for Te in Te_bin_centers[20:23]:
    mask = output_total[:, 1] == Te
    if np.any(mask):
        plt.plot(output_total[mask, 0], output_total[mask, 2], 'o-', label=f"Te = {Te} MeV (e)")

plt.xlabel("Neutrino Energy (MeV)")
plt.ylabel("Cross Section")
plt.title("Detector-Convolved Cross Section vs E_nu for different Te bins")
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# Save results
np.savetxt("SKIII/sigma_SKIII.csv", output_total, 
           delimiter=",", header="E_nu,Te_center,sigma_e,sigma_x", comments='')
