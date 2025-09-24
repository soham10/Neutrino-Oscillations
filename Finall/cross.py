import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

# -------------- Lambda (8B spectrum) --------------
lambda_df = pd.read_csv('Finall/lambda.csv')
lambda_interp = interp1d(lambda_df['energy'].values, lambda_df['lambda'].values, 
                         bounds_error=False, fill_value=0.0)

# ---- Constants and cross section functions ----
G_F = 1.1663787e-23  # eV^2
m_e = 0.510998950    # MeV
eV_to_cm = 1.23981e-4

def couplings(is_nu_e):
    if is_nu_e:
        g1 = 0.73
        g2 = 0.23
    else:
        g1 = -0.27
        g2 = 0.23
    return g1, g2

def dsigma_dTe(Ev_MeV, Te_MeV, is_nu_e):
    g1, g2 = couplings(is_nu_e)
    sigma0 = 88.06e-46  # cm^2
    prefac = sigma0 / m_e
    z = Te_MeV / Ev_MeV
    val = prefac * (g1**2 + g2**2 * (1 - z)**2 - g1 * g2 * (m_e/Ev_MeV) * z)
    return val

def EnuMin(Te):
    return 0.5 * (Te + np.sqrt(Te**2 + 2*m_e*Te))

# SK detector response function
def s_SK(T_prime):
    """Width of the Gaussian response function in MeV"""
    return 0.47 * np.sqrt(T_prime)

def R_SK(T, T_prime):
    s = s_SK(T_prime)
    if s <= 0:
        return 0.0
    return (1 / (np.sqrt(2*np.pi) * s)) * np.exp(-0.5 * ((T - T_prime) / s)**2)

# ------------ Bin structure -----------------
Te_bins = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0, 16.0])
bin_centers = (Te_bins[:-1] + Te_bins[1:]) / 2
bin_width = 0.5  # MeV

# Function to compute the convolved cross section for a given neutrino energy and Te bin
def compute_convolved_sigma(Enu, Te_bin, is_nu_e=True):
    T_min = Te_bin - 0.25  # lower bin edge
    T_max = Te_bin + 0.25  # upper bin edge
    
    # Maximum kinetic energy that electron can have (kinematic limit)
    T_prime_max = Enu / (1 + m_e/(2*Enu))
    
    if T_prime_max <= 0:
        return 0.0
    
    # Integrand for the double integration
    def integrand(T_prime, T):
        dsigma_dT_prime = dsigma_dTe(Enu, T_prime, is_nu_e)
        response = R_SK(T, T_prime)
        return response * dsigma_dT_prime
    
    # Perform the double integration over T and T'
    result, _ = dblquad( # type: ignore
        integrand, 
        T_min, T_max,                # T integration limits (bin edges)
        lambda x: 0.0,                # T' lower limit 
        lambda x: min(T_prime_max, 20.0)  # T' upper limit
    )
    
    return result

# ---------- Calculate convolved cross sections for each energy bin ----------
output = []
Enu_vals = np.linspace(5.0, 20.0, 30)  # Neutrino energy points

print("Computing detector-convolved cross sections...")
for Te in bin_centers:
    for Enu in Enu_vals:
        if Enu >= EnuMin(Te - 3*s_SK(Te)):  # Allow 3 sigma fluctuation down in true energy
            try:
                sigma_e = compute_convolved_sigma(Enu, Te, is_nu_e=True)
                sigma_x = compute_convolved_sigma(Enu, Te, is_nu_e=False)
                output.append([Enu, Te, sigma_e, sigma_x])
            except Exception as e:
                print(f"Error at Enu={Enu}, Te={Te}: {e}")
                output.append([Enu, Te, 0.0, 0.0])
        else:
            output.append([Enu, Te, 0.0, 0.0])

output = np.array(output)
np.savetxt("Finall/sigma_vs_E_nu_SK.csv", output, delimiter=",", header="E_nu,Te,sigma_e,sigma_x")
