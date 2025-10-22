import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
import pandas as pd

# -------------- Lambda (8B spectrum) --------------
lambda_df = pd.read_csv('Finall/lambda.csv')
lambda_interp = interp1d(lambda_df['energy'].values, lambda_df['lambda'].values, 
                         bounds_error=False, fill_value=0.0)

# ---- Constants and cross section functions ----
G_F = 1.1663787e-23  # eV^2
m_e = 0.510998950    # MeV
eV_to_cm = 1.23981e-4

def rho_NC_e():
    return 1.0126  # ±0.0016

def kappa_e(T):
    # T is the electron recoil kinetic energy
    x = np.sqrt(1 + 2*m_e/T)
    I_T = (1/6) * (1/3 + (3 - x**2) * (1/(2*x)) * np.log((x+1)/(x-1)) - 1)
    kappa = 0.9791 + 0.0097 * I_T + 0.0025
    return kappa

def gL_e(T):
    return rho_NC_e() * (0.5 - kappa_e(T) * 0.2317) - 1

def gR_e(T):
    return -rho_NC_e() * kappa_e(T) * 0.2317

def rho_NC_mu():
    return 1.0126

def kappa_mu(T):
    # Use the constant and the I(T) from above for muon scattering
    x = np.sqrt(1 + 2*m_e/T)
    I_T = (1/6) * (1/3 + (3 - x**2) * (1/(2*x)) * np.log((x+1)/(x-1)) - 1)
    kappa = 0.9970 - 0.00037 * I_T + 0.0025
    return kappa

def gL_mu(T):
    return rho_NC_mu() * (0.5 - kappa_mu(T) * 0.2317)

def gR_mu(T):
    return -rho_NC_mu() * kappa_mu(T) * 0.2317

def couplings_radiative(Te_MeV, is_nu_e):
    """Couplings with radiative corrections"""
    if is_nu_e:
        gL = gL_e(Te_MeV)
        gR = gR_e(Te_MeV)
    else:
        gL = gL_mu(Te_MeV)
        gR = gR_mu(Te_MeV)
    return gL, gR

def dsigma_dTe(Ev_MeV, Te_MeV, is_nu_e):
    """Differential cross section with radiative corrections"""
    gL, gR = couplings_radiative(Te_MeV, is_nu_e)
    sigma0 = 88.06e-46  # cm^2
    prefac = sigma0 / m_e
    z = Te_MeV / Ev_MeV
    val = prefac * (gL**2 + gR**2 * (1 - z)**2 - gL * gR * (m_e/Ev_MeV) * z)
    return val

def TMax(Enu):
    """Maximum kinetic energy that electron can have for given neutrino energy"""
    return Enu / (1 + m_e/(2*Enu))

# SK detector response function
def s_SK(T_prime):
    """Width of the Gaussian response function in MeV"""
    return 0.47 * np.sqrt(T_prime)

def R_SK(T, T_prime):
    s = s_SK(T_prime)
    if s <= 0:
        return 0.0
    return (1 / (np.sqrt(2*np.pi) * s)) * np.exp(-0.5 * ((T - T_prime) / s)**2)

# Function to compute the total convolved cross section for a given neutrino energy
def compute_total_convolved_sigma(Enu, T_center, bin_width=0.5, is_nu_e=True):
    """
    Compute total detector-convolved cross section for fixed Enu and Te bin
    T_center: center of the Te bin
    bin_width: width of Te bin (default 0.5 MeV)
    """
    T_min = T_center - bin_width/2  # T^i - 0.25
    T_max = T_center + bin_width/2  # T^i + 0.25
    
    # Maximum kinetic energy that electron can have (kinematic limit)
    T_prime_max = TMax(Enu)
    
    if T_prime_max <= 0 or T_max <= T_min:
        return 0.0
    
    # Double integrand for the integration over T and T'
    def double_integrand(T_prime, T):
        if T_prime <= 0 or T_prime > T_prime_max:
            return 0.0
        dsigma_dT_prime = dsigma_dTe(Enu, T_prime, is_nu_e)
        response = R_SK(T, T_prime)
        return response * dsigma_dT_prime
    
    # Perform double integration: outer over T, inner over T'
    result, _ = dblquad( # type: ignore
        double_integrand,
        T_min, T_max,  # T integration limits (Te bin)
        lambda x: 0.0,     # T' lower limit
        lambda x: T_prime_max  # T' upper limit (true energy)
    )
    
    return result

# ---------- Calculate convolved cross sections ----------
output_total = []

# Load Te bin centers from plot-data.csv
plot_data = pd.read_csv("Finall/plot-data.csv")
Te_bin_centers = plot_data['Recoil energy(MeV)'].values

# Use neutrino energies from lambda.csv file
Enu_vals = lambda_df['energy'].values

print("Computing detector-convolved cross sections...")
print(f"Using {len(Enu_vals)} neutrino energies from lambda.csv")
print(f"Using {len(Te_bin_centers)} Te bins from plot-data.csv")

for i_te, Te_center in enumerate(Te_bin_centers):
    print(f"Processing Te bin {i_te+1}/{len(Te_bin_centers)}: Te = {Te_center} MeV")
    
    for Enu in Enu_vals:
        sigma_e = compute_total_convolved_sigma(Enu, Te_center, bin_width=0.5, is_nu_e=True)
        sigma_x = compute_total_convolved_sigma(Enu, Te_center, bin_width=0.5, is_nu_e=False)
        output_total.append([Enu, Te_center, sigma_e, sigma_x])

output_total = np.array(output_total)

# Convert to units of 10^-46 cm^2
output_total[:, 2] *= 1e46
output_total[:, 3] *= 1e46

# Plot total cross section results
plt.figure(figsize=(8, 6))
for Te in Te_bin_centers[:5]:  # Plot first 5 Te bins as examples
    mask = output_total[:, 1] == Te
    if np.any(mask):
        plt.plot(output_total[mask, 0], output_total[mask, 2], 'o-', label=f"Te = {Te} MeV (e)")

plt.xlabel("Neutrino Energy (MeV)")
plt.ylabel("Cross Section (10^-46 cm^2)")
plt.title("Detector-Convolved Cross Section vs E_nu for different Te bins")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Save results
np.savetxt("Finall/sigma_vs_Enu_Te_bins_SK.csv", output_total, 
           delimiter=",", header="E_nu,Te_center,sigma_e,sigma_x", comments='')
