import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########################
## Constants
MeVsqtocmsq = (197.3*1e-13)**2  # Conversion factor MeV^-2 to cm^2

#####################
### Parameters
GF = 1.16*10**(-11)  ## Fermi constant in MeV^-2
me = 0.511  # Electron mass in MeV
sinsqthW = 0.23  # Weak mixing angle

###########
## SK Cross section ###########
def dSigmadTeNUe_SK(Te, Enu):
    ep = sinsqthW   # g2
    em = 0.5 + sinsqthW  # g1
    return (2/np.pi)*me*GF**2* ( em**2 + ep**2 * (1- Te/Enu)**2 - em* ep* me* Te/ (Enu)**2)*MeVsqtocmsq  ## in cm^2

def dSigmadTeNUmu_SK(Te, Enu):
    g1 = -0.5 + sinsqthW
    g2 = sinsqthW
    return (2/np.pi)*me*GF**2* ( g1**2 + g2**2 * (1- Te/Enu)**2 - g1* g2* me* Te/ (Enu)**2)*MeVsqtocmsq  ## in cm^2    

def TMax(Enu):
    """Maximum kinetic energy that electron can have for given neutrino energy"""
    return Enu / (1 + me/(2*Enu))

def integrate_cross_section(Enu, T_center, bin_width, is_nu_e=True, num_points=200):
    """Faithfully integrate the SK differential cross section over a Te bin."""
    half_width = 0.5 * bin_width
    T_min = max(T_center - half_width, 0.0)
    T_max = min(T_center + half_width, TMax(Enu))
    if T_max <= T_min:
        return 0.0
    Te_vals = np.linspace(T_min, T_max, num_points)
    if is_nu_e:
        dsigma_vals = dSigmadTeNUe_SK(Te_vals, Enu)
    else:
        dsigma_vals = dSigmadTeNUmu_SK(Te_vals, Enu)
    return np.trapz(dsigma_vals, Te_vals)

# ---------- Calculate cross sections ----------
output_total = []

# Load Te bin centers from plot-data.csv
plot_data = pd.read_csv("plot-data.csv")
Te_bin_centers = plot_data['energy'].values
sigma_energies = plot_data['bin_width'].values

# Load lambda (neutrino energies) from lambda.csv
lambda_df = pd.read_csv('lambda.csv')
Enu_vals = lambda_df['energy'].values

print("Computing SK cross sections without detector response...")
print(f"Using {len(Enu_vals)} neutrino energies from lambda.csv")
print(f"Using {len(Te_bin_centers)} Te bins from plot-data.csv")

# Compute cross sections for each Te and Enu combination
for i_te, (Te_center, sigma_energy) in enumerate(zip(Te_bin_centers, sigma_energies)):
    bin_width = sigma_energy
    print(f"Processing Te bin {i_te+1}/{len(Te_bin_centers)}: Te = {Te_center} MeV, bin_width = {bin_width} MeV")
    
    for Enu in Enu_vals:
        sigma_e = integrate_cross_section(Enu, Te_center, bin_width, is_nu_e=True)
        sigma_x = integrate_cross_section(Enu, Te_center, bin_width, is_nu_e=False)
        output_total.append([Enu, Te_center, sigma_e, sigma_x])

output_total = np.array(output_total)

sigma_df = pd.read_csv('sigma.csv')
multi_te_df = sigma_df.groupby('Te_center').filter(lambda g: len(g) > 1)
multi_te_values = multi_te_df['Te_center'].unique()

te_common = sorted(
    Te for Te in multi_te_values
    if np.isclose(Te_bin_centers, Te).any()
)
if not te_common:
    raise ValueError("No common Te bins found between σ data and computed grid.")
TARGET_TE = te_common[18]  # adjust if you want a different Te value

plt.figure(figsize=(8, 6))
sigma_mask = np.isclose(multi_te_df['Te_center'], TARGET_TE)
te_rows = multi_te_df.loc[sigma_mask]
output_mask = np.isclose(output_total[:, 1], TARGET_TE)
plt.plot(te_rows['E_nu'], te_rows['sigma_e'], 'x--', label=f"Te_soham = {TARGET_TE:.2f} MeV")
plt.plot(output_total[output_mask, 0], output_total[output_mask, 2],
         'o-', label=f"Te_MS = {TARGET_TE:.2f} MeV (νe)")        

plt.xlabel("Neutrino Energy (MeV)")
plt.ylabel("Cross Section")
plt.title("SK differential cross section vs $E_\\nu$")
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()