import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants as pc

# Constants
G_F = 1.1663787e-23  # eV^2
N_A = 6.02214076e23  # Avogadro's number
eV_to_1_by_m = 5.068e6
one_by_cm3_to_eV3 = (100/eV_to_1_by_m)**3
R_sol = 6.9634e8 * eV_to_1_by_m  # solar radius in eV^-1
R_earth = 1.496e11 * eV_to_1_by_m  # 1 AU in eV^-1

def N_e(r):
    """Baseline electron density in the sun"""
    if r <= R_sol:
        return 245*N_A*np.exp(-r*10.45/R_sol)*one_by_cm3_to_eV3
    else:
        return 0.0

def N_e_prime(r, beta, F):
    """Effective electron density: N_e' = (1 + beta * F) * N_e(r)
    where F is white noise with mean 0 and variance 1"""
    return (1 + beta * F) * N_e(r)

def effective_potential(r, beta, tau):
    """Effective matter potential with stochastic effects"""
    N = N_e(r)
    V_matter = G_F * N / np.sqrt(2)
    k_fluct = tau * (G_F * beta * N) ** 2
    return V_matter, k_fluct

# Create distance array from sun center to Earth
n_points = 10000
r_vals = np.linspace(0.0, 1.0, n_points) * R_earth

# Calculate baseline densities
densities = np.array([N_e(r) for r in r_vals])

# Generate white noise samples for demonstration (mean 0, variance 1)
np.random.seed(42)  # For reproducibility
F_samples = np.random.randn(n_points)

# Beta values to compare
beta_values = [0.0, 0.03, 0.05, 0.1]
tau = 10 * eV_to_1_by_m * 1000

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Baseline electron density profile
ax1 = axes[0, 0]
ax1.semilogy(r_vals/R_sol, densities/one_by_cm3_to_eV3/N_A, 'b-', linewidth=2.5, label=r'$N_e(r)$ (baseline)')
ax1.set_xlabel(r'Distance from Sun Center ($R_{\odot}$)', fontsize=12)
ax1.set_ylabel(r'Electron Density (g/cm$^3$)', fontsize=12)
ax1.set_title('Baseline Electron Density Profile (Log Scale)', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1.0)
ax1.grid(True, alpha=0.3, which='both')
ax1.axvline(1.0, color='red', linestyle='--', label=r'$R_{\odot}$', linewidth=1.5)
ax1.legend(fontsize=10)

# Plot 2: Effective density with fluctuations N_e' = (1 + beta*F)*N_e
ax2 = axes[0, 1]
colors = ['black', 'blue', 'green', 'orange']
ax2.semilogy(r_vals/R_sol, densities/one_by_cm3_to_eV3/N_A, 'k--', linewidth=2, 
         label=r'$N_e(r)$ (baseline)', alpha=0.7)
for beta, color in zip(beta_values[1:], colors[1:]):  # Skip beta=0
    N_prime = np.array([(1 + beta * F_samples[i]) * densities[i] for i in range(n_points)])
    label = r"$N_e' = (1+" + f"{beta}" + r"F)N_e$"
    ax2.semilogy(r_vals/R_sol, N_prime/one_by_cm3_to_eV3/N_A, color=color, 
             linewidth=1.5, alpha=0.6, label=label)

ax2.set_xlabel(r'Distance from Sun Center ($R_{\odot}$)', fontsize=12)
ax2.set_ylabel(r'Electron Density (g/cm$^3$)', fontsize=12)
ax2.set_title(r"Effective Density: $N_e' = (1 + \beta F) N_e(r)$ (Log Scale)", fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1.0)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=9)
ax2.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

# Plot 3: Fluctuation factor (1 + beta*F)
ax3 = axes[0, 2]
for beta, color in zip(beta_values[1:], colors[1:]):
    fluctuation_factor = 1 + beta * F_samples
    label = r'$\beta = $' + f'{beta}'
    ax3.plot(r_vals/R_sol, fluctuation_factor, color=color, linewidth=1.5, 
             alpha=0.7, label=label)

ax3.axhline(1.0, color='black', linestyle='--', linewidth=2, label='No fluctuation')
ax3.set_xlabel(r'Distance from Sun Center ($R_{\odot}$)', fontsize=12)
ax3.set_ylabel(r'Fluctuation Factor $(1 + \beta F)$', fontsize=12)
ax3.set_title(r'White Noise Fluctuations: $(1 + \beta F)$', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 1.0)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

# Plot 4: Matter potential (baseline vs with fluctuations)
ax4 = axes[1, 0]
V_matter = G_F * densities / np.sqrt(2)
ax4.plot(r_vals/R_sol, V_matter*1e14, 'k--', linewidth=2.5, 
         label=r'$V_{matter}$ (baseline)', alpha=0.7)
for beta, color in zip(beta_values[1:], colors[1:]):
    N_prime = np.array([(1 + beta * F_samples[i]) * densities[i] for i in range(n_points)])
    V_prime = G_F * N_prime / np.sqrt(2)
    label = r'$\beta = $' + f'{beta}'
    ax4.plot(r_vals/R_sol, V_prime*1e14, color=color, linewidth=1.5, alpha=0.6, label=label)

ax4.set_xlabel(r'Distance from Sun Center ($R_{\odot}$)', fontsize=12)
ax4.set_ylabel(r'Matter Potential ($\times 10^{-14}$ eV)', fontsize=12)
ax4.set_title(r"Effective Matter Potential with Fluctuations", fontsize=14, fontweight='bold')
ax4.set_xlim(0, 1.0)
ax4.grid(True, alpha=0.3)
ax4.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.legend(fontsize=9)

# Plot 5: Fluctuation term k for different beta
ax5 = axes[1, 1]
for beta, color in zip(beta_values, colors):
    k_vals = tau * (G_F * beta * densities) ** 2
    label = r'$\beta = $' + f'{beta}'
    if beta == 0.0:
        label = r'No fluctuations ($\beta = 0$)'
    ax5.plot(r_vals/R_sol, k_vals*1e14, color=color, linewidth=2, label=label)

ax5.set_xlabel(r'Distance from Sun Center ($R_{\odot}$)', fontsize=12)
ax5.set_ylabel(r'Fluctuation Term $k$ ($\times 10^{-14}$ eV)', fontsize=12)
ax5.set_title(r'Stochastic Fluctuation Term: $k = \tau(G_F \beta N_e)^2$', fontsize=14, fontweight='bold')
ax5.set_xlim(0, 1.0)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)
ax5.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

# Plot 6: Ratio of fluctuation to matter potential
ax6 = axes[1, 2]
V_matter_safe = V_matter.copy()
V_matter_safe[V_matter_safe == 0] = 1e-100  # Avoid division by zero
for beta, color in zip(beta_values[1:], colors[1:]):  # Skip beta=0
    k_vals = tau * (G_F * beta * densities) ** 2
    ratio = k_vals / V_matter_safe
    label = r'$\beta = $' + f'{beta}'
    ax6.plot(r_vals/R_sol, ratio, color=color, linewidth=2, label=label)

ax6.set_xlabel(r'Distance from Sun Center ($R_{\odot}$)', fontsize=12)
ax6.set_ylabel(r'$k / V_{matter}$', fontsize=12)
ax6.set_title(r'Ratio: Fluctuation / Matter Potential', fontsize=14, fontweight='bold')
ax6.set_xlim(0, 1.0)
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3, which='both')
ax6.legend(fontsize=10)
ax6.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig('solar_density_profile_with_fluctuations.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional plot: Zoom into solar interior
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

# Focus on solar interior (0 to 1 R_sol)
mask_sun = r_vals <= R_sol

# Plot 1: Baseline density in solar interior
ax_1 = axes2[0, 0]
ax_1.semilogy(r_vals[mask_sun]/R_sol, densities[mask_sun]/one_by_cm3_to_eV3/N_A, 'b-', linewidth=2.5)
ax_1.set_xlabel(r'Radial Distance ($R_{\odot}$)', fontsize=13)
ax_1.set_ylabel(r'Electron Density (g/cm$^3$)', fontsize=13)
ax_1.set_title('Solar Interior Density Profile (Log Scale)', fontsize=14, fontweight='bold')
ax_1.grid(True, alpha=0.3, which='both')
ax_1.set_yscale('log')
ax_1.set_xlim(0, 1.0)

# Plot 2: Effective density with fluctuations (zoomed)
ax_2 = axes2[0, 1]
ax_2.semilogy(r_vals[mask_sun]/R_sol, densities[mask_sun]/one_by_cm3_to_eV3/N_A, 
              'k--', linewidth=2.5, label=r'$N_e(r)$ (baseline)', alpha=0.8)
for beta, color in zip(beta_values[1:], colors[1:]):
    N_prime = np.array([(1 + beta * F_samples[i]) * densities[i] for i in range(n_points)])
    label = r'$\beta = $' + f'{beta}'
    ax_2.semilogy(r_vals[mask_sun]/R_sol, N_prime[mask_sun]/one_by_cm3_to_eV3/N_A, 
                  color=color, linewidth=1.8, alpha=0.7, label=label)

ax_2.set_xlabel(r'Radial Distance ($R_{\odot}$)', fontsize=13)
ax_2.set_ylabel(r'Effective Density (g/cm$^3$)', fontsize=13)
ax_2.set_title(r"$N_e' = (1 + \beta F) N_e(r)$ in Solar Interior", fontsize=14, fontweight='bold')
ax_2.grid(True, alpha=0.3, which='both')
ax_2.legend(fontsize=10)
ax_2.set_xlim(0, 1.0)
ax_2.set_yscale('log')

# Plot 3: Fluctuation terms in solar interior
ax_3 = axes2[1, 0]
for beta, color in zip(beta_values, colors):
    k_vals = tau * (G_F * beta * densities[mask_sun]) ** 2
    label = r'$\beta = $' + f'{beta}'
    if beta == 0.0:
        label = r'No fluctuations ($\beta = 0$)'
    ax_3.semilogy(r_vals[mask_sun]/R_sol, k_vals*1e14, color=color, linewidth=2.5, label=label)

ax_3.set_xlabel(r'Radial Distance ($R_{\odot}$)', fontsize=13)
ax_3.set_ylabel(r'Fluctuation Term $k$ ($\times 10^{-14}$ eV)', fontsize=13)
ax_3.set_title('Fluctuation Term in Solar Interior (Log Scale)', fontsize=14, fontweight='bold')
ax_3.grid(True, alpha=0.3, which='both')
ax_3.legend(fontsize=11)
ax_3.set_xlim(0, 1.0)
ax_3.set_yscale('log')

# Plot 4: Distribution of fluctuation factor (1 + beta*F)
ax_4 = axes2[1, 1]
for beta, color in zip(beta_values[1:], colors[1:]):
    fluctuation_factor = 1 + beta * F_samples
    ax_4.hist(fluctuation_factor, bins=50, alpha=0.5, color=color, 
              label=r'$\beta = $' + f'{beta}', density=True)

ax_4.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Mean (no fluctuation)')
ax_4.set_xlabel(r'Fluctuation Factor $(1 + \beta F)$', fontsize=13)
ax_4.set_ylabel('Probability Density', fontsize=13)
ax_4.set_title(r'Distribution of $(1 + \beta F)$ (White Noise)', fontsize=14, fontweight='bold')
ax_4.grid(True, alpha=0.3)
ax_4.legend(fontsize=11)
ax_4.set_yscale('log')
plt.tight_layout()
plt.savefig('solar_interior_density_fluctuations.png', dpi=300, bbox_inches='tight')
plt.show()
