# ================== Libraries ==================
import numpy as np
from scipy.integrate import odeint, quad
from scipy.constants import physical_constants as pc
import matplotlib.pyplot as plt

# ================== Plotting Style ==================
plt.style.use(['classic', "seaborn-v0_8-dark"])
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 15,
    'legend.fontsize': 15,
    'axes.labelsize': 18,
    'legend.frameon': True,
    'lines.linewidth': 1.5
})

# ================== Physical Constants ==================
G_F = pc['Fermi coupling constant'][0]*1e-18  # eV²
N_A = pc['Avogadro constant'][0]
eV_to_1_by_m = pc['electron volt-inverse meter relationship'][0]
eV_to_1_by_km = eV_to_1_by_m * 1e3
one_by_cm3_to_eV3 = (1.973e-5)**3

# ================== Oscillation Parameters ==================
del_m2_31 = 2.5e-3
del_m2_21 = 7.53e-5
theta_31 = np.arcsin(np.sqrt(2.18e-2))
theta_21 = np.arcsin(np.sqrt(0.307))
R0 = 40  # km

# ================== Vectors ==================
def L_vec(n_dim): return np.array([0, 0, 1]) if n_dim == 3 else ValueError("Invalid dim")
def B_vec(n_dim, theta): return np.array([-np.sin(2*theta), 0, np.cos(2*theta)]) if n_dim == 3 else ValueError("Invalid dim")

# ================== Matter & Potential ==================
def SN_density_profile(r, t):
    """
    Reference: https://arxiv.org/abs/hep-ph/0304056
    """
    r = np.atleast_1d(r)  # Ensure r is always an array
    rho0 = 1e14*r**(-2.4)  # g/cm³

    if t < 1:
        return rho0

    # Shockwave parameters
    epsilon = 10
    r_s0 = -4.6e3     # km
    v_s = 11.3e3      # km/s
    a_s = 0.2e3       # km/s²
    r_s = r_s0 + v_s*t + 0.5*a_s*t**2  # Shockwave position

    # Apply shockwave-modified profile where r <= r_s
    rho = rho0.copy()
    mask = r <= r_s
    with np.errstate(invalid='ignore'):
        a = (0.28 - 0.69*np.log(r[mask]))*(np.arcsin(1 - r[mask]/r_s)**1.1)
        rho[mask] = rho0[mask] * epsilon*np.exp(a)  # Modified to avoid in-place operation

    return rho

# Electron density profile
def lambda_sn(r, option, n=N_A, t=1.0):
    m_n = pc['neutron mass'][0]*1e3 # g
    Y_e = 0.5  # electron fraction

    if option == "no":
        n_eff = 0
    elif option == "const":
        n_eff = n
    elif option == "SN":
        r = np.atleast_1d(r)  # Ensure r is array
        n_eff = (Y_e/m_n)*SN_density_profile(r, t)  # electrons per cm³
    else:
        raise ValueError("Invalid option for matter density profile. Choose 'no', 'const', or 'SN'.")

    return np.sqrt(2)*G_F*n_eff*one_by_cm3_to_eV3

# Neutrino-Neutrino Potential
def mu_r(r, opt, mu0=0):
    if opt == "SN":
        if isinstance(r, (np.ndarray, list)):
            return np.where(r < R0, mu0, mu0*(R0/np.array(r))**4)
        else:
            return mu0 if r < R0 else mu0*(R0/r)**4
    
    elif opt == "const":
        return mu0
    
    else:
        raise ValueError("Invalid option chosen")

def radial_velocity(u, r):
    u = np.atleast_1d(u)
    r = np.atleast_1d(r)
    return np.sqrt(np.clip(1 - u * (R0/r)**2, 0, 1))

def integrate_vec(f, r):
    try:
        return np.array([quad(lambda u: f(u, r)[i], 0, 1, epsrel=1e-5)[0] for i in range(3)])
    except:
        return np.zeros(3)

def get_Pr(Pu, u_vals):
    return integrate_vec(lambda u, _: np.array([np.interp(u, u_vals, Pu[:, i]) for i in range(3)]), 0)

# ================== Main Comparison Plot ==================
N_u = 100
u_vals = np.linspace(0, 1, N_u)
r_vals = np.linspace(0.1, 100, 500)
P0 = np.tile([0, 0, 1], (N_u, 1))
y0 = np.concatenate([P0.flatten(), P0.flatten()])

hierarchies = ['NH', 'IH']
colors = ['tab:blue', 'tab:red']
labels = [r'Normal Hierarchy', r'Inverted Hierarchy']

plt.figure(figsize=(8, 5))

for h, color, label in zip(hierarchies, colors, labels):
    sign = 1 if h == 'NH' else -1

    def rhs_hier(y_flat, r, E, N_u, u_vals, n_dim=3, theta=theta_21, mu0=1e5, lam_opt="SN", mu_opt="SN"):
        P = y_flat[:3*N_u].reshape(N_u, 3)
        P_bar = y_flat[3*N_u:].reshape(N_u, 3)
        v_vals = radial_velocity(u_vals, r)

        B = B_vec(n_dim, theta)
        L = L_vec(n_dim)
        lam = lambda_sn(r, option=lam_opt)
        mu = mu_r(r, opt=mu_opt, mu0=mu0)
        vac = sign * del_m2_21 / (2 * E)

        def P_interp(u, r_):
            return np.array([np.interp(u, u_vals, P[:, i]) for i in range(3)])
        
        def Pbar_interp(u, r_):
            return np.array([np.interp(u, u_vals, P_bar[:, i]) for i in range(3)])
        
        P_r = integrate_vec(P_interp, r)
        Pbar_r = integrate_vec(Pbar_interp, r)
        J = integrate_vec(lambda u, r_: (P_interp(u, r_) - Pbar_interp(u, r_)) / radial_velocity(u, r_), r)

        dP, dP_bar = np.zeros_like(P), np.zeros_like(P_bar)

        for i, u in enumerate(u_vals):
            vu = v_vals[i]
            dP[i] = (np.cross(B, P[i]) * vac + np.cross(L, P[i]) * lam +
                     mu * R0**2 / r**2 * (np.cross(J, P[i] / vu) - np.cross(P_r - Pbar_r, P[i]))) / vu

            dP_bar[i] = (-np.cross(B, P_bar[i]) * vac + np.cross(L, P_bar[i]) * lam +
                         mu * R0**2 / r**2 * (np.cross(J, P_bar[i] / vu) - np.cross(P_r - Pbar_r, P_bar[i]))) / vu

        return np.concatenate([dP.flatten(), dP_bar.flatten()])

    sol = odeint(rhs_hier, y0, r_vals, args=(10.0, N_u, u_vals))
    P_all = sol[:, :3*N_u].reshape(-1, N_u, 3)
    P_surv = np.array([0.5 * (get_Pr(P_all[i], u_vals)[2] + 1) for i in range(len(r_vals))])

    plt.plot(r_vals, P_surv, label=label, color=color)

plt.xlabel("Radius $r$ (km)")
plt.ylabel("Survival Probability $P_{ee}$")
plt.title("Neutrino Survival Probability vs Radius for NH and IH")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()