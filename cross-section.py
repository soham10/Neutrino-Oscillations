import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.constants import physical_constants as pc

# Physical constants
G_F = pc['Fermi coupling constant'][0] * 1e-18  # eV^2
m = pc['electron mass'][0] * 5.60958616721986e35 # eV
sin_theta_v = 0.5
rho = 1.0126
k_hat = 0.9791

gL = rho * (0.5 - k_hat * sin_theta_v) - 1
gR = -rho * k_hat * sin_theta_v

def dsigma_dT(T, q):
    z = T / q
    term1 = gL**2
    term2 = gR**2 * (1 - z)**2
    term3 = -gL * gR * m / q * z
    factor = 2 * G_F**2 * m / np.pi
    return factor * (term1 + term2 + term3)

def s_SNO(Tp):
    return -0.0684 + 0.331 * np.sqrt(Tp) + 0.0425 * Tp

def R(T, Tp):
    sigma = s_SNO(Tp)
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((T - Tp)**2) / (2 * sigma**2))

def sigma(Ev):
    Ev = Ev*1e6
    T_min, T_max = 0, Ev / (1 + m / (2 * Ev))
    def integrand(T):
        def integrand_inner(Tp):
            return R(T, Tp) * dsigma_dT(Tp, Ev)
        inner_res, _ = quad(integrand_inner, T_min, T_max)
        return inner_res
    outer_res, _ = quad(integrand, 0, Ev)
    return outer_res

Ev_arr = pd.read_csv('lambda.csv')['energy']  # Neutrino energy
results = [{'energy': Ev, 'sigma_m2': sigma(Ev)} for Ev in Ev_arr]

df = pd.DataFrame(results)
df.to_csv("crosssections.csv", index=False)
