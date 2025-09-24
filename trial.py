from scipy.constants import physical_constants as pc


G_F = pc['Fermi coupling constant'][0] * 1e-18       # eV^-2
m_e = pc['electron mass'][0] * 5.60958616721986e35   # eV
N_A = pc['Avogadro constant'][0]
eV_to_1_by_m = pc['electron volt-inverse meter relationship'][0]
one_by_cm3_to_eV3 = (1.973e-5) ** 3
eV_to_1_by_km = eV_to_1_by_m * 1e3
R_sol = 6.9634e8 * eV_to_1_by_m
R_earth = 1.496e11 * eV_to_1_by_m


print(G_F,m_e, N_A)
print(eV_to_1_by_m)
print(1/(100*eV_to_1_by_m))