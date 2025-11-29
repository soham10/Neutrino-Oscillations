#!/usr/bin/env python3
"""
Compute χ² grid comparing MC.csv (experimental ratios) to theory.

Approach:
- Load `Finall/MC.csv` (columns: Recoil energy(MeV), rate, sigma_energy, sigma_event)
- Import `chi2_fast` to use `compute_Pee_spectrum` and `compute_theoretical_rates`.
- For each (dm2, tan2theta) compute:
    - Pee_spectrum = compute_Pee_spectrum(dm2, tan2theta, BETA_FIXED, TAU_FIXED)
    - rates_osc = compute_theoretical_rates(Pee_spectrum)
    - rates_noosc = compute_theoretical_rates(np.ones_like(Pee_spectrum))
    - theoretical_ratio = rates_osc / rates_noosc
    - χ² = Σ((exp_ratio - theoretical_ratio_at_Te)/exp_sigma)^2

Saves results to `chi2_grid_results.csv` in the same folder.
"""
import numpy as np
import pandas as pd
import os
import time

import chi2_fast


def load_mc(path):
    df = pd.read_csv(path)
    # Expect columns: Recoil energy(MeV), rate, sigma_energy, sigma_event
    df = df.rename(columns={df.columns[0]: 'Te', df.columns[1]: 'rate', df.columns[2]: 'sigma_energy', df.columns[3]: 'sigma_event'})
    return df


def map_theory_to_expt(theo_te, theo_ratio, expt_te):
    # theo_te is chi2_fast.exp_te (order used when computing theoretical rates)
    # theo_ratio length matches theo_te; interpolate onto expt_te
    return np.interp(expt_te, theo_te, theo_ratio)


def compute_chi2_for_grid(mc_df, dm2_vals, tan2_vals, out_fname='chi2_grid_results.csv'):
    results = []

    # Precompute rates with no oscillations (Pee=1)
    Pee_noosc = np.ones(len(chi2_fast.E_nu_vals))
    rates_noosc = chi2_fast.compute_theoretical_rates(Pee_noosc)
    theo_te = chi2_fast.exp_te  # Te bin centers used inside chi2_fast

    exp_te = mc_df['Te'].values
    exp_ratio = mc_df['rate'].values
    exp_sigma = mc_df['sigma_event'].values

    total = len(dm2_vals) * len(tan2_vals)
    tic = time.time()
    idx = 0
    for dm2 in dm2_vals:
        for t2 in tan2_vals:
            idx += 1
            t0 = time.time()
            # compute Pee spectrum (uses chi2_fast.BETA_FIXED and TAU_FIXED)
            Pee = chi2_fast.compute_Pee_spectrum(dm2, t2, chi2_fast.BETA_FIXED, chi2_fast.TAU_FIXED)

            rates_osc = chi2_fast.compute_theoretical_rates(Pee)
            # ratio per Te bin
            theo_ratio_full = rates_osc / rates_noosc

            # interpolate/match to experimental Te bins
            theo_ratio_at_exp = map_theory_to_expt(theo_te, theo_ratio_full, exp_te)

            chi2 = np.sum(((exp_ratio - theo_ratio_at_exp) / exp_sigma) ** 2)
            results.append((dm2, t2, float(chi2)))

            t1 = time.time()
            if idx % 5 == 0 or idx == total:
                eta = (t1 - tic) / idx * (total - idx)
                print(f"[{idx}/{total}] dm2={dm2:.3e} t2={t2:.3e} χ²={chi2:.2f} (Δt={t1-t0:.2f}s) ETA={eta/60:.1f}min")

    df_out = pd.DataFrame(results, columns=['dm2', 'tan2theta', 'chi2'])
    df_out.to_csv(out_fname, index=False)
    print(f"Saved χ² grid to {out_fname} ({len(df_out)} points)")
    return df_out


def main():
    base = os.path.dirname(__file__)
    mc_path = os.path.join(base, 'MC.csv')
    if not os.path.exists(mc_path):
        raise FileNotFoundError(f"MC.csv not found at {mc_path}")

    mc_df = load_mc(mc_path)

    # Example grid (small by default — increase for production)
    dm2_vals = np.logspace(-6, -3, 5)   # eV^2
    tan2_vals = np.logspace(-3, np.log10(5.0), 5)

    out_fname = os.path.join(base, 'chi2_grid_results.csv')
    compute_chi2_for_grid(mc_df, dm2_vals, tan2_vals, out_fname=out_fname)


if __name__ == '__main__':
    main()
