import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Set style for professional appearance
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Load chi-squared data from CSV
data = pd.read_csv('Plots and Data/chi2_results0.05.csv')

# Get unique values for grid dimensions
dm2_unique = np.sort(data['dm2'].unique())
tan2theta_unique = np.sort(data['tan2theta'].unique())

# Create meshgrid - swapped X and Y for desired axis orientation
Y, X = np.meshgrid(dm2_unique, tan2theta_unique)

# Reshape chi2 data to match the grid
data_sorted = data.sort_values(['tan2theta', 'dm2'])
chi2_grid = data_sorted['chi2'].values.reshape(len(tan2theta_unique), len(dm2_unique)) # type: ignore

# Find minimum chi-squared value and its position
min_chi2 = np.min(chi2_grid)
min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
min_tan2theta = tan2theta_unique[min_idx[0]]
min_dm2 = dm2_unique[min_idx[1]]

# Create figure with appropriate size
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot filled contours with orange fill for inner region
levels = [min_chi2 + 2.30, min_chi2 + 5.99, min_chi2 + 9.21]  # 68%, 95%, 99% CL
contours = plt.contourf(X, Y, chi2_grid, levels=[min_chi2, min_chi2 + 2.30], 
                       colors=['orange'], alpha=0.8)

# Add contour lines with different styles
line_contours_68 = plt.contour(X, Y, chi2_grid, levels=[min_chi2 + 2.30], 
                              colors=['red'], linewidths=2, linestyles='-')
line_contours_95 = plt.contour(X, Y, chi2_grid, levels=[min_chi2 + 5.99], 
                              colors=['red'], linewidths=1.5, linestyles='--')
line_contours_99 = plt.contour(X, Y, chi2_grid, levels=[min_chi2 + 9.21], 
                              colors=['gray'], linewidths=1, linestyles=':')

# Mark the minimum point with a cross
plt.plot(min_tan2theta, min_dm2, 'k+', markersize=12, markeredgewidth=3)

# Add text annotation for minimum chi-squared value
plt.text(min_tan2theta + 0.05, min_dm2, f'$\\chi^2_{{min}} = {min_chi2:.2f}$', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Set axis properties
ax.set_xlabel(r'$\tan^2\theta_{12}$', fontsize=14)
ax.set_ylabel(r'$\Delta m^2_{21}$ ($\times 10^{-5}$ eV$^2$)', fontsize=14)
ax.set_title(r'$2 \times 2$ Solar Neutrino Oscillations with Fluctuations', fontsize=16, pad=20)

# Format y-axis to show scientific notation
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x*1e5:.0f}'))

# Set axis limits to focus on relevant region
ax.set_xlim(0, 1)
ax.set_ylim(dm2_unique.min(), dm2_unique.max())

# Add grid
ax.grid(True, alpha=0.3)

# Add legend
legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='68% CL'),
                   Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', label='95% CL'),
                   Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='99% CL')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()