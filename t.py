import numpy as np
import matplotlib.pyplot as plt

# Sample data (2D Gaussian-like)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt((X**2 + Y**2))

# Define contour levels
levels = [0.2, 0.3, 0.4]

# Define line styles for each level
linestyles = ['dotted', 'dashed', 'solid', 'solid']

# Create the plot
fig, ax = plt.subplots()

# Filled contour
contour_filled = ax.contourf(X, Y, Z, levels=[0.3, 0.4], colors=['cyan'])

# Overlay contour lines with red and varying linestyles
for i, level in enumerate(levels):
    ax.contour(X, Y, Z, levels=[level], colors='red', linestyles=linestyles[i], linewidths=2)

# Optional: Customize tick colors (like the blue ticks/spines in your image)
ax.tick_params(axis='both', colors='blue')
for spine in ax.spines.values():
    spine.set_color('blue')
    spine.set_linewidth(2)

plt.show()
