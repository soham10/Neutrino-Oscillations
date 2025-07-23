import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("plot-data.csv")

x_vals = data['Energy']
y_vals = data['Observed']

plt.yscale('log')
plt.xlim(7,20)
plt.scatter(x_vals, y_vals)
plt.show()