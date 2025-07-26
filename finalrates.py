import pandas as pd
import matplotlib.pyplot as plt

# Read complete DataFrames
cross_section_data = pd.read_csv('crosssections.csv')
probability_data = pd.read_csv('Th Probability.csv')       
spectrum_data = pd.read_csv('lambda.csv')                  

# Merge on 'energy' column
df = cross_section_data.merge(probability_data, on='energy') \
                       .merge(spectrum_data, on='energy')

flux = 1.453e-2*4.51e6

# Compute the event rate for each energy bin
df['rate'] = df['results'] * df['sigma_m2'] * df['lambda'] * flux*1e36

df.to_csv('rates.csv', index=False)

plt.figure(figsize=(8, 5))
plt.scatter(df['energy'], df['rate'], marker='.',color='red', alpha=0.9)
plt.yscale('log')
plt.xlim(6,18)
plt.ylim(bottom = 1)
plt.xlabel('Energy')
plt.ylabel('Event Rate')
plt.title('Event Rates vs Energy')
plt.grid(True)
plt.show()
