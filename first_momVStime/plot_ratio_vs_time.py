import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
# If you add a column 'x_num' for the numerical value, everything works automatically.
data = pd.read_csv('ratio_vs_time.csv')

# Extract columns
time = data['time'].values
x_leading = data['x_leading'].values
x_infty = data['x_infty'].values
ratio = data['ratio'].values

# Optional: Load your simulation results as well, e.g. 'x_num'
has_sim_data = 'x_num' in data.columns
if has_sim_data:
    x_num = data['x_num'].values
    # Calculate the observed ratio vs theory
    ratio_num = x_leading / x_num

plt.figure(figsize=(8, 5))
# Plot theoretical ratio
plt.plot(time, ratio, marker='o', linestyle='-', color='navy', label=r'Theory $\langle x \rangle_{\mathrm{leading}} / \langle x \rangle_{\infty}$')

if has_sim_data:
    plt.plot(time, ratio_num, marker='^', linestyle='--', color='crimson',
             label=r'Simulation $\langle x \rangle_{\mathrm{leading}} / \langle x \rangle_{\mathrm{num}}$')

plt.xscale('log')
plt.xlabel('Time $t$', fontsize=14)
plt.ylabel(r'Ratio', fontsize=14)
plt.title('Ratio: Theory vs Simulation vs Time', fontsize=16)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("ratio_vs_time_with_sim.png", dpi=300)
plt.show()

# Print summary statistics:
print(f"Theory ratio mean: {np.mean(ratio):.6f}, std: {np.std(ratio):.3e}")
if has_sim_data:
    print(f"Simulated ratio mean: {np.mean(ratio_num):.6f}, std: {np.std(ratio_num):.3e}")
