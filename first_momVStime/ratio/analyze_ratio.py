import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load CSV data
filename = "sim_and_analytic_ratio.csv"
df = pd.read_csv(filename)

# Plot simulation and analytic moments vs time
plt.figure(figsize=(8,6))
plt.loglog(df['t'], df['sim_moment'], 'o-', label='Simulation')
plt.loglog(df['t'], df['analytic_leading'], 's-', label='Analytic Leading (Eq. 34)')
plt.loglog(df['t'], df['analytic_full'], '^-', label='Analytic Full (Eq. 20)')
plt.xlabel('Time $t$')
plt.ylabel('First Moment $\langle x \\rangle$')
plt.title('QTM: Simulation vs Analytic (Eq. 34 & 20)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot ratio_sim and ratio_analytic vs time
plt.figure(figsize=(8,6))
plt.semilogx(df['t'], df['ratio_sim'], 'o-', label='Simulation / Full Theory')
plt.semilogx(df['t'], df['ratio_analytic'], 's-', label='Leading / Full Theory')
plt.xlabel('Time $t$')
plt.ylabel('Ratio $R(t)$')
plt.title('QTM: Ratio of Simulation and Leading to Full Theory')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Optional: Plot steps statistics
plt.figure(figsize=(8,6))
plt.loglog(df['t'], df['steps_max'], 'o-', label='Max Steps')
plt.loglog(df['t'], df['steps_mean'], 's-', label='Mean Steps')
plt.xlabel('Time $t$')
plt.ylabel('Steps per Trajectory')
plt.title('QTM: Steps Statistics')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Print summary statistics
print("Ratio_sim mean:", df['ratio_sim'].mean())
print("Ratio_sim std:", df['ratio_sim'].std())
print("Ratio_sim min/max:", df['ratio_sim'].min(), df['ratio_sim'].max())
print("Ratio_analytic:", df['ratio_analytic'].iloc[0])  # Should be constant
