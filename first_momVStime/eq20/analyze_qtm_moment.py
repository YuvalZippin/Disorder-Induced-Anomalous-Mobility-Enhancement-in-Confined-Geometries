import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load CSV data
# Replace with your actual filename if different
filename = "sim_and_analytic_moment_eq20.csv"

# If you paste the data directly, you can use StringIO:
# from io import StringIO
# data = StringIO("""
# t,sim_moment,analytic_moment,ratio,steps_max,steps_mean
# 100000,0.06885,0.06197,1.1110,309,35.67
# ...
# """)
# df = pd.read_csv(data)

df = pd.read_csv(filename)

# Plot simulation and analytic moment vs time
plt.figure(figsize=(8,6))
plt.loglog(df['t'], df['sim_moment'], 'o-', label='Simulation')
plt.loglog(df['t'], df['analytic_moment'], 's-', label='Analytic (Eq. 20)')
plt.xlabel('Time $t$')
plt.ylabel('First Moment $\langle x \\rangle$')
plt.title('QTM: Simulation vs Analytic (Eq. 20)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot ratio vs time
plt.figure(figsize=(8,6))
plt.semilogx(df['t'], df['ratio'], 'x-', color='purple')
plt.xlabel('Time $t$')
plt.ylabel('Ratio (Simulation / Analytic)')
plt.title('QTM: Ratio of Simulation to Analytic (Eq. 20)')
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
print("Ratio mean:", df['ratio'].mean())
print("Ratio std:", df['ratio'].std())
print("Ratio min/max:", df['ratio'].min(), df['ratio'].max())
