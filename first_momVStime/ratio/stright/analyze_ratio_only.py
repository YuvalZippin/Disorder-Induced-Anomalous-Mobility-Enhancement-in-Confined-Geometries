import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma

# Load the CSV file
filename = "sim_ratio_only.csv"
df = pd.read_csv(filename)

# --- Analytic ratio calculation ---
# Parameters (set to match your C++ config)
alpha = 0.3
w = 5
F = 0.01
A = 10.0
Q0 = 0.339499
polylog_val = 0.5673127065079584

# Leading order (Eq. 34)
def analytic_leading(F, w, alpha, t, A):
    gamma1p_alpha = gamma(1.0 + alpha)
    prefactor = 6**(-alpha) / (A * gamma1p_alpha * gamma1p_alpha)
    w_factor = w**(-2.0 * (1.0 - alpha))
    F_factor = F**alpha
    return prefactor * w_factor * F_factor * t**alpha

# Full theory (Eq. 20)
def analytic_full(F, Q0, alpha, t, A):
    num = np.sinh(F / 2.0)
    denom = 2.0 + np.cosh(F / 2.0)
    gamma1p_alpha = gamma(1.0 + alpha)
    factor = Q0 / (A * gamma1p_alpha * (1.0 - Q0)**2 * polylog_val)
    return (num / denom) * factor * t**alpha

# Analytic ratio (constant for fixed parameters)
def analytic_ratio(F, w, Q0, alpha, A):
    # t cancels, so use t=1
    leading = analytic_leading(F, w, alpha, 1.0, A)
    full = analytic_full(F, Q0, alpha, 1.0, A)
    return leading / full

R_analytic = analytic_ratio(F, w, Q0, alpha, A)
print("Analytic ratio R_analytic =", R_analytic)

# --- Plot R_sim vs time and analytic line ---
plt.figure(figsize=(8,6))
plt.semilogx(df['t'], df['ratio_sim'], 'o-', label='Simulation/Analytic Full $R(t)$')
plt.axhline(R_analytic, color='r', linestyle='--', label='Analytic Ratio (Eq. 34 / Eq. 20)')
plt.xlabel('Time $t$')
plt.ylabel('Ratio $R(t)$')
plt.title('Simulation/Analytic Full Ratio $R(t)$ vs Time')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Plot steps statistics
plt.figure(figsize=(8,6))
plt.loglog(df['t'], df['steps_max'], 'o-', label='Max Steps')
plt.loglog(df['t'], df['steps_mean'], 's-', label='Mean Steps')
plt.xlabel('Time $t$')
plt.ylabel('Steps per Trajectory')
plt.title('Steps Statistics vs Time')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Print summary statistics
print("Ratio_sim mean:", df['ratio_sim'].mean())
print("Ratio_sim std:", df['ratio_sim'].std())
print("Ratio_sim min/max:", df['ratio_sim'].min(), df['ratio_sim'].max())
print("Analytic ratio R_analytic:", R_analytic)
