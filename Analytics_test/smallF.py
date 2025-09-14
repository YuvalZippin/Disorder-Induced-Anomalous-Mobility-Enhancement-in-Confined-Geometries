import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# --- Constants from the derivation ---
alpha = 0.3
A = 1.0
t = 10**14

# --- Equation for the leading-order first moment ---

def first_moment_leading(w, F):
    """
    Leading-order approximation for the first moment <x>.
    Equation (9)
    """
    gamma_term = gamma(1 + alpha)**2
    
    prefactor = (6**(-alpha)) / (A * gamma_term)
    
    # Handle the case where w is very small to avoid issues with exponentiation
    if np.any(w <= 0):
        w_safe = np.where(w <= 0, 1e-9, w)
    else:
        w_safe = w
    
    return prefactor * (w_safe**(-2 * (1 - alpha))) * (F**alpha) * (t**alpha)

# --- Plotting setup ---

# Define the range for the channel width w on a logarithmic scale
w_values = np.logspace(0, 2, 500)

# Define the forces F to plot
F_values = [0.1, 0.2, 0.5]

# Create a figure for the plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# --- Plotting the equation for each F value ---
for F in F_values:
    y_values = first_moment_leading(w_values, F)
    ax.plot(w_values, y_values, label=f'F = {F}')

# --- Customize the plot ---
ax.set_title(r'Leading-Order $\langle x \rangle$ vs. $w$', fontsize=16)
ax.set_xlabel('Channel Width ($w$)', fontsize=14)
ax.set_ylabel(r'First Moment ($\langle x \rangle$)', fontsize=14)
ax.set_xscale('log') # Logarithmic x-axis
ax.set_yscale('log') # Logarithmic y-axis
ax.legend(fontsize=12)
ax.text(0.5, 0.1, r'$\propto w^{-2(1-\alpha)}$', transform=ax.transAxes, fontsize=14)
plt.tight_layout()
plt.show()
