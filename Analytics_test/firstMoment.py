import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# Parameters
alpha = 0.3
A = 1.0
t = 1e14
F_values = [0.01, 0.04, 0.25]
w_values = np.arange(1, 101)  # system sizes (avoid w=1 to prevent singularity)

def Q0_star(w, F):
    B = 1.0 / (2 * np.cosh(F/2) + 4)
    total = 0.0
    for m in range(w):
        for n in range(w):
            term = (1 - 2*B*(np.cos(2*np.pi*m/w) + np.cos(2*np.pi*n/w)))**2 - 4*B**2
            total += term**(-0.5)
    return 1 - (w**2)/total

def x_mean(w, F, alpha, A, t):
    Q0 = Q0_star(w, F)
    if Q0 <= 0 or Q0 >= 1:
        return np.nan  # avoid invalid values
    prefactor = np.sinh(F/2) / (2 + np.cosh(F/2))
    numerator = Q0
    denominator = A * mp.gamma(1+alpha) * (1-Q0)**2 * mp.polylog(-alpha, Q0)
    return prefactor * numerator/denominator * (t**alpha)

# Compute results
plt.figure(figsize=(8,6))
for F in F_values:
    results = [x_mean(w, F, alpha, A, t) for w in w_values]
    plt.plot(w_values, results, label=f"F={F}")

plt.xlabel("System size w")
plt.ylabel("<x(t)>")
plt.xscale('log')
plt.yscale('log')
plt.title("First moment <x(t)> vs system size w")
plt.legend()
plt.grid(True)
plt.show()
