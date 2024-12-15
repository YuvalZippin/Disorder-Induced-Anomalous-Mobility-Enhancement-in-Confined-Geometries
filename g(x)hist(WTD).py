import numpy as np
import matplotlib.pyplot as plt

# Define the range of t values (adjust as needed)
t_values = np.linspace(1, 100, 1_000)  # Start from 1 to avoid division by zero
x_values = np.linspace(0.1,1,1_000)

# Define the functions
def g(x):
    return x**-2

def T(t):
    return 0.5 * t**(-3/2)

# Calculate function values
g_values = g(x_values)
T_values = T(t_values)

# Create the histogram for g(t)
plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
plt.subplot(2, 1, 1)  # Create a subplot for g(t)
plt.hist(g_values, bins=50, color='blue', alpha=0.7)
plt.title("Histogram of g(x) = x^-2")
plt.xlabel("g(x)")  # Corrected x-axis label
plt.ylabel("Frequency")
plt.grid(True)

# Plot T(t)
plt.subplot(2, 1, 2)  # Create a subplot for T(t)
plt.plot(t_values, T_values, color='red', label='T(t) = 1/2t^(-3/2)')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()