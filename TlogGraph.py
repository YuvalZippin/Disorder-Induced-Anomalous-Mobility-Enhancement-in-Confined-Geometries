import numpy as np
import matplotlib.pyplot as plt

# Define the range of values (adjust as needed)
x_values = np.linspace(0, 1, 1000)
t_values = np.linspace(1, 100, 1000)  # Start from 1 to avoid division by zero

# Define the functions
def g(x):
    return  (1-x)**(-2)

def T_half(t):
    return 0.5 * t**(-3/2)  # Renamed for clarity

# Calculate function values
g_values = g(x_values)
T_half_values = T_half(t_values)

# Create a new figure for logarithmic plots
plt.figure(figsize=(10, 6))

# Logarithmic plot for g(x)
plt.subplot(1, 2, 1)  # Create a subplot for g(x)
plt.plot(x_values, g_values, color='blue', label='g(x) = x^(-2)')
plt.xscale('log')  # Logarithmic x-axis for g(x)
plt.yscale('log')  # Logarithmic y-axis for g(x)
plt.xlabel("g(x) (Log Scale)")
plt.ylabel("g(x) (Log Scale)")
plt.title("Log-Log Plot of g(x) = x^-2")
plt.grid(True)

# Logarithmic plot for T(t)
plt.subplot(1, 2, 2)  # Create a subplot for T(t)
plt.plot(t_values, T_half_values, color='red', label='T(t) = 0.5 * t^(-3/2)')
plt.xscale('log')  # Logarithmic x-axis for T(t)
plt.yscale('log')  # Logarithmic y-axis for T(t)
plt.xlabel("t (Log Scale)")
plt.ylabel("T(t) (Log Scale)")
plt.title("Log-Log Plot of T(t) = 0.5 * t^(-3/2)")
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()