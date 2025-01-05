import numpy as np
import matplotlib.pyplot as plt

# Define the range of values (adjust as needed)
x_values = np.linspace(0, 0.9, 1_000) # [0 , 1]
t_values = np.linspace(1, 100, 1_000)  # [1 , inf]

# Define the functions
def g(x):
    return (1-x)**-2

def T(t):
    return 0.5 * t**(-3/2)

# Calculate function values
g_values = g(x_values)
T_values = T(t_values)

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the histogram for g(x)
ax1.hist(g_values, bins=50, color='blue', alpha=0.7)
ax1.set_xlabel("x")
ax1.set_ylabel("g(x) = (1 - x)^-2", color='blue')
ax1.tick_params('y', labelcolor='blue')
ax1.grid(True)

# Create a second y-axis
ax2 = ax1.twinx()

# Plot T(t) on the second y-axis
ax2.plot(t_values, T_values, color='red', label='T(t) = 1/2t^(-3/2)')
ax2.set_ylabel("T(t)", color='red')
ax2.tick_params('y', labelcolor='red')
ax2.legend()

# Set the title and display the plot
plt.title("Combined Plot of g(x) and T(t)")
plt.show()