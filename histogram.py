import numpy as np
import matplotlib.pyplot as plt

def T(t):
    """
    Defines the function T(t) = 1/2 * t^(-3/2)

    Args:
        t: The input value for the function.

    Returns:
        The value of T(t).
    """
    return 0.5 * t**(-3/2)

# Define the range of t values
t_min = 1
t_max = 10 # Adjust this upper limit for better zoom

# Generate a large number of random samples from the range
num_samples = 100000
t_samples = np.random.uniform(t_min, t_max, num_samples)

# Calculate the corresponding T(t) values
T_samples = T(t_samples)

# Create the histogram
plt.hist(T_samples, bins=100, density=True, alpha=0.7, label='Histogram') 

# Plot the actual function for comparison
t_plot = np.linspace(t_min, t_max, 1000)
T_plot = T(t_plot)
plt.plot(t_plot, T_plot, color='red', label='T(t)')

# Set x and y limits for zooming
plt.xlim(0, 50)  # Adjust x-axis limits as needed
plt.ylim(0, 1.5)  # Adjust y-axis limits as needed

plt.xlabel('t')
plt.ylabel('Probability Density')
plt.title('Histogram of T(t) = 1/2 * t^(-3/2)')
plt.legend()
plt.show()