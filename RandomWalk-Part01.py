# This code simulates multiple random walks, each consisting of a specified number of jumps, and optionally visualizes the results.
#! to see the graph of multiple walks uncomment line 23
#! to see the histogram of the experiments uncomment line 24

import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import norm, gaussian_kde
from sklearn.metrics import r2_score

JUMPS = 500
EXPERIMENTS = 125_000
RIGHT_PROB = 0.7  # Change this value to adjust the probability of going right

def random_walk(jumps, right_prob):
  position = 50
  track = [position]
  for i in range(jumps):
    if random.random() < right_prob:
      position += 1
    else:
      position -= 1
    track.append(position)
  return track, position

results = []
for j in range(EXPERIMENTS):
  results.append(random_walk(JUMPS, RIGHT_PROB))

# Get final positions from all experiments
final_positions = [result[1] for result in results]

# Calculate the mean and standard deviation of the final positions
mean = np.mean(final_positions)
std_dev = np.std(final_positions)

# Generate x-axis values for the Gaussian curve and histogram bins
x = np.linspace(min(final_positions), max(final_positions), 100)

# Create the histogram with the specified bins
plt.hist(final_positions, bins=x, density=True, alpha=0.7, label='Histogram')

# Plot the Gaussian curve
gaussian_fit = norm.pdf(x, mean, std_dev)
plt.plot(x, gaussian_fit, color='red', label='Gaussian Fit')

plt.xlabel("Final Position")
plt.ylabel("Density")
plt.title(f"Random Walks (Jumps: {JUMPS}, Experiments: {EXPERIMENTS}, Right Prob: {RIGHT_PROB})")
plt.legend()
plt.show()