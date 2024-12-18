import RWsimWTD  # Assuming RWsimWTD.py is saved as RWsimWTD.py
import matplotlib.pyplot as plt
import numpy as np


def multi_sim(num_sims, simulation_time, prob_right):
  """
  Performs multiple random walk simulations and returns final positions
  and experiment numbers.

  Args:
    num_sims: Number of simulations to run.
    simulation_time: Total simulation time for each run.
    prob_right: Probability of moving to the right.

  Returns:
    A list of tuples containing (experiment_number, final_position).
  """
  results = []
  for i in range(num_sims):
    positions, times = RWsimWTD.random_walk_with_waiting_time(
        simulation_time, prob_right, RWsimWTD.waiting_time_from_g)
    final_position = positions[-1]  # Get the last position (final)
    results.append((i + 1, final_position))  # Add experiment number and position
  return results


def main():
  """
  Main function to run multiple simulations, get results, and plot them.
  """
  num_sims = 10
  simulation_time = 100
  prob_right = 0.5

  results = multi_sim(num_sims, simulation_time, prob_right)

  # Extract positions and times from results
  positions = []
  times = []
  for experiment_num, final_position in results:
    positions.append(final_position)

  # Loop through each simulation and plot its trajectory
  for i, position in enumerate(positions):
    times = np.linspace(0, simulation_time, len(results[i][1]))
    plt.plot(times, position, label=f"Experiment {i+1}")  # Plot using times and positions

  # Configure plot with logarithmic time axis and labels
  plt.xlabel("Time (log scale)")
  plt.ylabel("Position")
  plt.title("Biased Random Walks with Waiting Time (g(x)=x^-2)")
  plt.yscale("linear")  # Ensure linear scale for position axis
  plt.xscale("log")  # Set logarithmic scale for time axis
  plt.grid(True)
  plt.legend()
  plt.show()

# Ensure main function is only called when script is run directly
if __name__ == "__main__":
  main()