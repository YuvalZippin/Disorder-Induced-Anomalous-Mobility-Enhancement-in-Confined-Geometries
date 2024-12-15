import random
import matplotlib.pyplot as plt
import numpy as np

#! exponential_waiting_time -> change to g(x)
def exponential_waiting_time(rate=1):
  """
  Generates a waiting time from an exponential distribution.
  """
  return np.random.exponential(1/rate)


def random_walk_with_waiting_time(simulation_time, prob_right, waiting_time_dist):
  """
  Simulates a 1-dimensional random walk with fixed step size of 1 and 
  a custom waiting time distribution.

  Args:
    simulation_time: The total simulation time.
    prob_right: Probability of moving to the right (between 0 and 1).
    waiting_time_dist: A function that generates waiting times.

  Returns:
    A list of positions representing the walker's path.
    A list of corresponding times for each position.
  """
  position = 0
  positions = [position]
  times = [0]
  current_time = 0

  while current_time < simulation_time:
    if random.random() < prob_right:
      step = 1
    else:
      step = -1
    position += step
    positions.append(position)

    # Generate waiting time using the provided distribution
    waiting_time = waiting_time_dist() 

    current_time += waiting_time
    times.append(current_time)

  return positions, times


def plot_random_walk(positions, times):
  plt.figure(figsize=(10, 6))

  # Plot jumps as vertical lines
  for i in range(len(positions) - 1):
    plt.plot([times[i], times[i]], [positions[i], positions[i+1]], 'b-', linewidth=2) 

  # Plot waiting times as horizontal lines
  for i in range(len(positions) - 1):
    plt.plot([times[i], times[i+1]], [positions[i+1], positions[i+1]], 'b--', linewidth=2) 

  plt.xlabel("Time")
  plt.ylabel("Position")
  plt.title("Biased Random Walk with Exponential Waiting Times")
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  simulation_time = 15  # Adjust as needed
  prob_right = 0.5  # Probability of moving right
  positions, times = random_walk_with_waiting_time(simulation_time, prob_right, exponential_waiting_time)
  plot_random_walk(positions, times)