import random
import matplotlib.pyplot as plt
import numpy as np

def func(x):
  """
  Defines the waiting time distribution g(x) = x^-2 
  within the range [0.1, 1].
  """
  if x <= 0:
    return 0  # Avoid division by zero and negative waiting times
  elif x < 0.1:
    return 0.1  # Lower bound for waiting time
  else:
    return x**-2

def waiting_time_from_g(a=0.1, b=1):
  """
  Generates a waiting time using rejection sampling from g(x) = x^-2 
  within the range [a, b].
  """
  while True:
    y = random.uniform(0, 1)
    return func(y)

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

    waiting_time = waiting_time_dist() 
    current_time += waiting_time
    times.append(current_time)

  return positions, times

# Main execution block
if __name__ == "__main__":
  simulation_time = 250
  prob_right = 0.5

  positions, times = random_walk_with_waiting_time(simulation_time, prob_right, waiting_time_from_g)

  plt.figure(figsize=(10, 6))

  for i in range(len(positions) - 1):
    plt.plot([times[i], times[i]], [positions[i], positions[i+1]], 'b-', linewidth=2) 
    plt.plot([times[i], times[i+1]], [positions[i+1], positions[i+1]], 'b--', linewidth=2) 

  plt.xlabel("Time")
  plt.ylabel("Position")
  plt.title("Biased Random Walk with Waiting Time (g(x)=x^-2)")
  plt.grid(True)
  plt.show()