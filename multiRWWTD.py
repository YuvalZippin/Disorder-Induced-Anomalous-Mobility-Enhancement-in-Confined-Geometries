import RWsimWTD  # Assuming RWsim(WTD).py is saved as RWsim.py

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
  Main function to run multiple simulations and print results.
  """
  num_sims = 10
  simulation_time = 250
  prob_right = 0.5

  results = multi_sim(num_sims, simulation_time, prob_right)

  print("Experiment Results:")
  for experiment_num, final_position in results:
    print(f"Experiment {experiment_num}: Final Position = {final_position}")

# Ensure main function is only called when script is run directly
if __name__ == "__main__":
  main()