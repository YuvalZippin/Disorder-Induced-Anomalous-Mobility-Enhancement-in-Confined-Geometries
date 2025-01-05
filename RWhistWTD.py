import RWsimWTD
import matplotlib.pyplot as plt
import numpy as np

def multi_sim(num_sims, simulation_time, prob_right):
    """
    Runs multiple random walk simulations and returns final positions.

    Args:
        num_sims: Number of simulations to run.
        simulation_time: Total simulation time for each run.
        prob_right: Probability of moving to the right.

    Returns:
        A list of final positions from all simulations.
    """
    final_positions = []
    for _ in range(num_sims):
        positions, _ = RWsimWTD.random_walk_with_waiting_time(simulation_time, prob_right, RWsimWTD.waiting_time_from_g)
        final_positions.append(abs(positions[-1]))
    return final_positions

def main():
    num_sims = 250_000
    simulation_time = 10_000
    prob_right = 0.5

    # Run the simulations
    final_positions = multi_sim(num_sims, simulation_time, prob_right)

    # Print some statistics of the final positions
    print(f"Mean final position: {np.mean(final_positions)}")
    print(f"Standard deviation of final positions: {np.std(final_positions)}")

    # You can also print individual final positions if desired
    for i in range(min(len(final_positions), 10)):  # Print only the first 10 positions
        print(f"Final position {i+1}: {final_positions[i]}")

    # Create a histogram (optional)
    plt.hist(final_positions, bins=50, density=True, alpha=0.7, label='Final Positions')
    plt.xlabel("Distance from Origin")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of Final Positions after {simulation_time} time steps")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()