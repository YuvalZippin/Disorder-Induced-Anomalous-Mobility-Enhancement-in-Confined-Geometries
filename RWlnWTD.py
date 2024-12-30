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
        positions, _ = RWsimWTD.random_walk_with_waiting_time(
            simulation_time, prob_right, RWsimWTD.waiting_time_from_g
        )
        final_positions.append(abs(positions[-1]))
    return final_positions


def main():
    num_sims = 100_000
    simulation_time = 500
    prob_right = 0.5

    # Run the simulations
    final_positions = multi_sim(num_sims, simulation_time, prob_right)

    # Filter out zero values to avoid log(0) issues
    filtered_positions = [pos for pos in final_positions if pos > 0]

    # Print some statistics of the filtered positions
    print(f"Mean final position: {np.mean(filtered_positions)}")
    print(f"Standard deviation of final positions: {np.std(filtered_positions)}")

    # You can also print individual final positions if desired
    for i in range(min(len(filtered_positions), 10)):  # Print only the first 10 positions
        print(f"Final position {i+1}: {filtered_positions[i]}")

    # Add a small constant to avoid log(0) issues
    epsilon = 1e-10
    ln_positions = np.log(np.array(filtered_positions) + epsilon)

    # Create a histogram with ln-transformed x-axis
    plt.hist(ln_positions, bins=50, density=True, alpha=0.7, label='Final Positions (ln)')
    plt.xlabel("ln(Distance from Origin)")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of ln(Final Positions) after {simulation_time} time steps")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
