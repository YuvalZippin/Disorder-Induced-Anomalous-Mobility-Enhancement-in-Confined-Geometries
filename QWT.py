import numpy as np
import random
import matplotlib.pyplot as plt

MIN_WAIT = 0
MAX_WAIT = 1_000

def generate_waiting_times(size: int , min_wait:int , max_wait:int) -> list:
    """Generate a shuffled list of waiting times."""
    waiting_times = np.random.uniform(min_wait, max_wait, size)  # Example: Uniformly distributed waiting times
    np.random.shuffle(waiting_times)
    return waiting_times

def RW_sim_fixed_wait(sim_time: int, prob_right: float, wait_list_size: int) -> tuple:
    """
    Simulate a random walk with fixed waiting times.
    
    Parameters:
    - sim_time: Total simulation time.
    - prob_right: Probability of stepping right.
    - wait_list_size: The size of the waiting time list.

    Returns:
    - positions: List of positions at each step.
    - times: List of corresponding times.
    """
    
    waiting_times = generate_waiting_times(wait_list_size, MIN_WAIT , MAX_WAIT)
    
    current_index = wait_list_size // 2  # Start in the middle of the waiting time list
    current_time = 0
    positions = [0]
    times = [0]

    while current_time < sim_time:
        step = 1 if random.random() < prob_right else -1
        current_index += step  # Move in the waiting time list
        
        # Ensure we stay within bounds
        current_index = max(0, min(wait_list_size - 1, current_index))
        
        # Get waiting time from precomputed list
        waiting_time = waiting_times[current_index]

        current_time += waiting_time
        positions.append(current_index - wait_list_size // 2)  # Center around 0
        times.append(current_time)

    return positions, times

def plot_random_walk(positions, times):
    """Plot the trajectory of the random walker."""
    plt.figure(figsize=(10, 6))

    for i in range(len(positions) - 1):
        plt.plot([times[i], times[i]], [positions[i], positions[i+1]], 'b-', linewidth=2)
        plt.plot([times[i], times[i+1]], [positions[i+1], positions[i+1]], 'b--', linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Random Walk with Fixed Waiting Times")
    plt.grid(True)
    plt.show()

def multi_RW_sim_fixed_wait(num_sims: int, sim_time: int, prob_right: float, wait_list_size: int) -> list:
    """
    Run multiple simulations of the random walker with fixed waiting times.

    Parameters:
    - num_sims: Number of independent simulations.
    - sim_time: Total simulation time.
    - prob_right: Probability of jumping right.
    - wait_list_size: The size of the waiting time list.

    Returns:
    - final_positions: List of final positions from each simulation.
    """
    final_positions = []
    for _ in range(num_sims):
        positions, _ = RW_sim_fixed_wait(sim_time, prob_right, wait_list_size)
        final_positions.append(positions[-1])  # Store the last position of each run
    return final_positions





def view_hist_fixed_wait(num_sims: int, sim_time: int, prob_right: float, wait_list_size: int = 1000) -> None:
    """
    Generate and plot a histogram of final positions from multiple random walk simulations
    with fixed waiting times.

    Parameters:
    - num_sims: Number of independent simulations.
    - sim_time: Total simulation time.
    - prob_right: Probability of jumping right.
    - wait_list_size: The size of the waiting time list.
    """
    
    final_positions = multi_RW_sim_fixed_wait(num_sims, sim_time, prob_right, wait_list_size)

    plt.figure(figsize=(8, 6))
    plt.hist(final_positions, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Final Positions')

    plt.xlabel("Distance from Origin", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(f"Histogram of Final Positions after {sim_time} time units", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()



#? Example usage:
def main():
    while True:
        print("\nMenu:")
        print("1. View Single Random Walk")
        print("2. View Histogram of Final Positions")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            positions, times = RW_sim_fixed_wait(sim_time = 15_000, prob_right = 0.5 , wait_list_size = 10_000)
            plot_random_walk(positions, times)

        elif choice == '2':
            view_hist_fixed_wait(num_sims = 200_000, sim_time = 15_000, prob_right = 0.5, wait_list_size = 10_000)

        elif choice == '3':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    main()
