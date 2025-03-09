import numpy as np
import random
import matplotlib.pyplot as plt

def func_transform(x):
    #explain: 
    return (x)**(-2)

def gen_wait_time(a = 0, b = 1):
    return func_transform(random.uniform(a , b))

def generate_waiting_times(size: int) -> list:
    """Generate a shuffled list of waiting times using func_transform."""
    waiting_times = [gen_wait_time() for _ in range(size)]  # Generate using the given function
    np.random.shuffle(waiting_times)  # Shuffle to remove any ordering bias
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
    
    waiting_times = generate_waiting_times(wait_list_size)
    
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


def view_hist_fixed_wait(num_sims: int, sim_time: int, prob_right: float, wait_list_size: int ) -> None:
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


def calculate_second_moment(positions: list) -> float:
    # Calculate the second moment of the positions
    second_moment = sum([pos**2 for pos in positions]) / len(positions)
    return second_moment


def multi_RW_second_moment_fixed_wait(num_sims: int, sim_time: int, prob_right: float, wait_list_size: int) -> list:
    """
    Run multiple simulations and compute second moment for each.
    
    Parameters:
    - num_sims: Number of simulations.
    - sim_time: Total simulation time.
    - prob_right: Probability of stepping right.
    - wait_list_size: Size of the waiting time list.
    
    Returns:
    - List of second moments from each run.
    """
    final_second_moment = []
    for _ in range(num_sims):
        positions, _ = RW_sim_fixed_wait(sim_time, prob_right, wait_list_size)
        single_sec_moment = calculate_second_moment(positions)
        final_second_moment.append(single_sec_moment)
    return final_second_moment


def second_moment_with_noise_fixed_wait(num_sims: int, sim_time_start: int, prob_right: float, sim_time_finish: int, time_step: int, wait_list_size: int):
    """
    Plot second moment vs. time while visualizing noise.

    Parameters:
    - num_sims: Number of simulations.
    - sim_time_start: Starting simulation time.
    - sim_time_finish: Ending simulation time.
    - time_step: Interval between measurements.
    - prob_right: Probability of jumping right.
    - wait_list_size: Size of the waiting time list.
    """
    time_values = np.arange(sim_time_start, sim_time_finish + 1, time_step)
    all_times = []
    all_second_moments = []

    for sim_time in time_values:
        second_moments = multi_RW_second_moment_fixed_wait(num_sims, sim_time, prob_right, wait_list_size)
        all_times.extend([sim_time] * len(second_moments))  
        all_second_moments.extend(second_moments)  

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(all_times, all_second_moments, alpha=0.5, s=10, label="Second Moments")
    plt.xlabel("Simulation Time")
    plt.ylabel("Second Moment ⟨x²⟩")
    plt.title("Second Moment vs. Time for Random Walk with Fixed Waiting Times (Noise)")
    plt.legend()
    plt.grid()
    plt.show()


def second_moment_without_noise_comp_fixed_wait(num_sims: int, sim_time_start: int, prob_right: float, sim_time_finish: int, time_step: int, wait_list_size: int, a=0.5):
    """
    Compare mean second moment to a power-law function.

    Parameters:
    - num_sims: Number of simulations.
    - sim_time_start: Starting simulation time.
    - sim_time_finish: Ending simulation time.
    - time_step: Interval between measurements.
    - prob_right: Probability of jumping right.
    - wait_list_size: Size of the waiting time list.
    - a: Exponent for theoretical function f(t) = t^a.
    """
    time_values = np.arange(sim_time_start, sim_time_finish + 1, time_step)
    mean_second_moments = []  

    for sim_time in time_values:
        second_moments = multi_RW_second_moment_fixed_wait(num_sims, sim_time, prob_right, wait_list_size)
        mean_second_moments.append(np.mean(second_moments))

    # Compute the function f(t) = t^a and normalize
    f_t = time_values.astype(float) ** a  
    #! f_t *= (mean_second_moments[-1] / f_t[-1])  # Scaling for comparison

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(time_values, mean_second_moments, marker='o', linestyle='-', label="Mean Second Moment")
    plt.plot(time_values, f_t, linestyle='--', color='red', label=f"$t^{a}$ Fit")

    plt.xlabel("Simulation Time")
    plt.ylabel("Mean Second Moment ⟨x²⟩")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Mean Second Moment vs. Time for Random Walk with Fixed Waiting Times")
    plt.legend()
    plt.grid()
    plt.show()






#? Example usage:
def main():
    while True:
        print("\nMenu:")
        print("1. View Single Random Walk")
        print("2. View Histogram of Final Positions")
        print("3. View Second Moment with Noise")
        print("4. View Mean Second Moment with Power-Law Fit")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            positions, times = RW_sim_fixed_wait(sim_time=1_000, prob_right=0.5, wait_list_size=1_000)
            plot_random_walk(positions, times)

        elif choice == '2':
            view_hist_fixed_wait(num_sims=200_000, sim_time=15_000, prob_right=0.5, wait_list_size=1_000)

        elif choice == '3':
            num_sims = 500  # Number of simulations
            sim_time_start = 100
            sim_time_finish = 15_000
            time_step = 500
            wait_list_size = 1_000
            prob_right = 0.5

            second_moment_with_noise_fixed_wait(num_sims, sim_time_start, prob_right, sim_time_finish, time_step, wait_list_size)

        elif choice == '4':
            num_sims = 5_000  # Number of simulations
            sim_time_start = 0
            sim_time_finish = 1_000
            time_step = 100
            wait_list_size = 1_000
            prob_right = 0.5
            a = 0.5  # Exponent for theoretical function

            second_moment_without_noise_comp_fixed_wait(num_sims, sim_time_start, prob_right, sim_time_finish, time_step, wait_list_size, a)

        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()