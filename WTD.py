import random
import matplotlib.pyplot as plt
import numpy as np

#? Wide Time Distribution => WTD

def wait_time_func(t):
    # explain:
    return 0.5*t**(-3/2)

def func_transform(x):
    #explain: 
    return (x)**(-2)

def theory_func(x):
    return (np.exp(-np.abs(x)))

def gen_wait_time(a = 0, b = 1):
    return func_transform(random.uniform(a , b))



def RW_sim(sim_time:int, prob_right:float) -> list:

    current_position = 0 
    current_time = 0
    positions = [current_position]
    times = [current_time]

    while (current_time < sim_time): 
        if (random.random() < prob_right):
            step = 1
        else : 
            step = -1
        current_position += step
        positions.append(current_position)

        waiting_time = gen_wait_time()
        current_time += waiting_time
        times.append(current_time)

    return positions, times


def multi_RW_sim(num_sims:int, sim_time:int, prob_right:float) -> list:
    final_positions = []
    for _ in range(num_sims):
        positions , _ = RW_sim(sim_time, prob_right)
        final_positions.append(positions[-1])
    return final_positions



def view_single_RW(sim_time:int , prob_right:float) -> None:
    
    positions , times = RW_sim(sim_time, prob_right)
    plt.figure(figsize=(10,6))
    
    for i in range(len(positions) - 1):
        plt.plot([times[i], times[i]], [positions[i], positions[i+1]], 'b-', linewidth=2)
        plt.plot([times[i], times[i+1]], [positions[i+1], positions[i+1]], 'b--', linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Random Walk with WTD Waiting Time")
    plt.grid(True)
    plt.show()


def view_hist(num_sims:int, sim_time:int, prob_right:float) -> None:

    wait_val = np.linspace(1,50)
    funcWait_val = wait_time_func(wait_val)
    #* plt.plot(wait_val, funcWait_val, color='green', label='T(t) = 1/2t^(-3/2)')

    theory_val = np.linspace(-50, 50)
    funcTheory_val = theory_func(theory_val)
    #* plt.plot(theory_val, funcTheory_val, color='red', label='f(x) = e^(-|x|)')

    final_positions = multi_RW_sim(num_sims, sim_time, prob_right)
    plt.hist(final_positions, bins=100, density=True, alpha=0.7, label='Final Positions')
    plt.xlabel("Distance from Origin")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of Final Positions after {sim_time} time steps")
    plt.legend()
    plt.show()


def view_logScale(num_sims:int, sim_time:int, prob_right:float, log_scale:bool = True) -> None:

    #wait_val = np.linspace(1, 50)
    #funcWait_val = wait_time_func(wait_val)
    #* plt.plot(wait_val, funcWait_val, color='green', label='T(t) = 1/2t^(-3/2)')

    #theory_val = np.linspace(-50, 50)
    #funcTheory_val = theory_func(theory_val)
    #* plt.plot(theory_val, funcTheory_val, color='red', label='f(x) = e^(-|x|)')

    # Run the simulation and filter positive positions
    final_positions = multi_RW_sim(num_sims, sim_time, prob_right)
    #! final_positions = [pos for pos in final_positions if pos > 0]  # Keep only positive positions

    plt.hist(final_positions, bins=100, density=True, alpha=0.7, label='Final Positions (Positive)')
    plt.xlabel("Distance from Origin")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of Final Positions after {sim_time} time steps")

    if log_scale:
        plt.xscale('log')  # Set the x-axis to a logarithmic scale
        plt.yscale('log')  # Set the y-axis to a logarithmic scale
        plt.xlabel("Log Distance from Origin (Natural Log)")
        plt.ylabel("Log Probability Density (Natural Log)")

    plt.legend()
    plt.show()


def comp_timeFunc_toTransform() -> None:

    # Generate random numbers between 0 and 1
    n_samples = 100000000  # Number of random samples
    random_numbers = np.random.uniform(0, 1, n_samples)

    # Apply the transformation g(x) to the random numbers
    transformed_values = func_transform(random_numbers)

    # Generate histogram of transformed values
    bins = np.linspace(1, 100, 100)  # Define histogram bins
    hist, bin_edges = np.histogram(transformed_values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the histogram and compare it with T(t)
    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.6, color='blue', label='Histogram (g(x))')

    # Plot the original function T(t)
    t_values = np.linspace(1, 100, 500)
    plt.plot(t_values, wait_time_func(t_values), color='red', label='T(t) = 1/2 t^(-3/2)', linewidth=2)

    # Add labels and legend
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.title('Comparison of Histogram and T(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_second_moment_pos (positions:list) -> float:
  
  # Extract only the positions from the list of tuples
  positions = [position for position, _ in positions]

  # Calculate the mean
  # mean = sum(positions) / len(positions)
  mean = 0

  # Calculate the squared differences from the mean
  squared_diffs = [(position - mean)**2 for position in positions]

  # Calculate the variance (second moment)
  variance = sum(squared_diffs) / len(positions)

  return variance