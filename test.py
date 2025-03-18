import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func_transform(x):
    """Transform a uniform variable into a power-law waiting time."""
    return x**(-2)

def gen_wait_time(a=0, b=1) -> float:
    """Generate a single waiting time using func_transform."""
    return func_transform(random.uniform(a, b))

def generate_waiting_times(size: int) -> list:
    """Generate a shuffled list of waiting times, centered around the middle."""
    waiting_times = [gen_wait_time() for _ in range(size)]
    np.random.shuffle(waiting_times)
    mid_index = size // 2
    waiting_times[mid_index] = 0  # Immediate jump at the middle
    return waiting_times

def sample_jump():
    """
    Returns a jump (dx, dy) chosen from:
      - RIGHT: (1, 0) with probability 3/8,
      - LEFT:  (-1, 0) with probability 1/8,
      - UP:    (0, 1) with probability 1/4,
      - DOWN:  (0, -1) with probability 1/4.
    """
    r = random.random()
    if r < 3/8:
        return (1, 0)       # RIGHT
    elif r < 3/8 + 1/8:
        return (-1, 0)      # LEFT
    elif r < 3/8 + 1/8 + 1/4:
        return (0, 1)       # UP
    else:
        return (0, -1)      # DOWN

def RW_sim_2d_fixed_wait(sim_time: int, wait_list_size: int, Y_min: int, Y_max: int) -> tuple:
    """Simulate a 2D random walk with quenched waiting times and periodic boundaries in y."""
    
    wait_x = generate_waiting_times(wait_list_size)
    wait_y = generate_waiting_times(wait_list_size)
    
    x, y = 0, 0
    positions = [(x, y)]
    times = [0]
    current_time = 0
    x_index = y_index = wait_list_size // 2

    while current_time < sim_time:
        dx, dy = sample_jump()
        x += dx
        new_y = y + dy

        # Periodic boundary conditions in y
        y = Y_min if new_y > Y_max else Y_max if new_y < Y_min else new_y

        # Update waiting time indices
        x_index = max(0, min(wait_list_size - 1, x_index + dx)) if dx else x_index
        y_index = max(0, min(wait_list_size - 1, y_index + dy)) if dy else y_index

        # Select waiting time
        waiting_time = 0 if (x, y) == (0, 0) else max(wait_x[x_index], wait_y[y_index])
        current_time += waiting_time
        positions.append((x, y))
        times.append(current_time)
    
    return positions, times

def calculate_first_moment(positions) -> tuple:
    """Calculate the first moment (mean X, mean Y)."""
    x_coords, y_coords = zip(*positions)
    return np.mean(x_coords), np.mean(y_coords)

def multi_RW_first_moment_fixed_wait(num_sims: int, sim_time: int, wait_list_size: int, Y_min: int, Y_max: int) -> tuple[list,list]:
    """
    Run multiple simulations and compute first moments ⟨J_x⟩ and ⟨J_y⟩.

    Returns:
    - Lists of mean values ⟨J_x⟩ and ⟨J_y⟩ for each run.
    """
    first_moments_x = []
    first_moments_y = []

    for _ in range(num_sims):
        positions, _ = RW_sim_2d_fixed_wait(sim_time, wait_list_size, Y_min, Y_max)  # Extract only positions
        mean_x, mean_y = calculate_first_moment(positions)
        first_moments_x.append(mean_x)
        first_moments_y.append(mean_y)

    return first_moments_x, first_moments_y

def power_law(x, A, b):
    return A * x**b

def A_vs_W(num_sims:int, sim_time_start:int, sim_time_finish:int, time_step:int, wait_list_size:int, W_start:int, W_finish:int, W_step:int) -> None:
    """Find coefficient A as a function of system width W and plot it."""
    W_values = np.arange(W_start, W_finish, W_step)
    A_values = []
    
    for W in W_values:
        Y_min, Y_max = -W//2, W//2
        time_values = np.arange(sim_time_start, sim_time_finish + 1, time_step)
        mean_first_moments_x = []

        for sim_time in time_values:
            first_moments = [calculate_first_moment(RW_sim_2d_fixed_wait(sim_time, wait_list_size, Y_min, Y_max)[0]) for _ in range(num_sims)]
            mean_first_moments_x.append(np.mean([fm[0] for fm in first_moments]))
        
        if len(mean_first_moments_x) < 2 or np.any(np.isnan(mean_first_moments_x)) or np.any(np.isinf(mean_first_moments_x)):
            print(f"Skipping W={W}: Not enough valid data points for curve fitting.")
            continue

        popt, _ = curve_fit(power_law, time_values, mean_first_moments_x)
        A_fit, _ = popt
        A_values.append(A_fit)
        print(f"For W={W}, Estimated A: {A_fit:.4f}")
    
    if not A_values:
        print("Error: No valid A values calculated.")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(W_values[:len(A_values)], A_values, marker='o', linestyle='-', color='blue', label="A vs W")
    plt.xlabel("System Width W")
    plt.ylabel("Coefficient A")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    A_vs_W(num_sims=25_000, sim_time_start=0, sim_time_finish= 1_000, time_step=100, wait_list_size=150, W_start=0, W_finish=1, W_step=0.25)