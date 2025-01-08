import random
import matplotlib.pyplot as plt
import numpy as np


# [1 , inf]
def wait_time_func(t):
    return 0.5*t**(-3/2)

# [0 , 1]
def func_transform(x):
    return (1-x)**(-2)

def theory_func(x):
    return np.exp(-np.abs(x))

def gen_wait_time(a = 0, b = 1):
    x = random.uniform(a , b)
    return func_transform(x)

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

    wait_val = np.linspace(1,25)
    funcWait_val = wait_time_func(wait_val)
    plt.plot(wait_val, funcWait_val, color='green', label='T(t) = 1/2t^(-3/2)')

    theory_val = np.linspace(-25, 25)
    funcTheory_val = theory_func(theory_val)
    #* plt.plot(theory_val, funcTheory_val, color='red', label='f(x) = e^(-|x|)')

    final_positions = multi_RW_sim(num_sims, sim_time, prob_right)
    plt.hist(final_positions, bins=100, density=True, alpha=0.7, label='Final Positions')
    plt.xlabel("Distance from Origin")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of Final Positions after {sim_time} time steps")
    plt.legend()
    plt.show()


def view_logScale(num_sims:int, sim_time:int, prob_right:float) -> None:

    wait_val = np.linspace(1,25)
    funcWait_val = wait_time_func(wait_val)
    #* plt.plot(wait_val, funcWait_val, color='green', label='T(t) = 1/2t^(-3/2)')

    theory_val = np.linspace(-25, 25)
    func_val = theory_func(theory_val)
    #* plt.plot(theory_val, func_val, color='red', label='h(x) = - |x|')

    final_positions = multi_RW_sim(num_sims, sim_time, prob_right)
    plt.hist(final_positions, bins=50, density=True, alpha=0.7, label='Final Positions')

    plt.xlabel("Distance from Origin")

    plt.ylabel("Probability Density (Log Scale)")  
    plt.yscale('log')  

    plt.title(f"Histogram of Final Positions after {sim_time} time steps")
    plt.legend()
    plt.show()