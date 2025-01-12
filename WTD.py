import random
import matplotlib.pyplot as plt
import numpy as np

#TODO: show that the waitime gen is working correctly 
#? Wide Time Distribution => WTD

#! functions we use in the sim.
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



#! sims. 
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



#! graphs.
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
    plt.plot(wait_val, funcWait_val, color='green', label='T(t) = 1/2t^(-3/2)')

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


def view_logScale(num_sims:int, sim_time:int, prob_right:float) -> None:

    wait_val = np.linspace(1,50)
    funcWait_val = wait_time_func(wait_val)
    plt.plot(wait_val, funcWait_val, color='green', label='T(t) = 1/2t^(-3/2)')

    theory_val = np.linspace(-50, 50)
    func_val = theory_func(theory_val)
    #* plt.plot(theory_val, func_val, color='red', label='h(x) = - |x|')

    final_positions = multi_RW_sim(num_sims, sim_time, prob_right)
    #only_pos = filter_positive_numbers(final_positions)

    plt.hist(final_positions, bins=50, density=True, alpha=0.7, label='Final Positions')
    plt.xlabel("Distance from Origin")
    plt.ylabel("Probability Density (Log Scale)")  
    plt.yscale('log')
    #plt.xscale('log')  
    plt.title(f"Histogram of Final Positions after {sim_time} time steps")
    plt.legend()
    plt.show()



def filter_positive_numbers(numbers:list)->list:
  return [num for num in numbers if num > 0]

def comp_timeFunc_toTransform() -> None:

    x_values = np.linspace(0, 1,1000) 
    func_g_val = wait_time_func(x_values)
    gen_val = np.array([gen_wait_time() for _ in range(100_000)])
    t_values = np.linspace(1, 1000)
    #! trans_val = func_transform(gen_val)
    time_val = wait_time_func(t_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(x_values,func_g_val, color = 'green', label = 'g(x) = x**-2 ')
    ax1.hist(gen_val, bins=50, density=True, alpha=0.7, label='gen_val')
    ax1.set_xlabel("x")
    ax1.set_ylabel("g(x) = (x)^-2", color='blue')
    ax1.tick_params('y', labelcolor='blue')
    ax1.grid(True)

    ax2.plot(t_values, time_val, color='red', label='T(t) = 1/2t^(-3/2)')
    ax2.set_xlabel("t")
    ax2.set_ylabel("T(t)", color='red')
    ax2.grid(True)
    ax2.legend()
    plt.suptitle("Separate Plots of g(x) and T(t)")
    plt.tight_layout()
    plt.show()
