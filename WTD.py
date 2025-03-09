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


def calculate_second_moment(positions: list) -> float:
    # Calculate the second moment of the positions
    second_moment = sum([pos**2 for pos in positions]) / len(positions)
    return second_moment


def multi_RW_second_moment(num_sims: int, sim_time: int, prob_right: float) -> list:
    final_second_moment = []
    for _ in range(num_sims):
        positions, _ = RW_sim(sim_time, prob_right)
        single_sec_moment = calculate_second_moment(positions)
        final_second_moment.append(single_sec_moment)
    return final_second_moment


def second_moment_with_noise(num_sims: int, sim_time_start: int, prob_right: float, sim_time_finish: int, time_step: int):
    time_values = np.arange(sim_time_start, sim_time_finish + 1, time_step)
    all_times = []
    all_second_moments = []

    for sim_time in time_values:
        second_moments = multi_RW_second_moment(num_sims, sim_time, prob_right)
        all_times.extend([sim_time] * len(second_moments))  # Repeat time for each value
        all_second_moments.extend(second_moments)  # Store all second moment values

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(all_times, all_second_moments, alpha=0.5, s=10, label="Second Moments")
    plt.xlabel("Simulation Time")
    plt.ylabel("Second Moment ⟨x²⟩")
    plt.title("Second Moment vs. Time for Random Walk (with Noise)")
    plt.legend()
    plt.grid()
    plt.show()


def second_moment_without_noise_comp(num_sims: int, sim_time_start: int, prob_right: float, sim_time_finish: int, time_step: int, a=0.5):
    time_values = np.arange(sim_time_start, sim_time_finish + 1, time_step)
    mean_second_moments = []  

    for sim_time in time_values:
        second_moments = multi_RW_second_moment(num_sims, sim_time, prob_right)
        mean_second_moments.append(np.mean(second_moments))

    # Compute the function f(t) = t^a and ensure it's a float array
    f_t = time_values.astype(float)**a  # Convert time_values to float

    # Normalize f(t) to match the scale of mean second moments
    #! f_t *= (mean_second_moments[-1] / f_t[-1])

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(time_values, mean_second_moments, marker='o', linestyle='-', label="Mean Second Moment")
    plt.plot(time_values, f_t, linestyle='--', color='red', label=f"$t^{a}$ Fit")

    plt.xlabel("Simulation Time")
    plt.ylabel("Mean Second Moment ⟨x²⟩")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Mean Second Moment vs. Time for Random Walk")
    plt.legend()
    plt.grid()
    plt.show()






import WTD

def main():
    
    while True:

        print("\nMenu:")
        print("1. View Single Random Walk")
        print("2. View Histogram of Final Positions")
        print("3. View log scale of Final Positions")
        print("4. comp wait time func to transform")
        print("5. To calc the second moment")
        print("6. View Second Moment vs. Time for Random Walk (with Noise)")
        print("7. View Mean Second Moment vs. Time for Random Walk && comp to f(t)=t^a [a = 0.5]")

        print("9. Exit")

        choice = input("Enter your choice (1-7) or 9 to EXIT: ")

        if choice == '1':
            WTD.view_single_RW(sim_time = 15_000, prob_right = 0.5)

        elif choice == '2':
            WTD.view_hist(num_sims = 200_000, sim_time = 10_000, prob_right = 0.5)

        elif choice =='3':
            WTD.view_logScale(num_sims = 200_000, sim_time = 15_000, prob_right = 0.5)

        elif choice == '4':
            WTD.comp_timeFunc_toTransform()

        elif choice == '5':
            positions,times = WTD.RW_sim(sim_time=500, prob_right=0.5)
            print(f"the positions are: {positions}")
            second_moment = WTD.calculate_second_moment(positions)
            print(f"Second moment of positions: {second_moment}")

        elif choice =='6':
            WTD.second_moment_with_noise(num_sims=25_000, sim_time_start=0, prob_right=0.5, sim_time_finish=100, time_step=1)

        elif choice =='7':
            WTD.second_moment_without_noise_comp(num_sims=100_000, sim_time_start=0, prob_right=0.5, sim_time_finish=1_000, time_step=25)
            


        elif choice == '9':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5 or 9 to EXIT.")

if __name__ == "__main__":
    main()