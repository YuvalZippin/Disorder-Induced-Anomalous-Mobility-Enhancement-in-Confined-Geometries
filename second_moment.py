import random
import numpy as np
import matplotlib.pyplot as plt
import WTD

def multi_RW_second_moment(num_sims: int, sim_time: int, prob_right: float) -> list:
    final_second_moment = []
    for _ in range(num_sims):
        positions, _ = WTD.RW_sim(sim_time, prob_right)
        single_sec_moment = WTD.calculate_second_moment(positions)
        final_second_moment.append(single_sec_moment)
    return final_second_moment


def Range_of_multi_sim(num_sims: int, sim_time_start: int, prob_right: float, sim_time_finish: int, time_step: int):
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

Range_of_multi_sim(num_sims=100_000, sim_time_start=0, prob_right=0.5, sim_time_finish=100, time_step=1)
