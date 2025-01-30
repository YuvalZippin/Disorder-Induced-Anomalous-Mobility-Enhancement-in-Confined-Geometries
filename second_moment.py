import random
import matplotlib.pyplot as plt
import WTD


def RW_sim(sim_time: int, prob_right: float) -> tuple:
    current_position = 0 
    current_time = 0
    positions = [current_position]
    times = [current_time]
    second_moments = [current_position**2]  # Initialize with the second moment of the first position
    
    while current_time < sim_time: 
        if random.random() < prob_right:
            step = 1
        else: 
            step = -1
        current_position += step
        positions.append(current_position)

        # Add waiting time after each step
        waiting_time = WTD.gen_wait_time()
        current_time += waiting_time
        times.append(current_time)
        
        # Calculate second moment at this point
        second_moment = sum([pos**2 for pos in positions]) / len(positions)
        second_moments.append(second_moment)

    return times, positions, second_moments

# Example of usage:
sim_time = 250  # total time for the simulation
prob_right = 0.5  # probability of moving right

# Run the random walk simulation and get the second moments
times, positions, second_moments = RW_sim(sim_time, prob_right)

   
for i in range(len(second_moments) - 1):
        plt.plot([times[i], times[i]], [second_moments[i], second_moments[i+1]], 'b-', linewidth=2)
        plt.plot([times[i], times[i+1]], [second_moments[i+1], second_moments[i+1]], 'b--', linewidth=2)

plt.xlabel("Time")
plt.ylabel("second momment")
plt.title("Random Walk with wide time distribution")
plt.grid(True)
plt.show()

print(second_moments)
print(positions)