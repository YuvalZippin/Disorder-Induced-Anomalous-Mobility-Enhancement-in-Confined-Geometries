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

# Plot the second moment as a function of time and the random walk positions
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot random walk positions
ax1.plot(times, positions, color='blue', label='Random Walk Positions')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Position', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis to plot the second moment
ax2 = ax1.twinx()
ax2.plot(times, second_moments, color='red', label='Second Moment of Positions', linestyle='--')
ax2.set_ylabel('Second Moment', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and legend
plt.title('Random Walk and Second Moment as a Function of Time')
fig.tight_layout()  # Adjust the layout to avoid overlap
plt.grid(True)

# Show the plot
plt.show()
