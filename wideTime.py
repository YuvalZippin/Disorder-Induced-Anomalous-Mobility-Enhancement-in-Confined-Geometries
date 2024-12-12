import numpy as np
import random
import matplotlib.pyplot as plt

def simulate_particle(start_time, end_time, step_size, initial_position, T, direction_distribution=[0.5, 0.5]):
    """
    Simulates the random walk of a particle in one dimension.

    Args:
        start_time: Start time of the simulation.
        end_time: End time of the simulation.
        step_size: Size of each step.
        initial_position: Initial position of the particle.
        T: Waiting time function.
        direction_distribution: Probability distribution of the direction of the step.

    Returns:
        A list of times and a list of corresponding positions.
    """

    time = start_time
    position = initial_position
    times = [time]
    positions = [position]

    while time < end_time:
        if time <= 0:
            raise ValueError("Time must be positive")

        direction = np.random.choice([-1, 1], p=direction_distribution)
        wait_time = T(time)
        time += wait_time
        position += direction * step_size
        times.append(time)
        positions.append(position)

    return times, positions

def main():
    # Parameters
    num_particles = 500
    start_time = 0.1
    end_time = 50
    step_size = 1
    initial_position = 0
    T = lambda t: 0.5 * t**(-3/2)
    direction_distribution = [0.5, 0.5]

    # Simulate particles and store final positions
    final_positions = []
    for _ in range(num_particles):
        times, positions = simulate_particle(start_time, end_time, step_size, initial_position, T, direction_distribution)
        final_positions.append(positions[-1])

    # Create histogram
    plt.hist(final_positions, bins=30)
    plt.xlabel("Final Position")
    plt.ylabel("Number of Particles")
    plt.title("Histogram of Final Positions")
    plt.show()

if __name__ == "__main__":
    main()