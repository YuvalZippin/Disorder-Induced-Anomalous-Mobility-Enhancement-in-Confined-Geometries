import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import expon, norm

def random_walk(max_time, lambd, prob_r):
    position = 0
    time = 0
    track = [(position, time)]  # Initial position and time

    while time < max_time:
        # Biased random step
        step = 1 if random.random() < prob_r else -1
        position += step

        # Wait for exponential time at the current position
        wait_time = expon.rvs(scale=1/lambd)
        time += wait_time

        # Add points to the track during the waiting time
        for t in np.linspace(time - wait_time, time, 100):
            track.append((t, position))

    return track

def main():
    MAX_TIME = 25
    LAMBDA = 100
    PROB_RIGHT = 0.5
    NUM_WALKS = 13_500

    mode = input("Enter 'single' for a single walk or 'histogram' for multiple walks: ")

    if mode == 'single':
        track = random_walk(MAX_TIME, LAMBDA, PROB_RIGHT)
        time_data, position_data = zip(*track)

        plt.figure(figsize=(10, 6))
        plt.plot(time_data, position_data, linewidth=2, color='blue', linestyle='--', marker='o', markersize=2)
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Biased Random Walk with Exponential Waiting Times")
        plt.show()

    elif mode == 'histogram':
        final_positions = []
        for _ in range(NUM_WALKS):
            track = random_walk(MAX_TIME, LAMBDA, PROB_RIGHT)
            final_positions.append(track[-1][1])

        # Calculate mean and standard deviation
        mean = np.mean(final_positions)
        std_dev = np.std(final_positions)

        # Generate x-axis values for the Gaussian curve
        x = np.linspace(min(final_positions), max(final_positions), 100)

        # Create the histogram with the specified bins
        plt.hist(final_positions, bins=x, density=True, alpha=0.7, label='Histogram')

        # Plot the Gaussian curve
        plt.plot(x, norm.pdf(x, mean, std_dev), color='red', label='Gaussian Fit')

        plt.xlabel("Final Position")
        plt.ylabel("Density")
        plt.title(f"Distribution of Final Positions (Time: {MAX_TIME}, Experiments: {NUM_WALKS})")
        plt.legend()
        plt.show()

    else:
        print("Invalid mode. Please enter 'single' or 'histogram'.")

if __name__ == "__main__":
    main()