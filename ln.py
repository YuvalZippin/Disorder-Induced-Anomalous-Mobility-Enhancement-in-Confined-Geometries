import WTD
import matplotlib.pyplot as plt
import numpy as np

def main():
    num_sims = 200_000
    simulation_time = 50_000
    prob_right = 0.5

    # Run the simulations
    final_positions = WTD.multi_RW_sim(num_sims, simulation_time, prob_right)

    # Filter out zero values to avoid log(0) issues
    filtered_positions = [pos for pos in final_positions if pos > 0]

    #wait_val = np.linspace(1,50)
    #funcWait_val = WTD.wait_time_func(wait_val)
    #lt.plot(wait_val, funcWait_val, color='green', label='T(t) = 1/2t^(-3/2)')

    # Add a small constant to avoid log(0) issues
    epsilon = 1e-10
    ln_positions = np.log(np.array(filtered_positions) + epsilon)

    # Create a histogram with ln-transformed x-axis
    plt.hist(ln_positions, bins=50, density=True, alpha=0.7, label='Final Positions (ln)')
    plt.xlabel("ln(Distance from Origin)")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of ln(Final Positions) after {simulation_time} time steps")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
