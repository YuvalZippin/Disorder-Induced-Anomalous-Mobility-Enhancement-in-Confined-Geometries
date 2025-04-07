import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def generate_eta(alpha: float) -> float:
    """
    Generate one-sided Lévy stable random variable η using the Chambers et al. method.
    """
    theta = np.random.uniform(0, np.pi)  # θ ~ U(0, π)
    W = -np.log(np.random.uniform(0, 1))  # W = -ln(U), U ~ U(0,1)
    a_theta = (
        np.sin((1 - alpha) * theta)
        * (np.sin(alpha * theta) ** (alpha / (1 - alpha)))
        / (np.sin(theta) ** (1 / (1 - alpha)))
    )
    eta = (a_theta / W) ** ((1 - alpha) / alpha)
    return eta

def compute_S_alpha(t: float, alpha: float) -> float:
    """
    Compute the hitting target S_α for a given time t and stability parameter alpha.
    """
    eta = generate_eta(alpha)
    S_alpha = (t / eta) ** alpha
    return S_alpha

def levy_pdf_alpha_half(x):
    """
    Compute the theoretical Lévy PDF for α = 1/2 as a function of x.
    """
    pdf = np.zeros_like(x)
    positive = x > 0
    pdf[positive] = (1.0 / (2.0 * np.sqrt(np.pi))) * (x[positive] ** -1.5) * np.exp(-1.0 / (4.0 * x[positive]))
    return pdf


def run_eta_validation_plot(alpha: float = 0.5, N_samples: int = 100_000, hist_range=(0.001, 25.0), num_bins=200):
    """
    Run validation of generate_eta by comparing histogram of samples to theoretical PDF.
    """
    print(f"\nGenerating {N_samples} samples with α = {alpha}...")
    np.random.seed(42)  # For reproducibility
    samples = np.array([generate_eta(alpha) for _ in range(N_samples)])
    counts, bins = np.histogram(samples, bins=num_bins, range=hist_range, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    x_vals = np.linspace(hist_range[0], hist_range[1], 1000)
    pdf_vals = levy_pdf_alpha_half(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, pdf_vals, 'r-', label="Theoretical Lévy PDF (α=0.5)", linewidth=2)
    plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), alpha=0.6, label="Generated η Histogram", color='skyblue', edgecolor='k')
    plt.title("Validation of Lévy Stable Sampling (α=0.5)")
    plt.xlabel("η")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_single_trajectory(t: float, alpha: float, F: float, Ly: int, Lz: int) -> list[tuple[int, int, int]]:
    """
    Simulate a single 3D biased random walk with periodic boundary conditions in y and z.
    
    The walk stops when the number of steps reaches or exceeds the operational time S_α,
    computed via compute_S_alpha(t, alpha). The x-direction is infinite while y and z are
    periodic with system sizes Ly and Lz, respectively.
    
    Parameters:
        t (float): Laboratory time.
        alpha (float): Stability parameter for the Lévy distribution.
        F (float): External field strength (bias in the x-direction).
        Ly (int): System size in the y-direction (allowed positions: 0 to Ly-1).
        Lz (int): System size in the z-direction (allowed positions: 0 to Lz-1).
    
    Returns:
        trajectory (list of tuples): The complete trajectory of the particle.
                                     Each element is a tuple (x, y, z) representing
                                     the position after each step (including the initial position).
    """
    S_alpha = compute_S_alpha(t, alpha)
    
    # Handle non-finite S_alpha (if eta was near zero)
    if not np.isfinite(S_alpha):
         print(f"Warning: S_alpha is non-finite ({S_alpha}). Returning only initial position.")
         return [(0, 0, 0)]
    
    pos = [0, 0, 0]
    trajectory = [tuple(pos)]
    n_steps = 0
    exp_F2 = np.exp(F / 2)
    exp_negF2 = np.exp(-F / 2)
    A = 4 + exp_F2 + exp_negF2
    
    if A == 0: 
        print("Error: Normalization constant A is zero.")
        return trajectory

    probs = [exp_F2/A, exp_negF2/A, 1/A, 1/A, 1/A, 1/A]
    cum_probs = np.cumsum(probs)
    target_steps = S_alpha

    while n_steps < target_steps:
        r = np.random.rand()
        if r < cum_probs[0]:
            pos[0] += 1
        elif r < cum_probs[1]:
            pos[0] -= 1
        elif r < cum_probs[2]:
            pos[1] += 1
        elif r < cum_probs[3]:
            pos[1] -= 1
        elif r < cum_probs[4]:
            pos[2] += 1
        else:
            pos[2] -= 1
        
        pos[1] %= Ly
        pos[2] %= Lz
        
        n_steps += 1
        trajectory.append(tuple(pos))
            
    return trajectory


def plot_trajectory_3d(trajectory: list[tuple[int, int, int]]):
    """
    Plot a 3D particle trajectory.
    
    Parameters:
        trajectory (list of tuples): Full trajectory with each tuple (x, y, z).
    """
    xs = [pos[0] for pos in trajectory]
    ys = [pos[1] for pos in trajectory]
    zs = [pos[2] for pos in trajectory]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(xs, ys, zs, label="Trajectory", color='blue', linewidth=1)
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=50, label='Start')
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=50, marker='s', label='End')
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    ax.set_title('3D Particle Trajectory')
    ax.legend()
    plt.show()


def run_multiple_and_plot_final_histograms(
    t: float,
    alpha: float,
    F: float,
    Ly: int,
    Lz: int,
    num_trials: int
):
    """
    Run multiple random walk simulations and plot histograms of final positions.

    Parameters:
        t (float): Target time.
        alpha (float): Stability parameter.
        F (float): Bias force in the x-direction.
        Ly (int): System size in Y with periodic boundary conditions.
        Lz (int): System size in Z with periodic boundary conditions.
        num_trials (int): Number of independent simulations to run.
    """
    final_x, final_y, final_z = [], [], []

    for i in range(num_trials):
        traj = run_single_trajectory(t, alpha, F, Ly, Lz)
        x, y, z = traj[-1]
        final_x.append(x)
        final_y.append(y)
        final_z.append(z)

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(final_x, bins=50, density=True, color='skyblue', edgecolor='black')
    axes[0].set_title("Final X Positions")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Probability Density")

    axes[1].hist(final_y, bins=Ly, range=(0, Ly), density=True, color='salmon', edgecolor='black')
    axes[1].set_title("Final Y Positions")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Probability Density")

    axes[2].hist(final_z, bins=Lz, range=(0, Lz), density=True, color='lightgreen', edgecolor='black')
    axes[2].set_title("Final Z Positions")
    axes[2].set_xlabel("z")
    axes[2].set_ylabel("Probability Density")

    plt.suptitle(f"Final Position Histograms over {num_trials} Simulations")
    plt.tight_layout()
    plt.show()


def calculate_mean_final_position(t: float, alpha: float, F: float, Ly: int, Lz: int, num_sims: int) -> tuple[float, float, float]:
    """
    Run multiple 3D random walk simulations and compute the average final position.
    
    Parameters:
        t (float): Target laboratory time used to compute S_alpha.
        alpha (float): Stability parameter for the Lévy distribution.
        F (float): Bias force in the x-direction.
        Ly (int): System size in y (with periodic boundary conditions).
        Lz (int): System size in z (with periodic boundary conditions).
        num_sims (int): Number of independent simulations to run.
    
    Returns:
        (mean_x, mean_y, mean_z): Average final x, y, z positions over all simulations.
    """
    sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
    
    for i in range(num_sims):
        trajectory = run_single_trajectory(t, alpha, F, Ly, Lz)
        final_pos = trajectory[-1]
        x, y, z = final_pos
        sum_x += x
        sum_y += y
        sum_z += z
        
    mean_x = sum_x / num_sims
    mean_y = sum_y / num_sims
    mean_z = sum_z / num_sims
    
    return mean_x, mean_y, mean_z


def plot_mean_moment_vs_time_S_alpha(t_values: list[float], alpha: float, F: float, Ly: int, Lz: int, num_sims: int):
    """
    For a range of target times, compute the mean final position from multiple 
    simulations and plot the first moment (<X>, <Y>, <Z>) versus target time t on a log-log scale.
    
    Parameters:
        t_values (list of float): List or array of target times.
        alpha (float): Stability parameter for the Lévy distribution.
        F (float): Bias force in the x-direction.
        Ly (int): System size in y (with periodic boundary conditions).
        Lz (int): System size in z (with periodic boundary conditions).
        num_sims (int): Number of simulations per target time.
    """
    mean_x_vals = []
    mean_y_vals = []
    mean_z_vals = []
    
    for t in t_values:
        mean_x, mean_y, mean_z = calculate_mean_final_position(t, alpha, F, Ly, Lz, num_sims)
        mean_x_vals.append(mean_x)
        mean_y_vals.append(mean_y)
        mean_z_vals.append(mean_z)
    
    plt.figure(figsize=(10, 6))
    # We assume all moments are non-negative or that plotting absolute values makes sense.
    # If negative values occur, consider using a symmetric log scale.
    plt.loglog(t_values, mean_x_vals, 'o-', label='<X>', markersize=5)
    plt.loglog(t_values, mean_y_vals, 's-', label='<Y>', markersize=5)
    plt.loglog(t_values, mean_z_vals, '^-', label='<Z>', markersize=5)
    
    plt.xlabel("Target Time t")
    plt.ylabel("Mean Final Position")
    plt.title("Mean Final Position vs. Target Time t (S_α method)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()






def main():
    while True:
        print("\nMenu:")
        print("1. Validate eta generation (Plot Histogram vs PDF for alpha=0.5)")
        print("2. Run and Plot Single Trajectory")
        print("3. Run Multiple Trajectories & Plot Final Position Histograms")
        print("4. Plot Mean Final Position vs. Target Time")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            run_eta_validation_plot()

        elif choice == '2':
            print("Running and plotting a single trajectory...")
            t = 100.0
            alpha = 0.8
            F = 0.5
            Ly = 30
            Lz = 30
            trajectory = run_single_trajectory(t, alpha, F, Ly, Lz)
            plot_trajectory_3d(trajectory)

        elif choice == '3':
            t = 100.0
            alpha = 0.8
            F = 0.5
            Ly = 30
            Lz = 30
            num_trials = 1000
            print(f"Running {num_trials} simulations for final position histograms...")
            run_multiple_and_plot_final_histograms(t, alpha, F, Ly, Lz, num_trials)

        elif choice == '4':
            # Define a range of target times (example values)
            t_values = np.logspace(1, 3, num=10)  # From t=10 to t=1000
            alpha = 0.5
            F = 0.5
            Ly = 10
            Lz = 10
            num_sims = 100_000  # Simulations per t value
            print("Calculating mean final positions over a range of target times...")
            plot_mean_moment_vs_time_S_alpha(t_values, alpha, F, Ly, Lz, num_sims)

        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
