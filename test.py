import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# --- Existing functions ---

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

def run_eta_validation_plot(alpha: float = 0.5, N_samples: int = 100_000, hist_range=(0.001, 5.0), num_bins=200):
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

# --- New function for multiple trajectory simulations ---

def run_multiple_and_plot_final_histograms(t: float, alpha: float, F: float, Ly: int, Lz: int, num_trials: int):
    """
    Run multiple simulations to collect final positions and plot their histograms.
    
    Parameters:
        t (float): Laboratory time.
        alpha (float): Stability parameter.
        F (float): External field strength.
        Ly (int): System size in y-direction (PBC applied).
        Lz (int): System size in z-direction (PBC applied).
        num_trials (int): Number of simulations to run.
    """
    final_x = []
    final_y = []
    final_z = []

    print(f"\nRunning {num_trials} simulations to obtain final position histograms...")
    for _ in range(num_trials):
        trajectory = run_single_trajectory(t, alpha, F, Ly, Lz)
        # Get the last position (final position of the trajectory)
        final_pos = trajectory[-1]
        final_x.append(final_pos[0])
        final_y.append(final_pos[1])
        final_z.append(final_pos[2])
    
    # Create a figure with three subplots for x, y, and z histograms.
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].hist(final_x, bins=50, density=True, color='skyblue', edgecolor='k')
    axs[0].set_title('Final X Distribution')
    axs[0].set_xlabel('Final X')
    axs[0].set_ylabel('Probability Density')
    
    axs[1].hist(final_y, bins=50, density=True, color='lightgreen', edgecolor='k')
    axs[1].set_title('Final Y Distribution')
    axs[1].set_xlabel('Final Y')
    axs[1].set_ylabel('Probability Density')
    
    axs[2].hist(final_z, bins=50, density=True, color='salmon', edgecolor='k')
    axs[2].set_title('Final Z Distribution')
    axs[2].set_xlabel('Final Z')
    axs[2].set_ylabel('Probability Density')
    
    plt.tight_layout()
    plt.show()

# --- Updated main function with new menu option ---

def main():
    while True:
        print("\nMenu:")
        print("1. Validate eta generation (Plot Histogram vs PDF for alpha=0.5)")
        print("2. Run and Plot Single Trajectory")
        print("3. Run Multiple Trajectories & Plot Final Position Histograms")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == '1':
            run_eta_validation_plot()
        elif choice == '2':
            # Default parameters for a single trajectory simulation.
            t = 100.0
            alpha = 0.8
            F = 0.5
            Ly = 30
            Lz = 30
            print(f"\nRunning single trajectory with t={t}, alpha={alpha}, F={F}, Ly={Ly}, Lz={Lz}...")
            trajectory = run_single_trajectory(t, alpha, F, Ly, Lz)
            plot_trajectory_3d(trajectory)
        elif choice == '3':
            # Default parameters for multiple trajectory simulations.
            t = 100.0
            alpha = 0.8
            F = 0.5
            Ly = 30
            Lz = 30
            num_trials = 250_000
            run_multiple_and_plot_final_histograms(t, alpha, F, Ly, Lz, num_trials)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option (1, 2, 3, or 4).")

if __name__ == "__main__":
    main()
