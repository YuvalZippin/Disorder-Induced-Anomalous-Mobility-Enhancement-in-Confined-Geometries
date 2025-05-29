import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import curve_fit
import multiprocessing as mp

#? Mathematical functions:
def generate_eta(alpha: float) -> float:
    """
    Generate one-sided Lévy stable random variable η using the Chambers et al. method.
    """
    theta = np.pi * np.random.uniform(0, 1)  # θ ~ U(0, π)
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

def power_law(x, A, beta):
    return A * x**beta


#? Simulation functions:
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

# !!!

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

# !!!

def plot_trajectory_3d(trajectory: list[tuple[int, int, int]]) -> None:
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

# !!!

def run_multiple_and_plot_final_histograms(
    t: float,
    alpha: float,
    F: float,
    Ly: int,
    Lz: int,
    num_trials: int
) -> None:
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

# !!!

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

# !!!

def plot_mean_moment_vs_time_S_alpha(t_values: list[float], alpha: float, F: float, Ly: int, Lz: int, num_sims: int) -> None:
    """
    For a range of target times, compute the mean final position from multiple 
    simulations, fit a power law g(t)=A*t^beta to the data, and plot the first moment 
    (<X>, <Y>, <Z>) versus target time t on a log-log scale.

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
    
    # Loop over target times and collect mean final positions.
    for t in t_values:
        mean_x, mean_y, mean_z = calculate_mean_final_position(t, alpha, F, Ly, Lz, num_sims)
        mean_x_vals.append(mean_x)
        mean_y_vals.append(mean_y)
        mean_z_vals.append(mean_z)
    
    t_array = np.array(t_values)
    mean_x_vals = np.array(mean_x_vals)
    mean_y_vals = np.array(mean_y_vals)
    mean_z_vals = np.array(mean_z_vals)
    
    # Perform curve fitting for X using the raw data.
    try:
        popt_x, _ = curve_fit(power_law, t_array, mean_x_vals, maxfev=10000)
        A_x, beta_x = popt_x
        fitted_x = power_law(t_array, A_x, beta_x)
        print(f"Fit for <X>: A = {A_x:.3e}, beta = {beta_x:.3f}")
    except Exception as e:
        print("Fit for <X> failed:", e)
        A_x, beta_x = None, None
        fitted_x = None

    # For Y and Z, fit the absolute values in case they are near zero.
    try:
        popt_y, _ = curve_fit(power_law, t_array, np.abs(mean_y_vals), maxfev=10000)
        A_y, beta_y = popt_y
        fitted_y = power_law(t_array, A_y, beta_y)
        print(f"Fit for <Y> (abs): A = {A_y:.3e}, beta = {beta_y:.3f}")
    except Exception as e:
        print("Fit for <Y> failed:", e)
        A_y, beta_y = None, None
        fitted_y = None

    try:
        popt_z, _ = curve_fit(power_law, t_array, np.abs(mean_z_vals), maxfev=10000)
        A_z, beta_z = popt_z
        fitted_z = power_law(t_array, A_z, beta_z)
        print(f"Fit for <Z> (abs): A = {A_z:.3e}, beta = {beta_z:.3f}")
    except Exception as e:
        print("Fit for <Z> failed:", e)
        A_z, beta_z = None, None
        fitted_z = None

    # Generate the log-log plot.
    plt.figure(figsize=(10, 6))
    plt.loglog(t_array, mean_x_vals, 'o', color='blue', label='<X> simulation')
    plt.loglog(t_array, np.abs(mean_y_vals), 's', color='red', label='<Y> simulation (abs)')
    plt.loglog(t_array, np.abs(mean_z_vals), '^', color='green', label='<Z> simulation (abs)')
    
    # Overlay fitted curves if available.
    if fitted_x is not None:
        plt.loglog(t_array, fitted_x, '-', color='blue',
                   label=f'<X> fit (β≈{beta_x:.2f})')
    if fitted_y is not None:
        plt.loglog(t_array, fitted_y, '-', color='red',
                   label=f'<Y> fit (β≈{beta_y:.2f})')
    if fitted_z is not None:
        plt.loglog(t_array, fitted_z, '-', color='green',
                   label=f'<Z> fit (β≈{beta_z:.2f})')
    
    plt.xlabel("Target Time t")
    plt.ylabel("Mean Final Position (absolute for Y and Z)")
    plt.title("Mean Final Position vs. Target Time t (Power-law Fit)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# !!!

def calculate_mean_squared_displacement(t: float, alpha: float, F: float, Ly: int, Lz: int, num_sims: int) -> tuple[float, float, float]:
    """
    Run multiple simulations and compute the Mean Squared Displacement (MSD) components.
    
    Parameters:
        t (float): Target time.
        alpha (float): Stability parameter.
        F (float): Bias in the x-direction.
        Ly (int): System size in Y.
        Lz (int): System size in Z.
        num_sims (int): Number of simulations.
    
    Returns:
        (mean_x_sq, mean_y_sq, mean_z_sq): Mean of the squared final positions in x, y, z.
    """
    sum_x2, sum_y2, sum_z2 = 0.0, 0.0, 0.0
    valid_runs = 0
    
    for _ in range(num_sims):
        traj = run_single_trajectory(t, alpha, F, Ly, Lz)
        # If trajectory is empty or invalid, skip.
        if not traj:
            continue
        x, y, z = traj[-1]
        sum_x2 += x**2
        sum_y2 += y**2
        sum_z2 += z**2
        valid_runs += 1
    
    if valid_runs == 0:
        return float('nan'), float('nan'), float('nan')
    
    mean_x_sq = sum_x2 / valid_runs
    mean_y_sq = sum_y2 / valid_runs
    mean_z_sq = sum_z2 / valid_runs
    
    return mean_x_sq, mean_y_sq, mean_z_sq

# !!!

def plot_msd_vs_time_S_alpha(t_values: list[float], alpha: float, F: float, Ly: int, Lz: int, num_sims: int) -> None:
    """
    Compute the MSD components (<X^2>, <Y^2>, <Z^2>) over a range of target times and plot them
    on a log-log scale. Also perform a power-law fit g(t) = A*t^beta for each component.
    
    Parameters:
        t_values (list of float): List/array of target times.
        alpha (float): Stability parameter.
        F (float): Bias force in the x-direction.
        Ly (int): System size in Y.
        Lz (int): System size in Z.
        num_sims (int): Number of simulations per target time.
    """
    mean_x2_vals = []
    mean_y2_vals = []
    mean_z2_vals = []
    
    for t in t_values:
        mean_x2, mean_y2, mean_z2 = calculate_mean_squared_displacement(t, alpha, F, Ly, Lz, num_sims)
        mean_x2_vals.append(mean_x2)
        mean_y2_vals.append(mean_y2)
        mean_z2_vals.append(mean_z2)
    
    t_arr = np.array(t_values)
    mean_x2_arr = np.array(mean_x2_vals)
    mean_y2_arr = np.array(mean_y2_vals)
    mean_z2_arr = np.array(mean_z2_vals)
    
    # Perform power-law fits on each MSD component.
    try:
        popt_x2, _ = curve_fit(power_law, t_arr, mean_x2_arr, maxfev=10000)
        A_x2, beta_x2 = popt_x2
        fitted_x2 = power_law(t_arr, A_x2, beta_x2)
        print(f"MSD <X^2> fit: A = {A_x2:.3e}, beta = {beta_x2:.3f}")
    except Exception as e:
        print("Fit for MSD <X^2> failed:", e)
        A_x2, beta_x2 = None, None
        fitted_x2 = None

    try:
        popt_y2, _ = curve_fit(power_law, t_arr, mean_y2_arr, maxfev=10000)
        A_y2, beta_y2 = popt_y2
        fitted_y2 = power_law(t_arr, A_y2, beta_y2)
        print(f"MSD <Y^2> fit: A = {A_y2:.3e}, beta = {beta_y2:.3f}")
    except Exception as e:
        print("Fit for MSD <Y^2> failed:", e)
        A_y2, beta_y2 = None, None
        fitted_y2 = None

    try:
        popt_z2, _ = curve_fit(power_law, t_arr, mean_z2_arr, maxfev=10000)
        A_z2, beta_z2 = popt_z2
        fitted_z2 = power_law(t_arr, A_z2, beta_z2)
        print(f"MSD <Z^2> fit: A = {A_z2:.3e}, beta = {beta_z2:.3f}")
    except Exception as e:
        print("Fit for MSD <Z^2> failed:", e)
        A_z2, beta_z2 = None, None
        fitted_z2 = None

    # Plot the data and fitted curves on a log-log plot.
    plt.figure(figsize=(10, 6))
    plt.loglog(t_arr, mean_x2_arr, 'o', color='blue', label='MSD <X^2> data')
    plt.loglog(t_arr, mean_y2_arr, 's', color='red', label='MSD <Y^2> data')
    plt.loglog(t_arr, mean_z2_arr, '^', color='green', label='MSD <Z^2> data')
    
    if fitted_x2 is not None:
        plt.loglog(t_arr, fitted_x2, '-', color='blue', label=f'<X^2> fit (β≈{beta_x2:.2f})')
    if fitted_y2 is not None:
        plt.loglog(t_arr, fitted_y2, '-', color='red', label=f'<Y^2> fit (β≈{beta_y2:.2f})')
    if fitted_z2 is not None:
        plt.loglog(t_arr, fitted_z2, '-', color='green', label=f'<Z^2> fit (β≈{beta_z2:.2f})')
    
    plt.xlabel("Target Time t")
    plt.ylabel("MSD Component")
    plt.title("MSD Components vs. Target Time t (Power-law Fit)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# !!!

def calculate_fitted_A(
    t_values: list[float],
    alpha: float,
    F: float,
    L: int,
    num_sims: int,
    moment_type: str
) -> float:
    """
    For a fixed L and moment_type ('first' for <X>, 'second' for <X^2>),
    run num_sims trajectories at each t in t_values, compute the mean moment,
    and fit A * t**beta to those means. Return the fitted prefactor A.
    """
    means = []
    for t in t_values:
        # collect X or X^2 over num_sims runs
        vals = []
        for _ in range(num_sims):
            traj = run_single_trajectory(t, alpha, F, L, L)
            x = traj[-1][0]
            vals.append(x if moment_type=='first' else x**2)
        means.append(np.mean(vals))
        
    t_arr = np.array(t_values)
    y = np.abs(np.array(means))  # ensure positive for fitting
    try:
        (A_fit, _), _ = curve_fit(
            power_law,
            t_arr,
            y,
            maxfev=10_000
        )
    except Exception as e:
        print(f"  [L={L}, moment={moment_type}] fit failed:", e)
        return np.nan
    return A_fit

def compute_mean_A_for_L(
    L: int,
    alpha: float,
    F: float,
    num_sims: int,
    num_tests: int,
    t_values: list[float],
    moment_type: str
) -> float:
    """
    Repeat calculate_fitted_A num_tests times for given L and moment_type,
    and return the average of all non-NaN A values.
    """
    A_list = []
    for _ in range(num_tests):
        A = calculate_fitted_A(t_values, alpha, F, L, num_sims, moment_type)
        if not np.isnan(A):
            A_list.append(A)
    return float(np.mean(A_list)) if A_list else np.nan

def plot_A_vs_L(
    L_values: list[int],
    alpha: float,
    F: float,
    num_sims: int,
    num_tests: int,
    t_values: list[float]
):
    """
    For each L in L_values, compute the average fitted prefactor A for both
    <X> (moment_type='first') and <X^2> (moment_type='second'), in parallel.
    Then plot A vs L on linear (left) and log-log (right) scales.
    """
    # Prepare argument tuples for pool
    args_first  = [(L, alpha, F, num_sims, num_tests, t_values, 'first')  for L in L_values]
    args_second = [(L, alpha, F, num_sims, num_tests, t_values, 'second') for L in L_values]
    
    with mp.Pool(mp.cpu_count()) as pool:
        A_first_list  = pool.starmap(compute_mean_A_for_L, args_first)
        A_second_list = pool.starmap(compute_mean_A_for_L, args_second)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(L_values,   A_first_list,  'o-', label=r'$⟨X⟩$ prefactor')
    axes[0].plot(L_values,   A_second_list, 's-', label=r'$⟨X^2⟩$ prefactor')
    axes[0].set_xlabel('L (Ly = Lz)')
    axes[0].set_ylabel('Fitted prefactor A')
    axes[0].set_title('A vs L (linear scale)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Log–log scale
    axes[1].loglog(L_values,   A_first_list,  'o-', label=r'$⟨X⟩$ prefactor')
    axes[1].loglog(L_values,   A_second_list, 's-', label=r'$⟨X^2⟩$ prefactor')
    axes[1].set_xlabel('L (Ly = Lz)')
    axes[1].set_ylabel('Fitted prefactor A')
    axes[1].set_title('A vs L (log–log scale)')
    axes[1].legend()
    axes[1].grid(True, which='both', ls='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# !!!



#? Main function to run the program:
def main():

    ALPHA = 0.5
    FORCE = 0.1
    NUM_SIMS = 15_000
    WIDTH = 30
    HIGHT = 30

    while True:
        print("\nMenu:")
        print("1. Validate eta generation (Plot Histogram vs PDF for alpha=0.5)")
        print("2. Run and Plot Single Trajectory")
        print("3. Run Multiple Trajectories & Plot Final Position Histograms")
        print("4. Plot Mean Final Position vs. Target Time")
        print("5. Plot MSD (Mean Squared Displacement) vs. Target Time")
        print("6. Plot A vs L (Fitted Prefactor)")


        print("9. Exit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            run_eta_validation_plot()

        elif choice == '2':
            print("Running and plotting a single trajectory...")
            t = 100.0
            alpha = ALPHA
            F = FORCE
            Ly = WIDTH
            Lz = HIGHT
            trajectory = run_single_trajectory(t, alpha, F, Ly, Lz)
            plot_trajectory_3d(trajectory)

        elif choice == '3':
            print("Running multiple trajectories for final position histograms...")
            t = 100.0
            alpha = ALPHA
            F = FORCE
            Ly = WIDTH
            Lz = HIGHT
            num_trials = NUM_SIMS
            run_multiple_and_plot_final_histograms(t, alpha, F, Ly, Lz, num_trials)

        elif choice == '4':
            print("Plotting mean final position vs target time...")
            t_values = np.logspace(1, 4, num=10)  # e.g., t from 10 to 1000
            alpha = ALPHA
            F = FORCE
            Ly = WIDTH
            Lz = HIGHT
            num_sims = NUM_SIMS
            plot_mean_moment_vs_time_S_alpha(t_values, alpha, F, Ly, Lz, num_sims)

        elif choice == '5':
            print("Plotting MSD vs target time...")
            t_values = np.logspace(1, 7, num=100)  # t values from 10 to 1000
            alpha = ALPHA
            F = FORCE
            Ly = WIDTH
            Lz = HIGHT
            num_sims = NUM_SIMS
            plot_msd_vs_time_S_alpha(t_values, alpha, F, Ly, Lz, num_sims)

        elif choice == '6':
            L_vals    = np.arange(0,10,1)  # e.g. L = 10,20,…,100
            alpha     = ALPHA
            F         = FORCE
            num_sims  = NUM_SIMS
            num_tests = 10
            t_vals    = np.logspace(1, 3, 8)   # eight t’s from 10 to 1000

            plot_A_vs_L(L_vals, alpha, F, num_sims, num_tests, t_vals)


        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()