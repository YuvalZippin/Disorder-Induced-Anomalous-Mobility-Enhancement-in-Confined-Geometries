import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import multiprocessing as mp
# from functools import partial # Option for multiprocessing if needed

# -----------------------------------------------------------------------------
# Lévy‐stable sampling (Chambers et al.), vectorized
# -----------------------------------------------------------------------------
def generate_eta_vector(alpha: float, size: int) -> np.ndarray:
    """
    Generate `size` one‐sided Lévy‐stable η samples for stability 0<α<1.
    (Unchanged from previous version)
    """
    # θ ~ Uniform(0,π), W = -ln(U)
    theta = np.random.uniform(0, np.pi, size)
    W = -np.log(np.random.uniform(0, 1, size))
    sin_theta = np.sin(theta)
    # Prevent division by zero or invalid values in power for edge cases
    sin_theta[sin_theta == 0] = 1e-12 # Add small epsilon if sin(theta) is zero
    alpha_term_base = np.sin(alpha * theta)
    alpha_term_base[alpha_term_base <= 0] = 1e-12 # Ensure base for power is positive

    a_theta = (
        np.sin((1 - alpha) * theta)
        * (alpha_term_base ** (alpha / (1 - alpha)))
        / (sin_theta ** (1 / (1 - alpha)))
    )
    # Handle potential infinities or NaNs resulting from edge cases
    a_theta = np.nan_to_num(a_theta, nan=1e12, posinf=1e12, neginf=-1e12)

    # Ensure W is not zero before division
    W[W == 0] = 1e-12

    eta_base = (a_theta / W)
    # Prevent negative base for fractional power if 1-alpha is negative (alpha > 1, not expected here but good practice)
    # eta_base[eta_base <= 0] = 1e-12 # Ensure base is positive

    eta = eta_base ** ((1 - alpha) / alpha)
    eta = np.nan_to_num(eta, nan=1e12, posinf=1e12, neginf=-1e12) # Final check
    # Ensure eta is positive, as it represents a scaling factor for time/steps
    eta[eta <= 0] = 1e-12
    return eta


def compute_S_alpha_vector(t: float, alpha: float, size: int) -> np.ndarray:
    """
    Compute S_α = (t/η)^α for an array of η’s.
    (Unchanged from previous version)
    """
    eta = generate_eta_vector(alpha, size)
    # Ensure eta is not zero to prevent division by zero
    eta[eta <= 0] = 1e-12 # Use a small positive number instead of zero
    S_alpha = (t / eta) ** alpha
    # Handle potential NaNs or infinities if t or eta were problematic
    S_alpha = np.nan_to_num(S_alpha, nan=0.0, posinf=np.finfo(np.float64).max)
     # Ensure S_alpha is non-negative as it relates to number of steps
    S_alpha[S_alpha < 0] = 0
    return S_alpha

# -----------------------------------------------------------------------------
# Single‐trial 3D final‐position simulator
# -----------------------------------------------------------------------------
def simulate_final_position(args):
    """
    Simulate one biased 3D walk for a given S_alpha, bias F, width W, height H.
    Applies periodic boundary conditions in Y and Z.
    Returns final position [x, y, z].
    """
    S_alpha, F, W, H = args
    n_steps = int(np.floor(S_alpha))

    if n_steps <= 0:
        return np.array([0.0, 0.0, 0.0]) # No steps taken if S_alpha is too small

    # Position vector [x, y, z] - Start at origin
    pos = np.zeros(3, dtype=float)

    # Probabilities for ±X, ±Y, ±Z steps
    exp_p = np.exp( F/2 )
    exp_m = np.exp(-F/2 )
    # Assuming equal probability (1) for lateral moves before normalization
    p_lat = 1.0
    A = exp_p + exp_m + 4 * p_lat # Normalization constant

    p_plus_x = exp_p / A
    p_minus_x = exp_m / A
    p_plus_y = p_lat / A
    p_minus_y = p_lat / A
    p_plus_z = p_lat / A
    p_minus_z = p_lat / A

    # Cumulative probabilities for selecting direction
    cum_prob = np.cumsum([
        p_plus_x, p_minus_x,
        p_plus_y, p_minus_y,
        p_plus_z, p_minus_z
    ])

    # Generate n_steps random numbers for direction choice
    r = np.random.rand(n_steps)

    # Simulate steps
    for i in range(n_steps):
        ri = r[i]
        if ri < cum_prob[0]:   # +X
            pos[0] += 1
        elif ri < cum_prob[1]: # -X
            pos[0] -= 1
        elif ri < cum_prob[2]: # +Y
            pos[1] += 1
        elif ri < cum_prob[3]: # -Y
            pos[1] -= 1
        elif ri < cum_prob[4]: # +Z
            pos[2] += 1
        else:                  # -Z (ri < cum_prob[5])
            pos[2] -= 1

        # Apply periodic boundary conditions for Y and Z
        # Using modulo arithmetic correctly handles positive and negative coordinates
        if W > 0: # Avoid modulo by zero if W=0
             pos[1] = pos[1] % W
        if H > 0: # Avoid modulo by zero if H=0
             pos[2] = pos[2] % H

    return pos # Return final [x, y, z]

# -----------------------------------------------------------------------------
# Compute ⟨X⟩ and ⟨X²⟩ over num_sims for one t in 3D
# -----------------------------------------------------------------------------
def compute_moments_for_t(t: float, alpha: float, F: float, W: int, H: int, num_sims: int, pool=None):
    """
    Returns mean_x, mean_x2 for given t by running num_sims 3D trials.
    """
    # 1) sample S_alpha for each trial
    S_vec = compute_S_alpha_vector(t, alpha, num_sims)

    # Prepare arguments for parallel processing
    # Each worker needs (S_alpha, F, W, H)
    sim_args = zip(S_vec, [F]*num_sims, [W]*num_sims, [H]*num_sims)

    # 2) simulate final_position in parallel
    if pool is None:
        # Serial execution if no pool provided
        final_positions = [simulate_final_position(args) for args in sim_args]
    else:
        # Parallel execution using the pool
        final_positions = pool.map(simulate_final_position, sim_args)

    # Extract only the final X coordinates
    final_x = np.array([pos[0] for pos in final_positions], dtype=float)

    # Compute moments for X
    mean_x = final_x.mean()
    mean_x2 = (final_x**2).mean()

    # Optional: could also compute moments for Y and Z if needed
    # final_y = np.array([pos[1] for pos in final_positions], dtype=float)
    # final_z = np.array([pos[2] for pos in final_positions], dtype=float)
    # mean_y = final_y.mean()
    # mean_z = final_z.mean()
    # mean_y2 = (final_y**2).mean()
    # mean_z2 = (final_z**2).mean()

    return mean_x, mean_x2

# -----------------------------------------------------------------------------
# Power law (Unchanged)
# -----------------------------------------------------------------------------
def power_law(x, A, beta):
    return A * x**beta

# -----------------------------------------------------------------------------
# Plotting routines (Unchanged)
# -----------------------------------------------------------------------------
def plot_mean_x_vs_time(t_values, mean_x_vals, popt):
    A, beta = popt
    plt.figure(figsize=(8,6))
    plt.loglog(t_values, mean_x_vals, 'o', label=r'$\langle X(t) \rangle$ data')
    # Plot absolute value for log-log scale if mean can be negative
    plt.loglog(t_values, np.abs(power_law(t_values, A, beta)), '-',
               label=fr'fit: $A \approx {A:.2e}, \beta \approx {beta:.2f}$')
    plt.xlabel('Time (t)')
    plt.ylabel(r'$\langle X(t) \rangle$')
    plt.title('Mean Displacement in X vs Time')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_msd_x_vs_time(t_values, mean_x2_vals, popt):
    A, beta = popt
    plt.figure(figsize=(8,6))
    plt.loglog(t_values, mean_x2_vals, 'o', label=r'$\langle X^2(t) \rangle$ data (MSD)')
    plt.loglog(t_values, power_law(t_values, A, beta), '-',
               label=fr'fit: $A \approx {A:.2e}, \beta \approx {beta:.2f}$')
    plt.xlabel('Time (t)')
    plt.ylabel(r'$\langle X^2(t) \rangle$')
    plt.title('Mean Squared Displacement in X vs Time')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Batch study: generate and fit moments over a grid of t (Updated args)
# -----------------------------------------------------------------------------
def study_power_law(alpha, F, W, H, num_sims, t_values, n_processes=None):
    """
    Compute ⟨X⟩(t) and ⟨X²⟩(t) over t_values for the 3D model, fit power laws, and plot.
    """
    # set up multiprocessing pool
    pool = mp.Pool(n_processes) if n_processes else None
    print(f"Starting simulation study with alpha={alpha}, F={F}, W={W}, H={H}, num_sims={num_sims}")
    print(f"Using {pool._processes if pool else 1} processes.") # Correct way to get pool size

    mean_x_list = []
    mean_x2_list = []

    # Use a temporary list to store valid t_values and results,
    # in case some t values lead to issues (e.g., zero mean for fits)
    valid_t = []
    valid_mean_x = []
    valid_mean_x2 = []

    for i, t in enumerate(t_values):
        print(f"Simulating for t = {t:.2f} ({i+1}/{len(t_values)})...")
        try:
             mx, mx2 = compute_moments_for_t(t, alpha, F, W, H, num_sims, pool)
             print(f"  -> <X> = {mx:.4f}, <X^2> = {mx2:.4f}")
             # Only include if results are suitable for log-log fitting (positive)
             # Need to handle potential zero or negative <X> for the first moment fit
             valid_t.append(t)
             valid_mean_x.append(mx) # Keep original mean_x
             if mx2 > 1e-9: # Ensure MSD is positive for log-log fit
                 valid_mean_x2.append(mx2)
             else:
                 print(f"  -> Warning: MSD near zero ({mx2:.2e}) for t={t}, might affect <X^2> fit.")
                 # Decide how to handle: append a small value, or skip? Let's append for now.
                 valid_mean_x2.append(1e-9)

        except Exception as e:
             print(f"  -> Error during simulation for t={t}: {e}")
             # Optionally skip this t value or handle error differently

    if pool:
        pool.close()
        pool.join()

    print("Simulations complete. Performing power law fits...")

    # Convert lists to numpy arrays for fitting
    t_fit = np.array(valid_t)
    mean_x_fit = np.array(valid_mean_x)
    mean_x2_fit = np.array(valid_mean_x2)


    if len(t_fit) < 2:
        print("Not enough valid data points to perform fits.")
        return

    # Fit ⟨X⟩ - Use absolute value for fitting if mean can be negative, but plot original
    # Need to be careful if mean_x_fit crosses zero. A simple power law A*t^beta might not fit well.
    # Consider fitting only positive or negative parts, or use a different model if sign changes.
    # For now, fit assuming mean_x stays on one side or use abs value.
    try:
        # Fit abs(<X>) vs t
        popt1, pcov1 = curve_fit(power_law, t_fit, np.abs(mean_x_fit), maxfev=10000, p0=[1, alpha]) # Added guess p0
        print(f"Fit for |<X>|: A = {popt1[0]:.3e}, β = {popt1[1]:.3f}")
        plot_mean_x_vs_time(t_fit, mean_x_fit, popt1) # Plot original data with fit line
    except Exception as e:
        print(f"Could not fit <X> vs t: {e}")


    # Fit ⟨X²⟩ vs t (MSD should be positive)
    try:
        # Only fit if mean_x2_fit contains positive values
        if np.all(mean_x2_fit > 0):
             popt2, pcov2 = curve_fit(power_law, t_fit, mean_x2_fit, maxfev=10000, p0=[1, 2*alpha]) # Added guess p0
             print(f"Fit for <X^2>: A = {popt2[0]:.3e}, β = {popt2[1]:.3f}")
             plot_msd_x_vs_time (t_fit, mean_x2_fit, popt2)
        else:
             print("Cannot fit <X^2> vs t because of non-positive values.")
    except Exception as e:
        print(f"Could not fit <X^2> vs t: {e}")


# -----------------------------------------------------------------------------
# Main menu (Updated params)
# -----------------------------------------------------------------------------
def main():
    # --- Simulation Parameters ---
    ALPHA = 0.5       # Lévy stability parameter (0 < alpha < 1 for one-sided eta generator used)
                      # Note: For symmetric Levy flights alpha can be up to 2.
                      # If alpha >= 1, generate_eta_vector needs modification or replacement.
    FORCE = 0.1       # Bias force along X-axis
    WIDTH = 10        # System width (Ly = W) for periodic BC in Y
    HEIGHT = 10       # System height (Lz = H) for periodic BC in Z
    NUM_SIMS = 15_000   # Number of simulations per time point
    # Choose time grid carefully for log-log plots
    t_values = np.logspace(1, 10, 15) # Example: 15 points from t=10 to t=10000

    # Set number of processes for parallelization (optional)
    # Uses mp.cpu_count() by default if set to None in study_power_law call.
    N_PROCESSES = mp.cpu_count()
    # N_PROCESSES = 4 # Or set manually

    while True:
        print("\n--- Simulation Menu ---")
        print(f"Params: alpha={ALPHA}, F={FORCE}, W={WIDTH}, H={HEIGHT}, N_sims={NUM_SIMS}")
        print("t values:", t_values)
        print("-----------------------")
        print("1. Study power-law of ⟨X⟩ and ⟨X²⟩ vs t")
        print("9. Exit")
        ch = input("Choice: ").strip()
        if ch == '1':
            study_power_law(ALPHA, FORCE, WIDTH, HEIGHT, NUM_SIMS, t_values, n_processes=N_PROCESSES)
        elif ch == '9':
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__ == '__main__':
    # Ensures multiprocessing works correctly when run as a script
    mp.freeze_support() # Good practice, esp. for Windows/macOS freezing executables
    main()