import numpy as np
import matplotlib.pyplot as plt

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
    return (t / eta) ** alpha

def simulate_S_alpha_over_t(alpha_val: float,
                            t_start: float,
                            t_finish: float,
                            num_t_points: int,
                            num_samples_per_t: int,
                            gridsize: int = 50):
    """
    1) Generate (t, S_alpha) pairs over a range of t values.
    2) Plot a 2D color histogram/hexbin with:
       - X-axis = t
       - Y-axis = S_alpha
    3) If alpha=1/2, overlay the theoretical mode curve ~ (1/6)*sqrt(t).
    
    Parameters:
        alpha_val (float): Stability parameter α.
        t_start (float): Minimum t.
        t_finish (float): Maximum t.
        num_t_points (int): Number of distinct t values to sample between t_start and t_finish.
        num_samples_per_t (int): Number of S_alpha samples to generate for each t.
        gridsize (int): Resolution of the hexbin.
    """
    t_values = np.linspace(t_start, t_finish, num_t_points)

    # Collect all (t, S_alpha) pairs in lists:
    t_all = []
    S_all = []
    
    # Generate the samples
    for t in t_values:
        for _ in range(num_samples_per_t):
            s_val = compute_S_alpha(t, alpha_val)
            t_all.append(t)
            S_all.append(s_val)
    
    t_all = np.array(t_all)
    S_all = np.array(S_all)
    
    # Plot a 2D hexbin
    plt.figure(figsize=(9, 6))
    hb = plt.hexbin(t_all, S_all, gridsize=gridsize, cmap='viridis',
                    bins='log', mincnt=1)
    plt.colorbar(hb, label='log10(count)')
    
    plt.xlabel("Laboratory time t")
    plt.ylabel(r"$S_\alpha$")
    plt.title(rf"2D distribution of $S_\alpha$ vs. $t$,  $\alpha = {alpha_val}$")
    
    # If alpha = 1/2, overlay a theoretical mode curve:
    if abs(alpha_val - 0.5) < 1e-14:
        # The mode of standard Lévy(1/2) is at ~ 1/6 for scale=1,
        # so for S_alpha ~ sqrt(t / eta), it should scale like sqrt(t).
        # We'll overlay y = (1/6)*sqrt(t).
        t_curve = np.linspace(t_start, t_finish, 200)
        mode_curve = (1.0/6.0)*np.sqrt(t_curve)
        
        plt.plot(t_curve, mode_curve, 'r--', linewidth=2,
                 label="Theoretical mode ~ (1/6)√t")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Example usage:
    alpha_val = 0.5      # If = 1/2, we'll overlay the mode curve
    t_start = 1.0
    t_finish = 10.0
    num_t_points = 50
    num_samples_per_t = 2000
    
    simulate_S_alpha_over_t(alpha_val,
                            t_start,
                            t_finish,
                            num_t_points,
                            num_samples_per_t,
                            gridsize=60)

if __name__ == "__main__":
    main()
