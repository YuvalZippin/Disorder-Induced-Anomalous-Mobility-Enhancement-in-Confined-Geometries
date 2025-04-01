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

def levy_pdf_alpha_half(x):
    """
    Compute the theoretical Lévy PDF for α = 1/2 as a function of x.
    
    Parameters:
        x (float or np.array): The random variable (must be positive).
    
    Returns:
        float or np.array: PDF values for the standard Lévy(1/2) distribution (scale=1, location=0).
    """
    # Ensure x>0
    return (1.0 / (2.0 * np.sqrt(np.pi))) * (x ** -1.5) * np.exp(-1.0 / (4.0 * x))

def simulate_and_plot_S_alpha_histogram(alpha_val: float,
                                        t_lab: float,
                                        num_samples: int,
                                        t_start: float,
                                        t_final: float,
                                        bins: int = 100):
    """
    1) Generate 'num_samples' of S_alpha for a single laboratory time 't_lab'.
    2) Build a histogram of S_alpha in the range [t_start, t_final].
    3) If alpha_val == 0.5, overlay the exact Lévy(1/2) PDF on the same plot.

    Parameters:
        alpha_val (float): Stability parameter α.
        t_lab (float): The fixed laboratory time t at which we simulate S_alpha.
        num_samples (int): Number of samples to generate for S_alpha.
        t_start (float): Lower bound for histogram & PDF plot (x-axis).
        t_final (float): Upper bound for histogram & PDF plot (x-axis).
        bins (int): Number of histogram bins.
    """
    # Generate samples
    samples = np.array([compute_S_alpha(t_lab, alpha_val) for _ in range(num_samples)])
    
    # Build the histogram (density=True to compare directly with a PDF)
    hist_vals, bin_edges = np.histogram(
        samples,
        range=(t_start, t_final),
        bins=bins,
        density=True
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist_vals, 
            width=(bin_edges[1] - bin_edges[0]),
            alpha=0.6, label="Simulated $S_{\\alpha}$")
    
    # If alpha=1/2, overlay the exact Lévy(1/2) PDF in the same x-range
    if abs(alpha_val - 0.5) < 1e-14:
        x_pdf = np.linspace(t_start, t_final, 200)
        pdf_vals = levy_pdf_alpha_half(x_pdf)
        plt.plot(x_pdf, pdf_vals, 'r--', linewidth=2, label="Theoretical Lévy PDF (α=1/2)")
    
    plt.xlim(t_start, t_final)
    plt.title(f"Histogram of $S_\\alpha$ (α={alpha_val}, t={t_lab})")
    plt.xlabel("$S_\\alpha$")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Example usage:
    alpha_val = 0.5   # If exactly 0.5, we'll overlay the Lévy(1/2) PDF
    t_lab = 1.0       # Laboratory time at which we generate S_alpha
    num_samples = 200000
    t_start = 0.0     # Lower bound for the histogram
    t_final = 10.0    # Upper bound for the histogram
    bins = 200        # Number of bins in the histogram
    
    simulate_and_plot_S_alpha_histogram(
        alpha_val, t_lab, num_samples, t_start, t_final, bins
    )

if __name__ == "__main__":
    main()
