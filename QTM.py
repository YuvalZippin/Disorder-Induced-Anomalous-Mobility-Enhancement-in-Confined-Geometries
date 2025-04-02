import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.5  # Stability parameter for Lévy process where 0 < α < 1

def generate_eta(alpha: float) -> float:
    """Generate one-sided Lévy stable random variable η using the correct formula."""
    theta = np.random.uniform(0, np.pi)  # θ ~ U(0, π)
    W = -np.log(np.random.uniform(0, 1))  # W = -ln(U), U ~ U(0,1)
    
    # Compute a(theta)
    a_theta = ((np.sin((1 - alpha) * theta) * (np.sin(alpha * theta) ** (alpha / (1 - alpha)))) /
               (np.sin(theta) ** (1 / (1 - alpha))))
    
    # Compute η
    eta = (a_theta / W) ** ((1 - alpha) / alpha)
    return eta

def compute_S_alpha(t: float, alpha: float) -> float:
    """
    Compute S_α for a given time t and alpha.
    
    Parameters:
        t (float): The time variable.
        alpha (float): Stability parameter (0 < alpha < 1).
    
    Returns:
        float: Computed S_alpha.
    """
    eta = generate_eta(alpha)  # Generate η using the existing function
    S_alpha = (t / eta) ** alpha  # Compute S_alpha using the general formula
    return S_alpha

def levy_pdf_alpha_half(t):
    """
    Compute the theoretical Lévy PDF for α = 1/2 as a function of t.
    
    Parameters:
        t (float or np.array): The time variable (must be positive).
    
    Returns:
        float or np.array: PDF values.
    """
    return (1 / (2 * np.sqrt(np.pi))) * t ** (-3/2) * np.exp(-1 / (4 * t))

def plot_histogram_vs_levy_pdf(alpha: float = 0.5, t_val: float = 1.0, n_samples: int = 100000):
    """
    Generate a histogram of S_α samples and overlay the theoretical Lévy PDF for α=0.5,
    showing both a regular (linear) scale and a logarithmic scale.1
    
    Parameters:
        alpha (float): Stability parameter (default 0.5).
        t_val (float): Fixed laboratory time.
        n_samples (int): Number of samples for the histogram.
    """
    # Generate samples of S_alpha
    S_samples = np.array([compute_S_alpha(t_val, alpha) for _ in range(n_samples)])
    
    # Create log-spaced bins for the histogram (suitable for heavy-tailed distributions)
    bins = np.logspace(np.log10(S_samples.min()), np.log10(S_samples.max()), 100)
    hist, bin_edges = np.histogram(S_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = bin_edges[1:] - bin_edges[:-1]
    
    # Prepare theoretical PDF values over a range of s values
    s_vals = np.linspace(bin_centers.min(), bin_centers.max(), 500)
    pdf_vals = levy_pdf_alpha_half(s_vals)
    
    # Create two subplots: one with linear scales and one with logarithmic scales
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Regular (linear) scale plot
    axs[0].bar(bin_centers, hist, width=widths, color='C0', alpha=0.7, label='Simulated S₀.₅ histogram')
    axs[0].plot(s_vals, pdf_vals, 'r-', linewidth=2, label='Theoretical Lévy PDF (α=0.5)')
    axs[0].set_xlabel('S₀.₅')
    axs[0].set_ylabel('Probability density')
    axs[0].set_title('Histogram vs. Theoretical Lévy PDF (Regular Scale)')
    axs[0].legend()
    
    # Log-log scale plot
    axs[1].bar(bin_centers, hist, width=widths, color='C0', alpha=0.7, label='Simulated S₀.₅ histogram')
    axs[1].plot(s_vals, pdf_vals, 'r-', linewidth=2, label='Theoretical Lévy PDF (α=0.5)')
    axs[1].set_xlabel('S₀.₅')
    axs[1].set_ylabel('Probability density')
    axs[1].set_title('Histogram vs. Theoretical Lévy PDF (Log Scale)')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()


def main():
    while True:
        print("\nMenu:")
        print("1. View Histogram of S_alpha")
        print("2. View Single 2D Random Walk")
        print("3. View Histogram of Final Positions")
        print("4. View First Moment with Noise")
        print("5. View First Moment with Power-Law Fit and Find Function")
        print("6. Test Relationship Between Coefficient A and Width W")  
        print("9. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            # Placeholder for View Histogram of S_alpha
            print(" View Histogram of S_alpha is not implemented yet.")
            plot_histogram_vs_levy_pdf(alpha=0.5, t_val=1.0, n_samples=100000)

        elif choice == '2':
            # Placeholder for 2D random walk visualization
            print("2D Random Walk visualization is not implemented yet.")

        elif choice == '3':
            # Placeholder for histogram of final positions
            print("Histogram of Final Positions visualization is not implemented yet.")

        elif choice == '4':
            # Placeholder for first moment with noise
            print("First Moment with Noise visualization is not implemented yet.")

        elif choice == '5':
            # Placeholder for first moment with power-law fit
            print("First Moment with Power-Law Fit visualization is not implemented yet.")

        elif choice == '6':
            # Placeholder for testing relationship between coefficient A and width W
            print("Testing relationship between coefficient A and width W is not implemented yet.")

        elif choice == '9':
            print("Exiting.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
