import numpy as np
import matplotlib.pyplot as plt

def generate_eta(alpha: float) -> float:
    """
    Generate one-sided Lévy stable random variable η using the Chambers et al. method.
    """
    theta = np.pi * np.random.uniform(0,1)  # θ ~ U(0, π)
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
        float or np.array: PDF values for the standard Lévy(1/2) distribution.
    """
    pdf = np.zeros_like(x)
    positive = x > 0
    pdf[positive] = (1.0 / (2.0 * np.sqrt(np.pi))) * (x[positive] ** -1.5) * np.exp(-1.0 / (4.0 * x[positive]))
    return pdf

def run_eta_validation_plot(alpha: float = 0.5, N_samples: int = 1_000_000, hist_range=(0.001, 25.0), num_bins=200):
    """
    Run validation of generate_eta by comparing histogram of samples to theoretical PDF.
    """
    print(f"\nGenerating {N_samples} samples with α = {alpha}...")

    np.random.seed(42)  # For reproducibility
    samples = np.array([generate_eta(alpha) for _ in range(N_samples)])

    # Histogram
    counts, bins = np.histogram(samples, bins=num_bins, range=hist_range, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Theoretical PDF
    x_vals = np.linspace(hist_range[0], hist_range[1], 1000)
    pdf_vals = levy_pdf_alpha_half(x_vals)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, pdf_vals, 'r-', label="Theoretical Lévy PDF (α=0.5)", linewidth=2)
    plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), alpha=0.6, label="Generated η Histogram", color='skyblue', edgecolor='k')

    plt.title(f"Validation of Lévy Stable Sampling (α={alpha})")
    plt.xlabel("η")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main function to run the simulation and validation
def main():
    while True:
        print("\nMenu:")
        print("1. Validate eta generation (Plot Histogram vs PDF for alpha=0.5)")
        print("2. Run full simulation (Not implemented yet)")
        print("9. Exit")
        choice = input("Enter your choice (1/2/9): ").strip()

        if choice == '1':
            run_eta_validation_plot()
        elif choice == '2':
            print("Full simulation functionality is not yet implemented.")
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
