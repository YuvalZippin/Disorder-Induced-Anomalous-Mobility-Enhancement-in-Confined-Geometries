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

def analyze_S_alpha_vs_t(alpha_val, t_start, t_finish, num_samples):
    """
    Analyze the behavior of S_alpha as a function of t for a fixed alpha.
    
    Parameters:
        alpha_val (float): Stability parameter (0 < alpha < 1).
        t_start (float): Starting value of t.
        t_finish (float): Ending value of t.
        num_samples (int): Number of samples to generate for each t.
    
    Returns:
        None: Displays a plot of S_alpha behavior as a function of t.
    """
    t_values = np.linspace(t_start, t_finish, num=50)  # Generate 50 evenly spaced t values
    mean_S_alpha = []  # Store the mean of S_alpha for each t
    median_S_alpha = []  # Store the median of S_alpha for each t

    for t in t_values:
        S_alpha_samples = [compute_S_alpha(t, alpha_val) for _ in range(num_samples)]
        mean_S_alpha.append(np.mean(S_alpha_samples))  # Compute mean
        median_S_alpha.append(np.median(S_alpha_samples))  # Compute median

    # Plot the behavior of S_alpha as a function of t
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, mean_S_alpha, label="Mean of S_alpha", color="blue", marker="o")
    plt.plot(t_values, median_S_alpha, label="Median of S_alpha", color="orange", marker="x")
    
    # If alpha = 1/2, overlay the theoretical Lévy PDF
    if alpha_val == 0.5:
        levy_pdf_values = levy_pdf_alpha_half(t_values)
        plt.plot(t_values, levy_pdf_values, label="Theoretical Lévy PDF (alpha=1/2)", color="green", linestyle="--")

    plt.xlabel("t (Laboratory Time)")
    plt.ylabel("S_alpha (Density)")
    plt.yscale("log")  # Log scale for better visualization
    plt.xscale("log")  # Log scale for better visualization
    plt.title(f"Behavior of S_alpha as a Function of t (alpha = {alpha_val})")
    plt.legend()
    plt.grid(True)
    plt.show()




    # Main function to call the analysis
if __name__ == "__main__":
    # Parameters
    alpha_val = 0.5  # Stability parameter
    t_start = 1.0  # Start of t range
    t_finish = 100.0  # End of t range
    num_samples = 100_000  # Number of samples for each t
    
    # Call the analysis function
    analyze_S_alpha_vs_t(alpha_val, t_start, t_finish, num_samples)