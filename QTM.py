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






...