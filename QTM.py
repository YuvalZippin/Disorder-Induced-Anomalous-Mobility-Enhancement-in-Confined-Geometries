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
