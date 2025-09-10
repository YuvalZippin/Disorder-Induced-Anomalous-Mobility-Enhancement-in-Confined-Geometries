import math

def calculate_B(F):
    """
    Calculates the force-dependent parameter B.
    """
    return 1.0 / (2.0 * math.cosh(F / 2.0) + 4.0)

def calculate_Q_star(w, F):
    """
    Calculates the probability of return Q_0* for a given w and F.
    """
    if w <= 0:
        return float('nan')
    
    # Handle the special case where the denominator becomes zero.
    if w == 1 and F == 0.0:
        return 1.0
        
    B = calculate_B(F)
    sum_of_inverse_denominators = 0.0
    
    # The loops for m and n
    for m in range(w):
        for n in range(w):
            cos_m = math.cos(2.0 * math.pi * m / w)
            cos_n = math.cos(2.0 * math.pi * n / w)
            
            term_inside_sqrt = (1.0 - 2.0 * B * cos_m - 2.0 * B * cos_n)**2 - 4.0 * B**2
            
            # Handle floating-point inaccuracies
            if term_inside_sqrt < 0.0:
                term_inside_sqrt = 0.0
            
            denominator = math.sqrt(term_inside_sqrt)
            
            if abs(denominator) < 1e-12:
                # This corresponds to a divergent sum, so Q_0* is 1
                return 1.0
            
            sum_of_inverse_denominators += 1.0 / denominator
            
    return 1.0 - (w**2 / sum_of_inverse_denominators)

if __name__ == "__main__":
    try:
        # Get user input for w and F
        w_input = int(input("Enter the lattice size (w): "))
        F_input = float(input("Enter the force (F): "))
        
        # Calculate Q_0*
        result = calculate_Q_star(w_input, F_input)
        
        # Print the result to the terminal
        if math.isnan(result):
            print("Invalid input for w.")
        else:
            print(f"For w = {w_input} and F = {F_input}, the probability of return Q_0* is: {result:.10f}")
            
    except ValueError:
        print("Invalid input. Please enter numbers for w and F.")