import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def Q0_3D_infinity():
    """3D continuous integral reference"""
    B0 = 1/6  # F=0 for 3D

    def integrand(kx, ky):
        denom = (1 - 2*B0*np.cos(kx) - 2*B0*np.cos(ky))**2 - (2*B0)**2
        return 1/np.sqrt(denom)

    val, err = integrate.nquad(integrand, [[0, 2*np.pi], [0, 2*np.pi]])
    avg = val / (2*np.pi)**2
    Q0 = 1 - 1/avg
    return Q0

def Q0_2D_infinity():
    """2D continuous integral reference"""
    B0 = 1/4  # F=0 for 2D

    def integrand(k):
        denom = (1 - 2*B0*np.cos(k))**2 - (2*B0)**2
        if denom <= 0:
            return 0  # Handle potential singularities
        return 1/np.sqrt(denom)

    val, err = integrate.quad(integrand, 0, 2*np.pi)
    avg = val / (2*np.pi)
    Q0 = 1 - 1/avg
    return Q0

def calculate_Q0_2D_discrete(w, F=0):
    """
    2D case: Q₀* = 1 - w/Σ[(1-2B cos(2πm/w))² - (2B)²]^(-1/2)
    where B = 1/(2 cosh(F/2) + 4) = 1/4 when F=0
    """
    B = 1 / (2 * np.cosh(F/2) + 4)  # For 2D: B = 1/4 when F=0
    
    sum_term = 0
    for m in range(w):
        cos_term = np.cos(2 * np.pi * m / w)
        denominator_term = (1 - 2*B*cos_term)**2 - (2*B)**2
        
        if denominator_term > 1e-15:
            sum_term += denominator_term**(-0.5)
        else:
            print(f"Warning: small denominator in 2D case at m={m}, w={w}")
    
    # For 2D: Q₀* = 1 - w/sum_term (check this normalization!)
    Q0_star = 1 - w/sum_term
    return Q0_star

def calculate_Q0_3D_discrete_2D_sum(w, F=0):
    """
    3D case using 2D sum discretization
    """
    B = 1 / (2 * np.cosh(F/2) + 4)  # B = 1/6 when F=0
    
    sum_term = 0
    for mx in range(w):
        for my in range(w):
            cos_x = np.cos(2 * np.pi * mx / w)
            cos_y = np.cos(2 * np.pi * my / w)
            
            denominator_term = (1 - 2*B*cos_x - 2*B*cos_y)**2 - (2*B)**2
            
            if denominator_term > 1e-15:
                sum_term += denominator_term**(-0.5)
    
    avg_term = sum_term / (w**2)
    Q0_star = 1 - 1/avg_term
    return Q0_star

def calculate_Q0_3D_original_formula(w, F=0):
    """
    Original 3D formula as written (single sum with 4B coefficient)
    """
    B = 1 / (2 * np.cosh(F/2) + 4)  # B = 1/6 when F=0
    
    sum_term = 0
    for m in range(w):
        cos_term = np.cos(2 * np.pi * m / w)
        denominator_term = (1 - 4*B*cos_term)**2 - 4*B**2
        
        if denominator_term > 1e-15:
            sum_term += denominator_term**(-0.5)
    
    W_tilde_1_0 = w**2 / sum_term
    Q0_star = 1 - 1/W_tilde_1_0
    return Q0_star

# Get reference values
reference_2D = Q0_2D_infinity()
reference_3D = Q0_3D_infinity()

print(f"2D Reference (continuous): {reference_2D:.8f}")
print(f"3D Reference (continuous): {reference_3D:.8f}")

# Debug: Check B values
B_2D = 1/4
B_3D = 1/6
print(f"B values: 2D = {B_2D:.6f}, 3D = {B_3D:.6f}")

# Debug: Check a simple 2D calculation
print(f"\nDebugging 2D case:")
test_w = 10
test_sum = 0
B = 1/4
for m in range(test_w):
    cos_term = np.cos(2 * np.pi * m / test_w)
    denom = (1 - 2*B*cos_term)**2 - (2*B)**2
    if denom > 0:
        test_sum += denom**(-0.5)
    print(f"m={m}: cos={cos_term:.4f}, denom={denom:.6f}, term={denom**(-0.5) if denom > 0 else 0:.6f}")

print(f"Sum = {test_sum:.6f}")
print(f"Q0 = 1 - {test_w}/{test_sum:.6f} = {1 - test_w/test_sum:.6f}")

print(f"Known 2D Random Walk return probability: 1.0 (certain return)")
print(f"Known 3D Random Walk return probability: ~0.35")

# Test range of w values
w_values = np.array([100, 150, 200, 300, 400, 500, 800, 1000, 2000])

print("\nCalculating all cases...")
Q0_2D_discrete = []
Q0_3D_2Dsum = []
Q0_3D_original = []

for w in w_values:
    q2d = calculate_Q0_2D_discrete(w, F=0)
    q3d_2dsum = calculate_Q0_3D_discrete_2D_sum(w, F=0)
    q3d_orig = calculate_Q0_3D_original_formula(w, F=0)
    
    Q0_2D_discrete.append(q2d)
    Q0_3D_2Dsum.append(q3d_2dsum)
    Q0_3D_original.append(q3d_orig)
    
    if w <= 50 or w % 100 == 0:
        print(f"w={w:4d}: 2D={q2d:.6f}, 3D_2Dsum={q3d_2dsum:.6f}, 3D_orig={q3d_orig:.6f}")

Q0_2D_discrete = np.array(Q0_2D_discrete)
Q0_3D_2Dsum = np.array(Q0_3D_2Dsum)
Q0_3D_original = np.array(Q0_3D_original)

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Linear scale - full comparison
ax1.plot(w_values, Q0_2D_discrete, 'mo-', linewidth=2, markersize=4, 
         label='2D Case (B=1/4)')
ax1.plot(w_values, Q0_3D_2Dsum, 'bo-', linewidth=2, markersize=4, 
         label='3D Case - 2D Sum (B=1/6)')
ax1.plot(w_values, Q0_3D_original, 'go-', linewidth=2, markersize=4, 
         label='3D Original Formula')

ax1.axhline(y=reference_2D, color='purple', linestyle='--', linewidth=2, 
            label=f'2D Reference = {reference_2D:.4f}')
ax1.axhline(y=reference_3D, color='red', linestyle='--', linewidth=2, 
            label=f'3D Reference = {reference_3D:.4f}')
ax1.axhline(y=0.35, color='orange', linestyle=':', linewidth=2, alpha=0.7,
            label='3D Random Walk ≈ 0.35')

ax1.set_xlabel('w (system size)', fontsize=12)
ax1.set_ylabel('Q₀*', fontsize=12)
ax1.set_title('Linear Scale: Q₀* vs w', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: Log-log scale - convergence analysis
ax2.loglog(w_values, np.abs(Q0_2D_discrete - reference_2D), 'mo-', 
           linewidth=2, markersize=4, label='|Q₀²ᴰ - Reference|')
ax2.loglog(w_values, np.abs(Q0_3D_2Dsum - reference_3D), 'bo-', 
           linewidth=2, markersize=4, label='|Q₀³ᴰ - Reference|')

# Add power law references
w_ref = w_values[5:].astype(float)  # Convert to float for negative powers
ax2.loglog(w_ref, 0.1 * w_ref**(-1), 'k--', alpha=0.5, label='w⁻¹')
ax2.loglog(w_ref, 0.01 * w_ref**(-2), 'k:', alpha=0.5, label='w⁻²')

ax2.set_xlabel('w (system size)', fontsize=12)
ax2.set_ylabel('|Error|', fontsize=12)
ax2.set_title('Log-Log: Convergence Error', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Plot 3: Focus on 2D case
ax3.plot(w_values, Q0_2D_discrete, 'mo-', linewidth=3, markersize=5)
ax3.axhline(y=reference_2D, color='purple', linestyle='--', linewidth=2, 
            label=f'2D Continuous = {reference_2D:.6f}')
ax3.axhline(y=1.0, color='red', linestyle=':', linewidth=2, alpha=0.7,
            label='2D Random Walk = 1.0')

ax3.set_xlabel('w (system size)', fontsize=12)
ax3.set_ylabel('Q₀*', fontsize=12)
ax3.set_title('2D Case: Return Probability', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

# Plot 4: Focus on 3D case  
ax4.plot(w_values, Q0_3D_2Dsum, 'bo-', linewidth=3, markersize=5)
ax4.axhline(y=reference_3D, color='red', linestyle='--', linewidth=2, 
            label=f'3D Continuous = {reference_3D:.6f}')
ax4.axhline(y=0.35, color='orange', linestyle=':', linewidth=2, alpha=0.7,
            label='3D Random Walk ≈ 0.35')

ax4.set_xlabel('w (system size)', fontsize=12)
ax4.set_ylabel('Q₀*', fontsize=12)
ax4.set_title('3D Case: Return Probability', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=11)

plt.tight_layout()
plt.show()

# Final analysis
print(f"\n" + "="*60)
print("FINAL CONVERGENCE ANALYSIS")
print("="*60)
print(f"2D Case (w={w_values[-1]}):")
print(f"  Discrete: {Q0_2D_discrete[-1]:.8f}")
print(f"  Reference: {reference_2D:.8f}")
print(f"  Error: {abs(Q0_2D_discrete[-1] - reference_2D):.2e}")
print(f"  Known 2D RW: 1.0 (certain return)")

print(f"\n3D Case (w={w_values[-1]}):")
print(f"  Discrete: {Q0_3D_2Dsum[-1]:.8f}")
print(f"  Reference: {reference_3D:.8f}")
print(f"  Error: {abs(Q0_3D_2Dsum[-1] - reference_3D):.2e}")
print(f"  Known 3D RW: ~0.35")

print(f"\nDifference from known values:")
print(f"  2D: |{reference_2D:.6f} - 1.0| = {abs(reference_2D - 1.0):.6f}")
print(f"  3D: |{reference_3D:.6f} - 0.35| = {abs(reference_3D - 0.35):.6f}")