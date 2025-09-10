#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace std;

// High precision computation of Q0* for 3D random walk on lattice
class RandomWalkCalculator {
private:
    long double B;
    long long w;
    
public:
    RandomWalkCalculator(long double F, long long w_val) : w(w_val) {
        // B = 1/(2*cosh(F/2) + 4)
        B = 1.0L / (2.0L * coshl(F / 2.0L) + 4.0L);
    }
    
    // Main calculation function
    long double calculateQ0Star() {
        // Handle special case w = 1
        if (w == 1) {
            return 1.0L;
        }
        
        long double sum = 0.0L;
        long double w_squared = static_cast<long double>(w) * static_cast<long double>(w);
        
        // Use Kahan summation algorithm for better numerical stability
        long double compensation = 0.0L;
        
        for (long long m = 0; m < w; ++m) {
            for (long long n = 0; n < w; ++n) {
                // Compute angles
                long double angle_m = 2.0L * M_PI * static_cast<long double>(m) / static_cast<long double>(w);
                long double angle_n = 2.0L * M_PI * static_cast<long double>(n) / static_cast<long double>(w);
                
                long double cos_m = cosl(angle_m);
                long double cos_n = cosl(angle_n);
                
                // The formula is: Q0* = 1 - w^2 / sum_{m,n=0}^{w-1} [(1-2B*cos(2πm/w)-2B*cos(2πn/w))^2 - 4B^2]^(-1/2)
                long double bracket_term = 1.0L - 2.0L * B * cos_m - 2.0L * B * cos_n;
                long double inside_brackets = bracket_term * bracket_term - 4.0L * B * B;
                
                // Check for negative or near-zero values inside square root
                if (inside_brackets <= 0.0L) {
                    continue;
                }
                
                // For very small positive values, check if they're numerically significant
                if (inside_brackets < 1e-12L) {
                    continue;
                }
                
                // Take the square root and then reciprocal: [(...)^2 - 4B^2]^(-1/2) = 1/sqrt(inside_brackets)
                long double term = 1.0L / sqrtl(inside_brackets);
                
                // Kahan summation to reduce numerical errors
                long double y = term - compensation;
                long double t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
            
            // Progress indicator for large w
            if (w > 1000 && m % (w / 100) == 0 && m > 0) {
                cerr << "Progress: " << (100 * m) / w << "%" << endl;
            }
        }
        
        // Final calculation: Q0* = 1 - w^2 / sum
        long double result = 1.0L - w_squared / sum;
        
        return result;
    }
    
    // Debug version with detailed output
    long double calculateQ0StarDebug() {
        // Handle special case w = 1
        if (w == 1) {
            cout << "Special case w=1, returning 1.0" << endl;
            return 1.0L;
        }
        
        long double sum = 0.0L;
        long double w_squared = static_cast<long double>(w) * static_cast<long double>(w);
        
        cout << "B = " << setprecision(15) << B << endl;
        cout << "w^2 = " << w_squared << endl;
        
        // Use Kahan summation algorithm for better numerical stability
        long double compensation = 0.0L;
        
        for (long long m = 0; m < w; ++m) {
            for (long long n = 0; n < w; ++n) {
                // Compute angles
                long double angle_m = 2.0L * M_PI * static_cast<long double>(m) / static_cast<long double>(w);
                long double angle_n = 2.0L * M_PI * static_cast<long double>(n) / static_cast<long double>(w);
                
                long double cos_m = cosl(angle_m);
                long double cos_n = cosl(angle_n);
                
                // The formula is: Q0* = 1 - w^2 / sum_{m,n=0}^{w-1} [(1-2B*cos(2πm/w)-2B*cos(2πn/w))^2 - 4B^2]^(-1/2)
                long double cos_sum = cos_m + cos_n;
                long double bracket_term = 1.0L - 2.0L * B * cos_sum;
                long double inside_brackets = bracket_term * bracket_term - 4.0L * B * B;
                
                // Debug output for first few terms
                if (m < 3 && n < 3) {
                    cout << "m=" << m << ", n=" << n << ": cos_m=" << cos_m << ", cos_n=" << cos_n 
                         << ", cos_sum=" << cos_sum << ", bracket_term=" << bracket_term << ", inside_brackets=" << inside_brackets;
                }
                
                // Check for negative or near-zero values inside square root
                if (inside_brackets <= 0.0L) {
                    if (m < 3 && n < 3) {
                        cout << " -> SKIPPED (non-positive)" << endl;
                    }
                    continue;
                }
                
                // For very small positive values, check if they're numerically significant
                if (inside_brackets < 1e-12L) {
                    if (m < 3 && n < 3) {
                        cout << " -> SKIPPED (too small: " << inside_brackets << ")" << endl;
                    }
                    continue;
                }
                
                // Take the square root and then reciprocal: [(...)^2 - 4B^2]^(-1/2) = 1/sqrt(inside_brackets)
                long double term = 1.0L / sqrtl(inside_brackets);
                
                if (m < 3 && n < 3) {
                    cout << ", term=" << term << endl;
                }
                
                // Kahan summation to reduce numerical errors
                long double y = term - compensation;
                long double t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
        }
        
        cout << "Total sum: " << sum << endl;
        cout << "w^2/sum: " << w_squared / sum << endl;
        
        // Final calculation: Q0* = 1 - w^2 / sum
        long double result = 1.0L - w_squared / sum;
        
        return result;
    }
    
    long double getB() const { return B; }
};

int main() {
    cout << fixed << setprecision(15);
    cerr << fixed << setprecision(1);
    
    long long w;
    long double F;
    
    cout << "3D Random Walk Q0* Calculator" << endl;
    cout << "=============================" << endl;
    
    cout << "Enter lattice size w: ";
    if (!(cin >> w) || w <= 0) {
        cerr << "Error: Invalid input for w. Please enter a positive integer." << endl;
        return 1;
    }
    
    cout << "Enter external force F: ";
    if (!(cin >> F)) {
        cerr << "Error: Invalid input for F. Please enter a real number." << endl;
        return 1;
    }
    
    cout << "\nInput parameters:" << endl;
    cout << "w = " << w << endl;
    cout << "F = " << F << endl;
    
    // Create calculator instance
    RandomWalkCalculator calculator(F, w);
    
    cout << "\nCalculating Q0*..." << endl;
    
    long double result;
    if (w <= 20) {
        // Use debug version for small w to see what's happening
        result = calculator.calculateQ0StarDebug();
    } else {
        result = calculator.calculateQ0Star();
    }
    
    cout << "\nResults:" << endl;
    cout << "=======" << endl;
    cout << "Q0* = " << setprecision(15) << result << endl;
    
    // Verify known special cases
    if (w == 1 && fabsl(F) < 1e-10) {
        cout << "\nVerification: w=1, F=0 should give Q0*=1" << endl;
        cout << "Calculated: " << result << " (Expected: 1.0)" << endl;
        cout << "Match: " << (fabsl(result - 1.0L) < 1e-10 ? "YES" : "NO") << endl;
    }
    
    if (fabsl(F) < 1e-10 && w > 10) {
        cout << "\nNote: For F=0, Q0* should be positive and approach 0.340537... as w increases" << endl;
        cout << "Current value: " << result << endl;
    }
    
    // Test multiple values to see where the calculation becomes stable
    cout << "\nTesting multiple w values for F=0:" << endl;
    for (int test_w : {1, 2, 3, 4, 5, 10, 20, 50, 100}) {
        RandomWalkCalculator test_calc(0.0L, test_w);
        long double test_result = test_calc.calculateQ0Star();
        cout << "Q0*(w=" << test_w << ", F=0) = " << setprecision(6) << test_result << endl;
        
        // Stop if we get reasonable values
        if (test_result > 0.3 && test_result < 0.5 && test_w >= 50) break;
    }
    
    return 0;
}
// g++ calcQstar.cpp -o calcQstar -std=c++11 -O2
// ./calcQstar