#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to calculate the force-dependent parameter B.
double calculate_B(double F) {
    return 1.0 / (2.0 * cosh(F / 2.0) + 4.0);
}

// Function to calculate the return probability Q_0* for a given w and force F.
double calculate_Q_star(long long w, double F) {
    if (w <= 0) {
        return NAN;
    }
    
    // Handle the special case where the denominator becomes zero.
    if (w == 1 && std::abs(F) < 1e-9) {
        return 1.0;
    }

    double B = calculate_B(F);
    double sum_of_inverse_denominators = 0.0;
    
    for (long long m = 0; m < w; ++m) {
        for (long long n = 0; n < w; ++n) {
            double cos_m = cos(2.0 * M_PI * m / w);
            double cos_n = cos(2.0 * M_PI * n / w);

            double term_inside_sqrt = pow(1.0 - 2.0 * B * cos_m - 2.0 * B * cos_n, 2) - 4.0 * B * B;
            
            if (term_inside_sqrt < 0.0) {
                term_inside_sqrt = 0.0;
            }

            double denominator = sqrt(term_inside_sqrt);

            if (std::abs(denominator) < 1e-12) {
                return 1.0; 
            }

            sum_of_inverse_denominators += 1.0 / denominator;
        }
    }
    
    // Cast w*w to double to avoid overflow with large numbers
    return 1.0 - (static_cast<double>(w) * static_cast<double>(w) / sum_of_inverse_denominators);
}

int main() {
    std::ofstream data_file("q_star_data.csv");
    if (!data_file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return 1;
    }

    data_file << std::fixed << std::setprecision(12);
    data_file << "w,F=0.0,F=0.1,F=0.2,F=0.5\n"; // CSV header

    // Define the list of w values, using long long for large numbers.
    // The points are logarithmically spaced to show convergence effectively.
    std::vector<long long> w_values = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};
    
    // Define the force values to plot.
    std::vector<double> F_values = {0.0, 0.1, 0.2, 0.5};

    std::cout << "Generating data points for various F values..." << std::endl;
    std::cout << "This will take a significant amount of time for w = 10^7." << std::endl;

    for (long long w : w_values) {
        data_file << w;
        for (double F : F_values) {
            double Q_star = calculate_Q_star(w, F);
            data_file << "," << Q_star;
        }
        data_file << "\n";
        std::cout << "Data for w = " << w << " complete." << std::endl;
    }

    data_file.close();
    std::cout << "Data generation complete. Output saved to 'q_star_data.csv'." << std::endl;

    return 0;
}