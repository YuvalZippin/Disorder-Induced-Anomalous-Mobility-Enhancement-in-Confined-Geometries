#include <iostream>
#include <cmath>

// Numerical integration parameters
const int N = 5000; // Number of points per axis (increase for higher accuracy)
const double PI = 3.14159265358979323846;

// Function to compute the integrand
double integrand(double kx, double ky, double B) {
    double S = std::cos(kx) + std::cos(ky);
    double denom = std::pow(1.0 - 2.0 * B * S, 2.0) - 4.0 * B * B;
    return 1.0 / std::sqrt(denom);
}

// Main function to compute Q0(infty, F)
double Q0_infty(double F) {
    double B = 1.0 / (2.0 * std::cosh(F / 2.0) + 4.0);
    double sum = 0.0;
    double dk = 2.0 * PI / N;
    for (int i = 0; i < N; ++i) {
        double kx = -PI + i * dk;
        for (int j = 0; j < N; ++j) {
            double ky = -PI + j * dk;
            sum += integrand(kx, ky, B);
        }
    }
    double integral = sum * dk * dk;
    double Q0 = 1.0 - 1.0 / (integral / (4.0 * PI * PI));
    return Q0;
}

int main() {
    double F = 0.01; // Set your bias here
    double Q0 = Q0_infty(F);
    std::cout << "Q0(infty, F=" << F << ") = " << Q0 << std::endl;
    return 0;
}
