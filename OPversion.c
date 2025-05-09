#include <math.h>
#include <stdlib.h>
#include <time.h>

// Uniform random number between 0 and 1
static inline double uniform_rand() {
    return (double)rand() / (double)RAND_MAX;
}

// Generate one-sided Lévy stable random variable η using the Chambers et al. method
double generate_eta(double alpha) {
    const double PI = 3.141592653589793;
    double theta = PI * uniform_rand();       // θ ~ U(0, π)
    double W = -log(uniform_rand());          // W = -ln(U), U ~ U(0,1)

    double sin_theta = sin(theta);
    double sin_alpha_theta = sin(alpha * theta);
    double sin_1_alpha_theta = sin((1.0 - alpha) * theta);

    double numerator = sin_1_alpha_theta * pow(sin_alpha_theta, alpha / (1.0 - alpha));
    double denominator = pow(sin_theta, 1.0 / (1.0 - alpha));
    double a_theta = numerator / denominator;

    double eta = pow(a_theta / W, (1.0 - alpha) / alpha);
    return eta;
}

// Compute S_alpha given laboratory time t and stability parameter alpha
double compute_S_alpha(double t, double alpha) {
    double eta = generate_eta(alpha);
    double S_alpha = pow(t / eta, alpha);
    return S_alpha;
}

