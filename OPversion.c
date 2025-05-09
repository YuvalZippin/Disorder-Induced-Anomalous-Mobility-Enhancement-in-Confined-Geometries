#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Thread-safe RNG using rand_r
static inline double uniform_rand_r(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

// Generate one-sided Lévy stable random variable using Chambers method
double generate_eta(double alpha, unsigned int *seed) {
    double theta = M_PI * uniform_rand_r(seed);
    double W = -log(uniform_rand_r(seed));
    double sin_alpha_theta = sin(alpha * theta);
    double sin_one_minus_alpha_theta = sin((1 - alpha) * theta);
    double sin_theta = sin(theta);

    double a_theta = sin_one_minus_alpha_theta *
        pow(sin_alpha_theta, alpha / (1 - alpha)) /
        pow(sin_theta, 1 / (1 - alpha));

    return pow(a_theta / W, (1 - alpha) / alpha);
}

// Batch simulation of N walkers using SIMD-friendly arrays
void run_batch_walkers(
    int N, double t, double alpha, double F,
    int Ly, int Lz, int max_steps
) {
    // Allocate aligned memory
    int *x = aligned_alloc(32, N * sizeof(int));
    int *y = aligned_alloc(32, N * sizeof(int));
    int *z = aligned_alloc(32, N * sizeof(int));
    unsigned int *seeds = malloc(N * sizeof(unsigned int));
    long *targets = malloc(N * sizeof(long));

    // Initialize walkers
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = y[i] = z[i] = 0;
        seeds[i] = (unsigned int)(time(NULL) ^ (i + omp_get_thread_num()));
        double eta = generate_eta(alpha, &seeds[i]);
        targets[i] = (long)(pow(t / eta, alpha));
    }

    // Compute biased probabilities
    double exp_F2 = exp(F / 2.0);
    double exp_negF2 = exp(-F / 2.0);
    double A = 4.0 + exp_F2 + exp_negF2;

    double cum_probs[6] = {
        exp_F2 / A,
        (exp_F2 + exp_negF2) / A,
        (exp_F2 + exp_negF2 + 1.0) / A,
        (exp_F2 + exp_negF2 + 2.0) / A,
        (exp_F2 + exp_negF2 + 3.0) / A,
        1.0
    };

    // Main simulation loop (batch update)
    for (int step = 0; step < max_steps; step++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (step >= targets[i]) continue;

            double r = uniform_rand_r(&seeds[i]);

            if (r < cum_probs[0]) {
                x[i]++;
            } else if (r < cum_probs[1]) {
                x[i]--;
            } else if (r < cum_probs[2]) {
                y[i] = (y[i] + 1) % Ly;
            } else if (r < cum_probs[3]) {
                y[i] = (y[i] - 1 + Ly) % Ly;
            } else if (r < cum_probs[4]) {
                z[i] = (z[i] + 1) % Lz;
            } else {
                z[i] = (z[i] - 1 + Lz) % Lz;
            }
        }
    }

    // Example output
    for (int i = 0; i < N; i++) {
        printf("Walker %d final position: (%d, %d, %d), steps = %ld\n", i, x[i], y[i], z[i], targets[i]);
    }

    // Cleanup
    free(x); free(y); free(z); free(seeds); free(targets);
}
