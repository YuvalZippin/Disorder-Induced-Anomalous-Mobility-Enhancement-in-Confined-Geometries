#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double uniform_rand_r(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

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

void compute_mean_final_position(
    double t, double alpha, double F,
    int Ly, int Lz, int num_sims,
    double *mean_x, double *mean_y, double *mean_z
) {
    int *x = aligned_alloc(32, num_sims * sizeof(int));
    int *y = aligned_alloc(32, num_sims * sizeof(int));
    int *z = aligned_alloc(32, num_sims * sizeof(int));
    unsigned int *seeds = malloc(num_sims * sizeof(unsigned int));
    long *targets = malloc(num_sims * sizeof(long));

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

    #pragma omp parallel for
    for (int i = 0; i < num_sims; i++) {
        x[i] = y[i] = z[i] = 0;
        seeds[i] = (unsigned int)(time(NULL) ^ (i + omp_get_thread_num()));
        double eta = generate_eta(alpha, &seeds[i]);
        targets[i] = (long)(pow(t / eta, alpha));
    }

    // Simulate each walker
    #pragma omp parallel for
    for (int i = 0; i < num_sims; i++) {
        for (long step = 0; step < targets[i]; step++) {
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

    // Average final positions
    double sum_x = 0, sum_y = 0, sum_z = 0;
    for (int i = 0; i < num_sims; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_z += z[i];
    }

    *mean_x = sum_x / num_sims;
    *mean_y = sum_y / num_sims;
    *mean_z = sum_z / num_sims;

    free(x); free(y); free(z); free(seeds); free(targets);
}

// Export results over range of t values
void export_first_moment_vs_time(
    double t_min, double t_max, int t_steps,
    double alpha, double F,
    int Ly, int Lz, int num_sims,
    const char *filename
) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open output file");
        return;
    }

    fprintf(fp, "t,mean_x,mean_y,mean_z\n");

    for (int i = 0; i < t_steps; i++) {
        double t = t_min + i * (t_max - t_min) / (t_steps - 1);
        double mean_x, mean_y, mean_z;

        compute_mean_final_position(t, alpha, F, Ly, Lz, num_sims, &mean_x, &mean_y, &mean_z);

        fprintf(fp, "%.8f,%.8f,%.8f,%.8f\n", t, mean_x, mean_y, mean_z);
        printf("Computed t = %.3f: ⟨x⟩ = %.3f\n", t, mean_x);
    }

    fclose(fp);
}


int main() {
    double alpha = 0.5;
    double F = 0.1;
    int Ly = 32, Lz = 32;
    int num_sims = 1000000;
    double t_min = 1.0, t_max = 1000.0;
    int t_steps = 50;

    export_first_moment_vs_time(
        t_min, t_max, t_steps,
        alpha, F, Ly, Lz, num_sims,
        "first_moment_vs_time.csv"
    );

    return 0;
}
// Compile with: gcc -fopenmp -o OPcheck OPcheck.c -lm
// Run with: ./OPcheck