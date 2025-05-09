#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int x, y, z;
} Position;

typedef struct {
    Position *data;
    size_t size;
    size_t capacity;
} Trajectory;

// Initialize trajectory
void init_trajectory(Trajectory *traj, size_t initial_capacity) {
    traj->data = (Position *)malloc(initial_capacity * sizeof(Position));
    traj->size = 0;
    traj->capacity = initial_capacity;
}

// Append to trajectory
void append_trajectory(Trajectory *traj, int x, int y, int z) {
    if (traj->size >= traj->capacity) {
        traj->capacity *= 2;
        traj->data = (Position *)realloc(traj->data, traj->capacity * sizeof(Position));
    }
    traj->data[traj->size++] = (Position){x, y, z};
}

// Free trajectory
void free_trajectory(Trajectory *traj) {
    free(traj->data);
}

// Uniform random number between 0 and 1
static inline double uniform_rand() {
    return (double)rand() / (double)RAND_MAX;
}

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


Trajectory run_single_trajectory(double t, double alpha, double F, int Ly, int Lz) {
    Trajectory traj;
    init_trajectory(&traj, 1000); // Start with capacity for 1000 steps

    double S_alpha = compute_S_alpha(t, alpha);
    if (!isfinite(S_alpha)) {
        printf("Warning: S_alpha is non-finite (%f). Returning only initial position.\n", S_alpha);
        append_trajectory(&traj, 0, 0, 0);
        return traj;
    }

    int x = 0, y = 0, z = 0;
    append_trajectory(&traj, x, y, z);

    double exp_F2 = exp(F / 2.0);
    double exp_negF2 = exp(-F / 2.0);
    double A = 4.0 + exp_F2 + exp_negF2;
    
    if (A == 0.0) {
        printf("Error: Normalization constant A is zero.\n");
        return traj;
    }

    double probs[6] = {
        exp_F2 / A,         // +X
        exp_negF2 / A,      // -X
        1.0 / A,            // +Y
        1.0 / A,            // -Y
        1.0 / A,            // +Z
        1.0 / A             // -Z
    };

    // Convert to cumulative probabilities for binary decision
    for (int i = 1; i < 6; i++) {
        probs[i] += probs[i - 1];
    }

    long n_steps = 0;
    long target_steps = (long)(S_alpha);

    while (n_steps < target_steps) {
        double r = uniform_rand();
        if (r < probs[0]) {
            x += 1;
        } else if (r < probs[1]) {
            x -= 1;
        } else if (r < probs[2]) {
            y = (y + 1) % Ly;
        } else if (r < probs[3]) {
            y = (y - 1 + Ly) % Ly;
        } else if (r < probs[4]) {
            z = (z + 1) % Lz;
        } else {
            z = (z - 1 + Lz) % Lz;
        }

        append_trajectory(&traj, x, y, z);
        n_steps++;
    }

    return traj;
}
