#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <numeric>

// Parameters tuned for accuracy and convergence
const int N_traj = 3000000;    // Large number for good stats
const double alpha = 0.3;
const double F = 0.01;          // Fixed small force to satisfy F << 2/w^2
const std::vector<int> Ws = {1, 5, 10};
const int measurement_points = 15;
const double t_min = 1e14;
const double t_max = 1e16;

// Step probabilities container
struct Probabilities {
    double p_plus_x, p_minus_x, p_plus_y, p_minus_y, p_plus_z, p_minus_z;
};

// Calculate step probabilities for given F
Probabilities compute_probabilities(double F) {
    double B = 1.0 / (2 * cosh(F / 2.0) + 4);
    return {B * exp(F / 2.0), B * exp(-F / 2.0), B, B, B, B};
}

// Generate logarithmically spaced time points
std::vector<double> generate_log_times(int n, double t_min, double t_max) {
    std::vector<double> times(n);
    double log_min = log10(t_min);
    double log_max = log10(t_max);
    double delta = (log_max - log_min) / (n - 1);
    for (int i = 0; i < n; ++i) times[i] = pow(10, log_min + i * delta);
    return times;
}

// Generate random number of steps S_alpha per Lévy stable distribution, capped for practicality
uint64_t generate_steps(double alpha, double t, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform_theta(0.0, M_PI);
    std::uniform_real_distribution<double> uniform_0_1(0.0, 1.0);

    double theta = uniform_theta(rng);
    double x = uniform_0_1(rng);
    double W = -log(x);

    double a_theta_num = sin((1 - alpha) * theta) * pow(sin(alpha * theta), alpha / (1 - alpha));
    double a_theta_den = pow(sin(theta), 1 / (1 - alpha));
    double a_theta = a_theta_num / a_theta_den;

    double eta = pow(a_theta / W, (1 - alpha) / alpha);

    uint64_t steps = static_cast<uint64_t>(ceil(pow(t / eta, alpha)));
    steps = std::min(steps, static_cast<uint64_t>(1e8));  // cap steps for computational feasibility
    return (steps < 1) ? 1 : steps;
}

// Simulate one biased random walk trajectory with periodic y,z boundaries, return x displacement
int64_t simulate_trajectory(int w, const Probabilities& p, double alpha, double t, std::mt19937_64& rng) {
    uint64_t steps = generate_steps(alpha, t, rng);

    int64_t x = 0;
    int y = 0;
    int z = 0;

    std::uniform_real_distribution<double> uniform_0_1(0.0, 1.0);

    double cum_probs[6] = {
        p.p_plus_x,
        p.p_plus_x + p.p_minus_x,
        p.p_plus_x + p.p_minus_x + p.p_plus_y,
        p.p_plus_x + p.p_minus_x + p.p_plus_y + p.p_minus_y,
        p.p_plus_x + p.p_minus_x + p.p_plus_y + p.p_minus_y + p.p_plus_z,
        1.0
    };

    int64_t x_displacement_sum = 0;

    for (uint64_t step = 0; step < steps; ++step) {
        double r = uniform_0_1(rng);
        if (r < cum_probs[0]) {
            x++;
            x_displacement_sum++;
        } else if (r < cum_probs[1]) {
            x--;
            x_displacement_sum--;
        } else if (r < cum_probs[2]) {
            y = (y + 1) % w;
        } else if (r < cum_probs[3]) {
            y = (y - 1 + w) % w;
        } else if (r < cum_probs[4]) {
            z = (z + 1) % w;
        } else {
            z = (z - 1 + w) % w;
        }
    }
    return x_displacement_sum;
}

// Worker thread function to accumulate results over trajectory range
void worker(int w, const Probabilities& p, double alpha, double t, int start, int end, std::atomic<double>& accumulator) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    rng.discard(start * 1000);
    double local_sum = 0.0;
    for (int i = start; i < end; ++i) {
        local_sum += static_cast<double>(simulate_trajectory(w, p, alpha, t, rng));
    }
    accumulator =+ local_sum;  // single atomic update per thread
}

// Fit power law \(\langle x \rangle = \mu t^{\alpha}\), extracting \(\mu\)
double fit_mu(const std::vector<double> &times, const std::vector<double> &x_averages, double alpha) {
    int n = times.size();
    std::vector<double> ln_t(n), ln_x(n);
    for (int i = 0; i < n; i++) {
        ln_t[i] = log(times[i]);
        ln_x[i] = log(std::max(x_averages[i], 1e-10));
    }
    double mean_ln_t = std::accumulate(ln_t.begin(), ln_t.end(), 0.0) / n;
    double mean_ln_x = std::accumulate(ln_x.begin(), ln_x.end(), 0.0) / n;
    double ln_mu = mean_ln_x - alpha * mean_ln_t;
    return exp(ln_mu);
}

int main() {
    Probabilities p = compute_probabilities(F);
    auto times = generate_log_times(measurement_points, t_min, t_max);
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;

    std::ofstream outfile("full_sim_results.csv");
    std::ofstream mu_file("mu_vs_w.csv");
    outfile << "w,t,average_x_displacement\n";
    mu_file << "w,mu\n";

    std::cout << "=== Running simulation for F = " << F << ", alpha = " << alpha << std::endl;
    std::cout << "Threads: " << n_threads << ", Trajectories: " << N_traj << std::endl;

    for (int w : Ws) {
        std::vector<double> avg_x_all_times;

        std::cout << "\nProcessing w = " << w << std::endl;
        for (int m = 0; m < measurement_points; ++m) {
            double t = times[m];
            std::atomic<double> x_accumulator(0.0);
            int traj_per_thread = N_traj / n_threads;
            std::vector<std::thread> threads;

            for (unsigned int i = 0; i < n_threads; ++i) {
                int start_idx = i * traj_per_thread;
                int end_idx = (i == n_threads - 1) ? N_traj : start_idx + traj_per_thread;
                threads.emplace_back(worker, w, std::ref(p), alpha, t, start_idx, end_idx, std::ref(x_accumulator));
            }
            for (auto &thread : threads) thread.join();

            double avg_x = x_accumulator / static_cast<double>(N_traj);
            avg_x_all_times.push_back(avg_x);

            std::cout << "  t=" << t << ", <x>=" << avg_x << std::endl;
            outfile << w << "," << t << "," << avg_x << "\n";
        }

        double mu = fit_mu(times, avg_x_all_times, alpha);
        std::cout << "Fitted mu for w=" << w << ": " << mu << std::endl;
        mu_file << w << "," << mu << "\n";
    }

    outfile.close();
    mu_file.close();

    std::cout << "\nSimulation complete. Data saved to full_sim_results.csv and mu_vs_w.csv\n";

    return 0;
}
// Compile with: g++ -std=c++17 -O3 -pthread full_sim_quicktest.cpp -o full_sim_quicktest
// Run with: ./full_sim_quicktest
// Output files: full_sim_results.csv, mu_vs_w.csv