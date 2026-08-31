#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>
#include <algorithm>
#include <utility> // for std::pair

const int N_traj = 10000; // Increase for better statistics
const double alpha_input = 0.3; // Used for step generation
const double F = 0.01;
const std::vector<int> Ws = {1, 10}; // Add more widths as needed
const int measurement_points = 15; // Number of time points
const double t_min = 1e16;
const double t_max = 1e17;

struct Probabilities {
    double p_plus_x, p_minus_x, p_plus_y, p_minus_y, p_plus_z, p_minus_z;
};

Probabilities compute_probabilities(double F) {
    double B = 1.0 / (2 * cosh(F / 2.0) + 4);
    Probabilities p;
    p.p_plus_x = B * exp(F / 2.0);
    p.p_minus_x = B * exp(-F / 2.0);
    p.p_plus_y = B;
    p.p_minus_y = B;
    p.p_plus_z = B;
    p.p_minus_z = B;
    return p;
}

std::vector<double> generate_log_times(int n, double t_min, double t_max) {
    std::vector<double> times(n);
    double log_min = log10(t_min);
    double log_max = log10(t_max);
    double delta = (log_max - log_min) / (n - 1);
    for (int i = 0; i < n; ++i) {
        times[i] = pow(10, log_min + i * delta);
    }
    return times;
}

uint64_t generate_steps(double alpha, double t, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform_0_1(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_theta(0.0, M_PI);
    double theta = uniform_theta(rng);
    double x = uniform_0_1(rng);
    double W = -log(x);
    double a_theta_num = sin((1 - alpha) * theta) * pow(sin(alpha * theta), alpha / (1 - alpha));
    double a_theta_den = pow(sin(theta), 1 / (1 - alpha));
    double a_theta = a_theta_num / a_theta_den;
    double eta = pow(a_theta / W, (1 - alpha) / alpha);
    uint64_t steps = static_cast<uint64_t>(std::ceil(pow(t / eta, alpha)));
    if (steps < 1) steps = 1;
    return steps;
}

int64_t simulate_trajectory(int w, const Probabilities& p, double alpha, double t, std::mt19937_64& rng) {
    uint64_t steps = generate_steps(alpha, t, rng);
    int64_t x = 0;
    int y = 0, z = 0;
    std::uniform_real_distribution<double> uniform_0_1(0.0, 1.0);
    double cum_prob_plus_x = p.p_plus_x;
    double cum_prob_minus_x = cum_prob_plus_x + p.p_minus_x;
    double cum_prob_plus_y = cum_prob_minus_x + p.p_plus_y;
    double cum_prob_minus_y = cum_prob_plus_y + p.p_minus_y;
    double cum_prob_plus_z = cum_prob_minus_y + p.p_plus_z;
    double cum_prob_minus_z = 1.0;
    int64_t x_displacement_sum = 0;
    for (uint64_t step = 0; step < steps; ++step) {
        double r = uniform_0_1(rng);
        if (r < cum_prob_plus_x) {
            x += 1;
            x_displacement_sum += 1;
        } else if (r < cum_prob_minus_x) {
            x -= 1;
            x_displacement_sum -= 1;
        } else if (r < cum_prob_plus_y) {
            y = (y + 1) % w;
        } else if (r < cum_prob_minus_y) {
            y = (y - 1 + w) % w;
        } else if (r < cum_prob_plus_z) {
            z = (z + 1) % w;
        } else {
            z = (z - 1 + w) % w;
        }
    }
    return x_displacement_sum;
}

void worker(int w, const Probabilities& p, double alpha, double t, int traj_start, int traj_end,
            double& accumulator, std::mutex& acc_mutex) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    rng.discard(traj_start * 1000);
    double local_sum = 0.0;
    for (int i = traj_start; i < traj_end; ++i) {
        int64_t x_disp = simulate_trajectory(w, p, alpha, t, rng);
        local_sum += static_cast<double>(x_disp);
    }
    std::lock_guard<std::mutex> lock(acc_mutex);
    accumulator += local_sum;
}

// Linear regression: returns {alpha_fit, mu_fit}
std::pair<double, double> fit_power_law(const std::vector<double>& times, const std::vector<double>& x_averages) {
    int n = times.size();
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int i = 0; i < n; ++i) {
        double x = std::log(times[i]);
        double y = std::log(std::max(x_averages[i], 1e-10));
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double denom = n * sum_xx - sum_x * sum_x;
    double alpha_fit = (n * sum_xy - sum_x * sum_y) / denom;
    double ln_mu_fit = (sum_y - alpha_fit * sum_x) / n;
    double mu_fit = std::exp(ln_mu_fit);
    return {alpha_fit, mu_fit};
}

int main() {
    Probabilities p = compute_probabilities(F);
    std::vector<double> times = generate_log_times(measurement_points, t_min, t_max);
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;

    std::ofstream outfile("enhanced_sim_results.csv");
    outfile << "w,t,average_x_displacement" << std::endl;

    std::cout << "Running simulation with " << measurement_points << " time points from "
              << t_min << " to " << t_max << std::endl;
    std::cout << "Using " << n_threads << " threads for " << N_traj << " trajectories each." << std::endl;

    for (int w : Ws) {
        std::vector<double> x_averages;
        std::cout << "\nProcessing w = " << w << ":" << std::endl;
        for (int m = 0; m < measurement_points; ++m) {
            double t = times[m];
            double x_accumulator = 0.0;
            std::mutex acc_mutex;
            int traj_per_thread = N_traj / n_threads;
            std::vector<std::thread> threads;
            for (unsigned int thread_id = 0; thread_id < n_threads; ++thread_id) {
                int start_idx = thread_id * traj_per_thread;
                int end_idx = (thread_id == n_threads - 1) ? N_traj : start_idx + traj_per_thread;
                threads.emplace_back(worker, w, std::ref(p), alpha_input, t, start_idx, end_idx, std::ref(x_accumulator), std::ref(acc_mutex));
            }
            for (auto& th : threads) {
                th.join();
            }
            double average_x = x_accumulator / static_cast<double>(N_traj);
            x_averages.push_back(average_x);
            std::cout << "  t=" << t << ", <x>=" << average_x << std::endl;
            outfile << w << "," << t << "," << average_x << std::endl;
        }
        // Fit both alpha and mu
        auto fit = fit_power_law(times, x_averages);
        std::cout << "Fitted alpha for w=" << w << ": " << fit.first << std::endl;
        std::cout << "Fitted mu for w=" << w << ": " << fit.second << std::endl;
    }
    outfile.close();
    std::cout << "\nSimulation complete! Results saved to enhanced_sim_results.csv" << std::endl;
    return 0;
}

// Compile with: g++ -std=c++17 -O3 -pthread full_sim_quicktest.cpp -o full_sim_quicktest
// Run with: ./full_sim_quicktest
// Output files: full_sim_results.csv, mu_vs_w.csv