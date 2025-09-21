#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <numeric>
#include <algorithm>

// Parameters for quick PC test
const int N_traj = 5000;               // Much smaller for quick turnaround
const double alpha = 0.3;
const std::vector<double> Forces = {0.01, 0.04, 0.25};
const std::vector<int> Ws = {1, 10, 100};
const int measurement_points = 8;      // Fewer time points for testing
const double t_min = 1e12;
const double t_max = 1e15;

// Probability vector structure
struct Probabilities {
    double p_plus_x, p_minus_x, p_plus_y, p_minus_y, p_plus_z, p_minus_z;
};

Probabilities compute_probabilities(double F) {
    double B = 1.0 / (2 * cosh(F / 2.0) + 4);
    return {B * exp(F / 2.0), B * exp(-F / 2.0), B, B, B, B};
}

std::vector<double> generate_log_times(int n, double t_min, double t_max) {
    std::vector<double> times(n);
    double log_min = log10(t_min), log_max = log10(t_max);
    double delta = (log_max - log_min) / (n - 1);
    for (int i = 0; i < n; ++i) times[i] = pow(10, log_min + i * delta);
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
    steps = std::min(steps, static_cast<uint64_t>(1e7)); // Lower cap for PC test
    return (steps < 1) ? 1 : steps;
}

int64_t simulate_trajectory(int w, const Probabilities& p, double alpha, double t, std::mt19937_64& rng) {
    uint64_t steps = generate_steps(alpha, t, rng);
    int64_t x = 0; int y = 0; int z = 0;
    std::uniform_real_distribution<double> uniform_0_1(0.0, 1.0);

    double cum[6];
    cum[0] = p.p_plus_x; cum[1] = cum[0] + p.p_minus_x;
    cum[2] = cum[1] + p.p_plus_y; cum[3] = cum[2] + p.p_minus_y;
    cum[4] = cum[3] + p.p_plus_z; cum[5] = 1.0;

    int64_t x_disp_sum = 0;
    for (uint64_t step = 0; step < steps; ++step) {
        double r = uniform_0_1(rng);
        if (r < cum[0]) { x++; x_disp_sum++; }
        else if (r < cum[1]) { x--; x_disp_sum--; }
        else if (r < cum[2]) { y = (y + 1) % w; }
        else if (r < cum[3]) { y = (y - 1 + w) % w; }
        else if (r < cum[4]) { z = (z + 1) % w; }
        else { z = (z - 1 + w) % z; }
    }
    return x_disp_sum;
}

void worker(int w, const Probabilities& p, double alpha, double t, int start, int end, std::atomic<double>& acc) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    rng.discard(start * 1000);
    double local_sum = 0.0;
    for (int i = start; i < end; ++i) {
        local_sum += static_cast<double>(simulate_trajectory(w, p, alpha, t, rng));
    }
    acc =+ local_sum;
}

double fit_mu(const std::vector<double>& times, const std::vector<double>& x_avgs, double alpha) {
    int n = times.size();
    std::vector<double> ln_t(n), ln_x(n);
    for (int i = 0; i < n; ++i) {
        ln_t[i] = log(times[i]);
        ln_x[i] = log(std::max(x_avgs[i], 1e-10));
    }
    double mean_ln_t = std::accumulate(ln_t.begin(), ln_t.end(), 0.0) / n;
    double mean_ln_x = std::accumulate(ln_x.begin(), ln_x.end(), 0.0) / n;
    return exp(mean_ln_x - alpha * mean_ln_t);
}

int main() {
    auto times = generate_log_times(measurement_points, t_min, t_max);
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;

    std::ofstream output_file("mu_vs_w_F_quicktest.csv");
    output_file << "F,w,mu\n";

    std::cout << "=== PC TEST: μ vs w for multiple F with reduced traj and points ===\n";

    for (double F : Forces) {
        Probabilities p = compute_probabilities(F);
        std::cout << "\nForce F = " << F << std::endl;

        for (int w : Ws) {
            std::vector<double> x_averages;

            for (int i = 0; i < measurement_points; ++i) {
                double t = times[i];
                std::atomic<double> x_accum(0.0);
                int per_thread = N_traj / n_threads;
                std::vector<std::thread> threads;

                for (unsigned int tid = 0; tid < n_threads; ++tid) {
                    int start = tid * per_thread;
                    int end = (tid == n_threads - 1) ? N_traj : start + per_thread;
                    threads.emplace_back(worker, w, std::ref(p), alpha, t, start, end, std::ref(x_accum));
                }

                for (auto& th : threads) th.join();

                double avg_x = x_accum / static_cast<double>(N_traj);
                x_averages.push_back(avg_x);

                std::cout << "  t=" << t << " <x>=" << avg_x << std::endl;
            }

            double mu = fit_mu(times, x_averages, alpha);
            std::cout << "  Fitted mu for w=" << w << ": " << mu << std::endl;
            output_file << F << "," << w << "," << mu << "\n";
        }
    }
    output_file.close();

    std::cout << "\nQuick test complete. Check mu_vs_w_F_quicktest.csv\n";
    return 0;
}
