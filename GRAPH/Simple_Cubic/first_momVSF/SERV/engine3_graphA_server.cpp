// engine3_graphA_server.cpp
// PRO-Grade Subordination Engine for Geometry A - Force (F) Dependence
// Optimized for Dual Intel Xeon E5-2683 v4 (O(1) Binomial Leap + Dynamic Scheduling)
// Compile: g++ -std=c++17 -O3 -march=broadwell -fopenmp -o engine3_graphA_server engine3_graphA_server.cpp
// Run:     export OMP_NUM_THREADS=64 && ./engine3_graphA_server

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <array>

// ---------------------------------------------------------------------
// PRO Configuration Parameters
// ---------------------------------------------------------------------
namespace Config {
    constexpr uint64_t N_TRAJ = 100000;          // High precision statistics
    constexpr double T_TARGET = 1e14;           // Observation time limit
    constexpr double ALPHA = 0.3;               // Fixed anomalous exponent
    constexpr double KAPPA_ALPHA = 10.0;        // Amplitude scale factor
    constexpr uint64_t MASTER_SEED = 0x9e3779b97f4a7c15ULL;
    
    // Test space: w
    const std::array<int, 3> WIDTHS = {5, 10, 25};
    const std::string CSV_FILE = "results_graph_a.csv";
}

// ---------------------------------------------------------------------
// Hardware-Aligned Memory Structures
// ---------------------------------------------------------------------
struct alignas(64) ThreadData {
    double x_sum = 0.0;
    uint64_t total_steps = 0;
};

// ---------------------------------------------------------------------
// High-Speed Math & PRNG Subroutines
// ---------------------------------------------------------------------
[[nodiscard]] static inline uint64_t splitmix64(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct CMSParams {
    const double one_minus_alpha;
    const double alpha_ratio;
    const double inv_alpha;
    
    CMSParams(double a) : 
        one_minus_alpha(1.0 - a), 
        alpha_ratio(a / (1.0 - a)), 
        inv_alpha((1.0 - a) / a) {}
};

[[nodiscard]] inline uint64_t generate_qtm_steps(const double t, 
                                                 const double alpha, 
                                                 const CMSParams& cms, 
                                                 std::mt19937_64& rng, 
                                                 std::uniform_real_distribution<double>& dist_u, 
                                                 std::uniform_real_distribution<double>& dist_th) noexcept 
{
    const double theta = dist_th(rng);
    const double w_exp = -std::log(dist_u(rng));
    
    const double num = std::sin(cms.one_minus_alpha * theta) * std::pow(std::sin(alpha * theta), cms.alpha_ratio);
    const double den = std::pow(std::sin(theta), 1.0 / cms.one_minus_alpha);
    const double eta = std::pow(num / (den * w_exp), cms.inv_alpha);
    
    const uint64_t steps = static_cast<uint64_t>(std::ceil(std::pow(t / eta, alpha)));
    return (steps < 1ULL) ? 1ULL : steps;
}

// ---------------------------------------------------------------------
// Main Simulation Pipeline
// ---------------------------------------------------------------------
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const int max_threads = omp_get_max_threads();
    std::cout << "[SYSTEM] Initializing Engine 3 (Force Dependence - Server Edition)\n";
    std::cout << "[SYSTEM] Hardware Threads: " << max_threads << "\n";
    std::cout << "[SYSTEM] Alpha = " << Config::ALPHA << " | T = " << std::scientific << Config::T_TARGET << "\n\n";

    // Generate log-spaced forces
    std::vector<double> forces;
    constexpr int NUM_F_POINTS = 10;
    constexpr double F_MIN = 0.01;
    constexpr double F_MAX = 1.0;
    for (int i = 0; i < NUM_F_POINTS; ++i) {
        double exp_val = std::log10(F_MIN) + i * (std::log10(F_MAX) - std::log10(F_MIN)) / (NUM_F_POINTS - 1);
        forces.push_back(std::pow(10.0, exp_val));
    }

    std::ofstream csv(Config::CSV_FILE);
    csv << "w,F,N_traj,avg_x,mean_steps\n";

    const CMSParams cms_params(Config::ALPHA);

    for (const int w : Config::WIDTHS) {
        for (const double F_phys : forces) {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Compute effective force based on theoretical expectation
            const double F_eff = Config::KAPPA_ALPHA * std::pow(F_phys, Config::ALPHA) * std::pow(static_cast<double>(w), -2.0 * (1.0 - Config::ALPHA));
            const double norm_factor = 1.0 / (2.0 * std::cosh(F_eff / 2.0) + 4.0);
            
            // OPTIMIZATION: Calculate explicit transition probabilities outside the loop
            const double P_plus = norm_factor * std::exp(F_eff / 2.0);
            const double P_minus = norm_factor * std::exp(-F_eff / 2.0);
            const double P_minus_cond = P_minus / (1.0 - P_plus);

            std::vector<ThreadData> thread_stats(max_threads);

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                
                // Deterministic local seeding mixing w and F
                uint64_t state = Config::MASTER_SEED ^ static_cast<uint64_t>(w) ^ static_cast<uint64_t>(F_phys * 100000.0);
                for (int i = 0; i <= tid; ++i) state = splitmix64(state);
                
                std::mt19937_64 rng(state);
                std::uniform_real_distribution<double> dist_u(0.0, 1.0);
                std::uniform_real_distribution<double> dist_th(0.0, M_PI);

                // OPTIMIZATION: Dynamic scheduling for heavy-tailed wait times
                #pragma omp for schedule(dynamic, 256)
                for (uint64_t n = 0; n < Config::N_TRAJ; ++n) {
                    const uint64_t steps = generate_qtm_steps(Config::T_TARGET, Config::ALPHA, cms_params, rng, dist_u, dist_th);
                    thread_stats[tid].total_steps += steps;

                    // ALGORITHMIC LEAP: O(1) Binomial sampling
                    std::binomial_distribution<uint64_t> binom_plus(steps, P_plus);
                    uint64_t steps_plus = binom_plus(rng);

                    std::binomial_distribution<uint64_t> binom_minus(steps - steps_plus, P_minus_cond);
                    uint64_t steps_minus = binom_minus(rng);

                    int64_t x_disp = static_cast<int64_t>(steps_plus) - static_cast<int64_t>(steps_minus);
                    thread_stats[tid].x_sum += static_cast<double>(x_disp);
                }
            }

            // O(1) Reduction
            double total_x = 0.0;
            uint64_t total_steps = 0;
            for (const auto& ts : thread_stats) {
                total_x += ts.x_sum;
                total_steps += ts.total_steps;
            }

            const double avg_x = total_x / static_cast<double>(Config::N_TRAJ);
            const double mean_steps = static_cast<double>(total_steps) / static_cast<double>(Config::N_TRAJ);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;

            std::cout << std::fixed << std::setprecision(3)
                      << "[T: " << duration.count() << "s] "
                      << "w = " << std::setw(2) << w 
                      << " | F = " << std::fixed << std::setprecision(4) << F_phys 
                      << " | <x> = " << std::scientific << std::setprecision(5) << avg_x 
                      << " | Avg Steps = " << std::scientific << std::setprecision(2) << mean_steps << "\n";

            csv << w << "," << std::fixed << std::setprecision(17) << F_phys << "," << Config::N_TRAJ << "," 
                << std::scientific << std::setprecision(17) << avg_x << "," 
                << mean_steps << "\n";
        }
    }

    csv.close();
    std::cout << "\n[SYSTEM] Run fully completed. Outputs dumped to: " << Config::CSV_FILE << "\n";
    return 0;
}