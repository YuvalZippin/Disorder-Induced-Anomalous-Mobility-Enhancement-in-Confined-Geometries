// sim_engine.cpp
// PRO-Grade Monte Carlo simulation engine for Geometry A (Rectangular Channel, PBC)
// Optimized for Dual Intel Xeon E5-2683 v4 (64 logical threads, 128GB RAM)
// Compile: g++ -std=c++17 -O3 -march=broadwell -fopenmp -o sim_engine sim_engine.cpp
// run: ./sim_engine

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
    constexpr uint64_t N_TRAJ = 100000;         // High-precision production runs
    constexpr double T_TARGET = 1e14;            // Observation time limit
    constexpr double ALPHA = 0.3;                // Anomalous diffusion exponent
    constexpr double KAPPA_ALPHA = 10.0;         // Amplitude scale factor
    constexpr uint64_t MASTER_SEED = 0x9e3779b97f4a7c15ULL;
    
    // Test space: w and F
    const std::array<int, 5> WIDTHS = {1, 3, 5, 7, 9};
    const std::array<double, 3> FORCES = {0.01, 0.02, 0.05};
    const std::string CSV_FILE = "results_geometry_a.csv";
}

// ---------------------------------------------------------------------
// Hardware-Aligned Memory Structures (Avoids False Sharing)
// ---------------------------------------------------------------------
struct alignas(64) ThreadData {
    double x_sum = 0.0;
    uint64_t total_steps = 0;
};

// ---------------------------------------------------------------------
// High-Speed Math & PRNG Subroutines
// ---------------------------------------------------------------------
// Fast, lock-free deterministic seed distribution
[[nodiscard]] static inline uint64_t splitmix64(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Precomputed CMS parameters for alpha = 0.3
struct CMSParams {
    const double one_minus_alpha;
    const double alpha_ratio;
    const double inv_alpha;
    
    CMSParams(double a) : 
        one_minus_alpha(1.0 - a), 
        alpha_ratio(a / (1.0 - a)), 
        inv_alpha((1.0 - a) / a) {}
};

// Highly optimized CMS method for generating anomalous waiting times
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
    std::cout << "[SYSTEM] Initializing PRO Simulation Engine\n";
    std::cout << "[SYSTEM] Hardware Threads: " << max_threads << "\n";
    std::cout << "[SYSTEM] Geometry: A (Simple Cubic, PBC)\n";
    std::cout << "[SYSTEM] Trajectories: " << Config::N_TRAJ << " @ t = " << Config::T_TARGET << "\n\n";

    std::ofstream csv(Config::CSV_FILE);
    csv << "w,F,N_traj,avg_x\n";

    const CMSParams cms_params(Config::ALPHA);

    for (const int w : Config::WIDTHS) {
        for (const double F_phys : Config::FORCES) {
            auto start_time = std::chrono::high_resolution_clock::now();

            // 1. Compute physical transition probabilities
            const double F_eff = Config::KAPPA_ALPHA * std::pow(F_phys, Config::ALPHA) * std::pow(static_cast<double>(w), -2.0 * (1.0 - Config::ALPHA));
            const double norm_factor = 1.0 / (2.0 * std::cosh(F_eff / 2.0) + 4.0);
            
            // Cumulative thresholds for x-axis only.
            // Transverse (y,z) movements are absorbed into the remainder of the probability space [c2, 1.0]
            // We do not need to track them because of translational invariance in Geometry A.
            const double c1 = norm_factor * std::exp(F_eff / 2.0);
            const double c2 = c1 + norm_factor * std::exp(-F_eff / 2.0);

            std::vector<ThreadData> thread_stats(max_threads);

            // 2. OpenMP Parallel Monte Carlo Engine
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                
                // Deterministic local seeding
                uint64_t state = Config::MASTER_SEED ^ static_cast<uint64_t>(w) ^ static_cast<uint64_t>(F_phys * 1000.0);
                for (int i = 0; i <= tid; ++i) state = splitmix64(state);
                
                std::mt19937_64 rng(state);
                std::uniform_real_distribution<double> dist_u(0.0, 1.0);
                std::uniform_real_distribution<double> dist_th(0.0, M_PI);

                #pragma omp for schedule(static)
                for (uint64_t n = 0; n < Config::N_TRAJ; ++n) {
                    const uint64_t steps = generate_qtm_steps(Config::T_TARGET, Config::ALPHA, cms_params, rng, dist_u, dist_th);
                    thread_stats[tid].total_steps += steps;

                    int64_t x_disp = 0;

                    // Extremely tight inner loop. 
                    // Y and Z spatial tracking removed due to PBC translational invariance.
                    for (uint64_t s = 0; s < steps; ++s) {
                        const double r = dist_u(rng);
                        if (r < c1) {
                            ++x_disp;
                        } else if (r < c2) {
                            --x_disp;
                        }
                    }
                    thread_stats[tid].x_sum += static_cast<double>(x_disp);
                }
            }

            // 3. O(1) Reduction
            double total_x = 0.0;
            uint64_t total_steps = 0;
            for (const auto& ts : thread_stats) {
                total_x += ts.x_sum;
                total_steps += ts.total_steps;
            }

            const double avg_x = total_x / static_cast<double>(Config::N_TRAJ);
            const double mean_steps = static_cast<double>(total_steps) / static_cast<double>(Config::N_TRAJ);

            // 4. Profiling and I/O
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;

            std::cout << std::fixed << std::setprecision(3)
                      << "[T: " << duration.count() << "s] "
                      << "w = " << w << " | F = " << F_phys 
                      << " | <x> = " << std::scientific << std::setprecision(6) << avg_x 
                      << " | Avg Steps = " << std::fixed << std::setprecision(1) << mean_steps << "\n";

            csv << w << "," << F_phys << "," << Config::N_TRAJ << "," 
                << std::scientific << std::setprecision(16) << avg_x << "\n";
        }
    }

    csv.close();
    std::cout << "\n[SYSTEM] Run fully completed. Outputs securely dumped to: " << Config::CSV_FILE << "\n";
    return 0;
}