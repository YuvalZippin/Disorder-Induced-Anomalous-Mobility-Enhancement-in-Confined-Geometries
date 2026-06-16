// engine2_server_norm.cpp
// PRO-Grade Subordination Engine - Server Edition (64 Threads)
// Compile: g++ -std=c++17 -O3 -march=broadwell -fopenmp -o engine2_server_norm engine2_server_norm.cpp
// Run:     export OMP_NUM_THREADS=64 && ./engine2_server_norm

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <array>

namespace Config {
    // INCREASED for high precision on 64-core server
    constexpr uint64_t N_TRAJ = 50000;         
    constexpr double T_TARGET = 1e7;            
    constexpr double F_PHYS = 0.05;              
    constexpr double KAPPA_ALPHA = 10.0;         
    constexpr uint64_t MASTER_SEED = 0x9e3779b97f4a7c15ULL;
    
    const std::array<double, 8> ALPHAS = {0.2, 0.4, 0.6, 0.8};
    const std::array<int, 4> WIDTHS = {5, 10, 20, 40};
    const std::string CSV_FILE = "results_graph2_norm_server.csv";
}

struct alignas(64) ThreadData {
    double x_sum = 0.0;
    uint64_t total_steps = 0;
};

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
    CMSParams(double a) : one_minus_alpha(1.0 - a), alpha_ratio(a / (1.0 - a)), inv_alpha((1.0 - a) / a) {}
};

[[nodiscard]] inline uint64_t generate_qtm_steps(const double t, const double alpha, const CMSParams& cms, std::mt19937_64& rng, std::uniform_real_distribution<double>& dist_u, std::uniform_real_distribution<double>& dist_th) noexcept {
    const double theta = dist_th(rng);
    const double w_exp = -std::log(dist_u(rng));
    const double num = std::sin(cms.one_minus_alpha * theta) * std::pow(std::sin(alpha * theta), cms.alpha_ratio);
    const double den = std::pow(std::sin(theta), 1.0 / cms.one_minus_alpha);
    const double eta = std::pow(num / (den * w_exp), cms.inv_alpha);
    const uint64_t steps = static_cast<uint64_t>(std::ceil(std::pow(t / eta, alpha)));
    return (steps < 1ULL) ? 1ULL : steps;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const int max_threads = omp_get_max_threads();
    std::cout << "[SYSTEM] Initializing Engine 2 (Server Edition) on " << max_threads << " threads.\n";
    std::cout << "[SYSTEM] Trajectories per point: " << Config::N_TRAJ << "\n\n";

    std::ofstream csv(Config::CSV_FILE);
    csv << "alpha,w,N_traj,avg_x,norm_Y,ln_norm_Y,mean_steps\n";

    for (const double alpha : Config::ALPHAS) {
        const CMSParams cms_params(alpha);

        for (const int w : Config::WIDTHS) {
            auto start_time = std::chrono::high_resolution_clock::now();

            const double F_eff = Config::KAPPA_ALPHA * std::pow(Config::F_PHYS, alpha) * std::pow(static_cast<double>(w), -2.0 * (1.0 - alpha));
            const double norm_factor_prob = 1.0 / (2.0 * std::cosh(F_eff / 2.0) + 4.0);
            
            const double c1 = norm_factor_prob * std::exp(F_eff / 2.0);
            const double c2 = c1 + norm_factor_prob * std::exp(-F_eff / 2.0);

            std::vector<ThreadData> thread_stats(max_threads);

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                uint64_t state = Config::MASTER_SEED ^ static_cast<uint64_t>(w) ^ static_cast<uint64_t>(alpha * 10000.0);
                for (int i = 0; i <= tid; ++i) state = splitmix64(state);
                
                std::mt19937_64 rng(state);
                std::uniform_real_distribution<double> dist_u(0.0, 1.0);
                std::uniform_real_distribution<double> dist_th(0.0, M_PI);

                // Optimized dynamic scheduling for heavy-tailed wait times
                #pragma omp for schedule(dynamic, 256)
                for (uint64_t n = 0; n < Config::N_TRAJ; ++n) {
                    const uint64_t steps = generate_qtm_steps(Config::T_TARGET, alpha, cms_params, rng, dist_u, dist_th);
                    thread_stats[tid].total_steps += steps;
                    int64_t x_disp = 0;

                    for (uint64_t s = 0; s < steps; ++s) {
                        const double r = dist_u(rng);
                        if (r < c1) ++x_disp;
                        else if (r < c2) --x_disp;
                    }
                    thread_stats[tid].x_sum += static_cast<double>(x_disp);
                }
            }

            double total_x = 0.0;
            uint64_t total_steps = 0;
            for (const auto& ts : thread_stats) {
                total_x += ts.x_sum;
                total_steps += ts.total_steps;
            }

            const double avg_x = total_x / static_cast<double>(Config::N_TRAJ);
            const double mean_steps = static_cast<double>(total_steps) / static_cast<double>(Config::N_TRAJ);

            // --- NORMALIZATION BLOCK ---
            const double A_const = 1.0; 
            const double D_0 = 1.0 / 6.0; 
            const double A_alpha = A_const * std::tgamma(1.0 + alpha);
            
            const double normalizer = std::pow(Config::F_PHYS, alpha) * std::pow(Config::T_TARGET, alpha);
            const double norm_Y = (avg_x * A_alpha) / normalizer;
            const double ln_norm_Y = std::log(norm_Y);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;

            std::cout << std::fixed << std::setprecision(3)
                      << "[T: " << duration.count() << "s] "
                      << "Alpha = " << std::setprecision(2) << alpha 
                      << " | w = " << std::setw(2) << w 
                      << " | ln(norm_Y) = " << std::scientific << std::setprecision(5) << ln_norm_Y << "\n";

            csv << std::fixed << std::setprecision(2) << alpha << "," 
                << w << "," << Config::N_TRAJ << "," 
                << std::scientific << std::setprecision(16) << avg_x << "," 
                << norm_Y << "," << ln_norm_Y << "," << mean_steps << "\n";
        }
    }
    csv.close();
    std::cout << "\n[SYSTEM] Run fully completed. Outputs dumped to: " << Config::CSV_FILE << "\n";
    return 0;
}