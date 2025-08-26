#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <thread>
#include <atomic>
#include <immintrin.h>

class UltraFastMSD {
private:
    // Parameters
    double alpha;
    double F;
    int Ly, Lz;
    int num_sims;
    
    // Pre-computed values for speed
    double exp_F2, exp_negF2, inv_A;
    std::vector<double> cumulative_probs;
    
    // Thread-local RNG
    thread_local static std::mt19937_64 rng;
    thread_local static std::uniform_real_distribution<double> uniform_dist;
    
    // Vectorized Levy stable generation (optimized)
    inline double generate_eta_fast() {
        double theta = M_PI * uniform_dist(rng);
        double W = -std::log(uniform_dist(rng));
        
        // Optimized computation using precomputed values
        double sin_theta = std::sin(theta);
        double sin_alpha_theta = std::sin(alpha * theta);
        double sin_1_alpha_theta = std::sin((1.0 - alpha) * theta);
        
        double a_theta = sin_1_alpha_theta * 
                        std::pow(sin_alpha_theta / sin_theta, alpha / (1.0 - alpha)) /
                        std::pow(sin_theta, 1.0 / (1.0 - alpha));
        
        return std::pow(a_theta / W, (1.0 - alpha) / alpha);
    }
    
    inline double compute_S_alpha_fast(double t) {
        double eta = generate_eta_fast();
        return std::pow(t / eta, alpha);
    }
    
    // Ultra-optimized single trajectory
    std::array<long long, 3> run_single_trajectory_fast(double t) {
        double S_alpha = compute_S_alpha_fast(t);
        
        if (!std::isfinite(S_alpha) || S_alpha <= 0) {
            return {0, 0, 0};
        }
        
        long long pos_x = 0, pos_y = 0, pos_z = 0;
        long long n_steps = 0;
        long long target_steps = static_cast<long long>(S_alpha);
        
        // Batch processing for better cache performance
        const int batch_size = 1024;
        
        while (n_steps < target_steps) {
            long long remaining = target_steps - n_steps;
            long long current_batch = std::min(static_cast<long long>(batch_size), remaining);
            
            for (long long i = 0; i < current_batch; ++i) {
                double r = uniform_dist(rng);
                
                // Optimized branching using precomputed cumulative probabilities
                if (r < cumulative_probs[0]) {
                    pos_x++;
                } else if (r < cumulative_probs[1]) {
                    pos_x--;
                } else if (r < cumulative_probs[2]) {
                    pos_y++;
                } else if (r < cumulative_probs[3]) {
                    pos_y--;
                } else if (r < cumulative_probs[4]) {
                    pos_z++;
                } else {
                    pos_z--;
                }
            }
            
            n_steps += current_batch;
            
            // Apply periodic boundary conditions in batches
            pos_y = ((pos_y % Ly) + Ly) % Ly;
            pos_z = ((pos_z % Lz) + Lz) % Lz;
        }
        
        return {pos_x, pos_y, pos_z};
    }
    
public:
    UltraFastMSD(double alpha_val, double F_val, int Ly_val, int Lz_val, int num_sims_val) 
        : alpha(alpha_val), F(F_val), Ly(Ly_val), Lz(Lz_val), num_sims(num_sims_val) {
        
        // Pre-compute probabilities for speed
        exp_F2 = std::exp(F / 2.0);
        exp_negF2 = std::exp(-F / 2.0);
        double A = 4.0 + exp_F2 + exp_negF2;
        inv_A = 1.0 / A;
        
        cumulative_probs = {
            exp_F2 * inv_A,
            (exp_F2 + exp_negF2) * inv_A,
            (exp_F2 + exp_negF2 + 1.0) * inv_A,
            (exp_F2 + exp_negF2 + 2.0) * inv_A,
            (exp_F2 + exp_negF2 + 3.0) * inv_A,
            1.0
        };
    }
    
    // Thread worker function
    void worker_thread(double t, int start_sim, int end_sim, 
                      std::atomic<double>& sum_x2, std::atomic<double>& sum_y2, std::atomic<double>& sum_z2,
                      std::atomic<int>& completed) {
        
        double local_sum_x2 = 0.0, local_sum_y2 = 0.0, local_sum_z2 = 0.0;
        
        for (int i = start_sim; i < end_sim; ++i) {
            auto final_pos = run_single_trajectory_fast(t);
            local_sum_x2 += static_cast<double>(final_pos[0]) * final_pos[0];
            local_sum_y2 += static_cast<double>(final_pos[1]) * final_pos[1];
            local_sum_z2 += static_cast<double>(final_pos[2]) * final_pos[2];
        }
        
        // Atomic accumulation
        double expected = sum_x2.load();
        while (!sum_x2.compare_exchange_weak(expected, expected + local_sum_x2));
        
        expected = sum_y2.load();
        while (!sum_y2.compare_exchange_weak(expected, expected + local_sum_y2));
        
        expected = sum_z2.load();
        while (!sum_z2.compare_exchange_weak(expected, expected + local_sum_z2));
        
        completed += (end_sim - start_sim);
    }
    
    std::array<double, 3> calculate_msd_parallel(double t) {
        std::atomic<double> sum_x2{0.0}, sum_y2{0.0}, sum_z2{0.0};
        std::atomic<int> completed{0};
        
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        
        int sims_per_thread = num_sims / num_threads;
        int remaining_sims = num_sims % num_threads;
        
        for (int i = 0; i < num_threads; ++i) {
            int start_sim = i * sims_per_thread;
            int end_sim = (i == num_threads - 1) ? start_sim + sims_per_thread + remaining_sims 
                                                 : start_sim + sims_per_thread;
            
            threads.emplace_back(&UltraFastMSD::worker_thread, this, t, start_sim, end_sim,
                               std::ref(sum_x2), std::ref(sum_y2), std::ref(sum_z2), std::ref(completed));
        }
        
        // Progress monitoring
        while (completed.load() < num_sims) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            double progress = 100.0 * completed.load() / num_sims;
            std::cout << "\rProgress for t=" << t << ": " << std::fixed << std::setprecision(1) 
                      << progress << "%" << std::flush;
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        double mean_x2 = sum_x2.load() / num_sims;
        double mean_y2 = sum_y2.load() / num_sims;
        double mean_z2 = sum_z2.load() / num_sims;
        
        std::cout << " - Done!" << std::endl;
        return {mean_x2, mean_y2, mean_z2};
    }
    
    void run_msd_analysis(const std::vector<double>& t_values, const std::string& output_file) {
        std::ofstream file(output_file);
        file << "t,msd_x,msd_y,msd_z\n";
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < t_values.size(); ++i) {
            double t = t_values[i];
            
            std::cout << "\n=== Processing t = " << t << " (" << (i+1) << "/" << t_values.size() << ") ===\n";
            
            auto start_t = std::chrono::high_resolution_clock::now();
            auto msd = calculate_msd_parallel(t);
            auto end_t = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_t - start_t);
            
            std::cout << "MSD results: X²=" << msd[0] << ", Y²=" << msd[1] << ", Z²=" << msd[2] 
                      << " (Time: " << duration.count() << "s)\n";
            
            file << std::scientific << t << "," << msd[0] << "," << msd[1] << "," << msd[2] << "\n";
            file.flush(); // Save immediately
        }
        
        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(end_total - start_total);
        
        std::cout << "\n=== Analysis Complete! Total time: " << total_duration.count() << " minutes ===\n";
        file.close();
    }
};

// Thread-local static members initialization
thread_local std::mt19937_64 UltraFastMSD::rng(std::chrono::steady_clock::now().time_since_epoch().count());
thread_local std::uniform_real_distribution<double> UltraFastMSD::uniform_dist(0.0, 1.0);

int main() {
    // Parameters - adjust as needed
    double ALPHA = 0.5;
    double FORCE = 0.1;
    int WIDTH = 30;
    int HEIGHT = 30;
    int NUM_SIMS = 10000;  // Start with smaller number for very large t values
    
    // Create logarithmic time range from 10^0 to 10^14
    std::vector<double> t_values;
    
    std::cout << "Generating time values from 10^0 to 10^14...\n";
    
    // More strategic sampling for extreme range
    for (double exponent = 0; exponent <= 14; exponent += 0.5) {
        t_values.push_back(std::pow(10.0, exponent));
    }
    
    std::cout << "Generated " << t_values.size() << " time points.\n";
    std::cout << "First few values: ";
    for (int i = 0; i < std::min(5, (int)t_values.size()); ++i) {
        std::cout << t_values[i] << " ";
    }
    std::cout << "\nLast few values: ";
    for (int i = std::max(0, (int)t_values.size()-5); i < t_values.size(); ++i) {
        std::cout << std::scientific << t_values[i] << " ";
    }
    std::cout << std::endl;
    
    // Create the calculator
    UltraFastMSD calculator(ALPHA, FORCE, WIDTH, HEIGHT, NUM_SIMS);
    
    // Run analysis
    std::string output_file = "msd_results_ultra_long.csv";
    std::cout << "\nStarting MSD analysis with " << NUM_SIMS << " simulations per time point...\n";
    std::cout << "Results will be saved to: " << output_file << "\n";
    
    calculator.run_msd_analysis(t_values, output_file);
    
    std::cout << "\n=== Python plotting code ===\n";
    std::cout << "import pandas as pd\n";
    std::cout << "import matplotlib.pyplot as plt\n";
    std::cout << "import numpy as np\n\n";
    std::cout << "# Load data\n";
    std::cout << "df = pd.read_csv('" << output_file << "')\n\n";
    std::cout << "# Plot\n";
    std::cout << "plt.figure(figsize=(12, 8))\n";
    std::cout << "plt.loglog(df['t'], df['msd_x'], 'o-', label='MSD <X²>', markersize=4)\n";
    std::cout << "plt.loglog(df['t'], df['msd_y'], 's-', label='MSD <Y²>', markersize=4)\n";
    std::cout << "plt.loglog(df['t'], df['msd_z'], '^-', label='MSD <Z²>', markersize=4)\n";
    std::cout << "plt.xlabel('Time t')\n";
    std::cout << "plt.ylabel('MSD')\n";
    std::cout << "plt.title('MSD vs Time (10^0 to 10^14)')\n";
    std::cout << "plt.legend()\n";
    std::cout << "plt.grid(True, alpha=0.3)\n";
    std::cout << "plt.show()\n";
    
    return 0;
}