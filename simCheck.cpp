#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <numeric>

class LevyStableGenerator {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    
    // Cached alpha-dependent constants for optimization
    mutable double cached_alpha;
    mutable double alpha_over_1_minus_alpha;
    mutable double one_over_1_minus_alpha;
    mutable double one_minus_alpha_over_alpha;
    mutable bool cache_valid;
    
    void update_alpha_cache(double alpha) const {
        if (!cache_valid || std::abs(cached_alpha - alpha) > 1e-12) {
            cached_alpha = alpha;
            double one_minus_alpha = 1.0 - alpha;
            alpha_over_1_minus_alpha = alpha / one_minus_alpha;
            one_over_1_minus_alpha = 1.0 / one_minus_alpha;
            one_minus_alpha_over_alpha = one_minus_alpha / alpha;
            cache_valid = true;
        }
    }
    
public:
    LevyStableGenerator() : rng(std::random_device{}()), uniform_dist(0.0, 1.0), 
                           cache_valid(false) {}
    
    double generate_eta(double alpha) {
        // Update cached constants if alpha changed
        update_alpha_cache(alpha);
        
        double theta = M_PI * uniform_dist(rng);
        double W = -std::log(uniform_dist(rng));
        
        // Optimized trigonometric calculations
        double sin_theta = std::sin(theta);
        double alpha_theta = alpha * theta;
        double sin_alpha_theta = std::sin(alpha_theta);
        double sin_1_minus_alpha_theta = std::sin(theta - alpha_theta);  // Using trig identity
        
        // Use cached exponents to avoid repeated division
        double sin_alpha_theta_power = std::pow(sin_alpha_theta, alpha_over_1_minus_alpha);
        double sin_theta_power = std::pow(sin_theta, one_over_1_minus_alpha);
        
        double a_theta = sin_1_minus_alpha_theta * sin_alpha_theta_power / sin_theta_power;
        
        // Use cached exponent
        double eta = std::pow(a_theta / W, one_minus_alpha_over_alpha);
        return eta;
    }
    
    double compute_S_alpha(double t, double alpha) {
        double eta = generate_eta(alpha);
        double S_alpha = std::pow(t / eta, alpha);
        return S_alpha;
    }
};

class RandomWalker3D {
private:
    LevyStableGenerator levy_gen;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    
    // Position
    double x, y, z;
    
    // System parameters
    int L;     // System size (L = W = H)
    double F;  // Force parameter
    
    // Probability calculation
    std::vector<double> calculate_probabilities(double F) {
        double exp_F2 = std::exp(F / 2.0);
        double exp_negF2 = std::exp(-F / 2.0);
        double A = 4.0 + exp_F2 + exp_negF2;
        
        return {
            exp_F2 / A,    // +X
            exp_negF2 / A, // -X
            1.0 / A,       // +Y
            1.0 / A,       // -Y
            1.0 / A,       // +Z
            1.0 / A        // -Z
        };
    }
    
    void make_jump() {
        std::vector<double> probs = calculate_probabilities(F);
        double rand_val = uniform_dist(rng);
        
        double cumulative = 0.0;
        for (int i = 0; i < 6; i++) {
            cumulative += probs[i];
            if (rand_val < cumulative) {
                switch(i) {
                    case 0: x += 1.0; break;  // +X
                    case 1: x -= 1.0; break;  // -X
                    case 2: y += 1.0; break;  // +Y
                    case 3: y -= 1.0; break;  // -Y
                    case 4: z += 1.0; break;  // +Z
                    case 5: z -= 1.0; break;  // -Z
                }
                break;
            }
        }
        
        // Apply periodic boundary conditions for Y and Z
        if (L > 0) {  // Handle L=0 case
            y = fmod(y + L, L);
            if (y < 0) y += L;
            
            z = fmod(z + L, L);
            if (z < 0) z += L;
        }
    }
    
public:
    RandomWalker3D(int system_size, double force) 
        : rng(std::random_device{}()), uniform_dist(0.0, 1.0), 
          L(system_size), F(force) {
        reset_position();
    }
    
    void reset_position() {
        x = 0.0;
        y = (L > 0) ? L / 2.0 : 0.0;  // Start at center or origin
        z = (L > 0) ? L / 2.0 : 0.0;  // Start at center or origin
    }
    
    double run_single_trajectory(double t, double alpha) {
        reset_position();
        double S_alpha = levy_gen.compute_S_alpha(t, alpha);
        int num_jumps = static_cast<int>(std::round(S_alpha));
        
        for (int i = 0; i < num_jumps; i++) {
            make_jump();
        }
        
        return x;  // Return only final X position
    }
};

class PowerLawFitter {
public:
    struct FitResult {
        double A;
        double beta;
        double r_squared;
        bool success;
        bool beta_valid;
    };
    
    static FitResult fit_power_law(const std::vector<double>& t_values, 
                                   const std::vector<double>& x_values,
                                   double expected_alpha = -1.0,
                                   double beta_tolerance = 0.005)
    {
        FitResult result = {0.0, 0.0, 0.0, false, false};
        
        if (t_values.size() != x_values.size() || t_values.size() < 3) {
            return result;
        }
        
        // Convert to log-log for linear regression
        // MODIFIED: Handle negative values by checking sign consistency
        std::vector<double> log_t, log_x;
        bool all_positive = true, all_negative = true;
        
        for (size_t i = 0; i < t_values.size(); i++) {
            if (t_values[i] > 0 && x_values[i] != 0) {
                if (x_values[i] > 0) all_negative = false;
                if (x_values[i] < 0) all_positive = false;
                
                log_t.push_back(std::log(t_values[i]));
                log_x.push_back(std::log(std::abs(x_values[i])));
            }
        }
        
        if (log_t.size() < 3) return result;
        
        // Linear regression: log(|x|) = log(|A|) + beta * log(t)
        size_t n = log_t.size();
        double sum_log_t = std::accumulate(log_t.begin(), log_t.end(), 0.0);
        double sum_log_x = std::accumulate(log_x.begin(), log_x.end(), 0.0);
        
        double sum_log_t_sq = 0.0, sum_log_t_log_x = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum_log_t_sq += log_t[i] * log_t[i];
            sum_log_t_log_x += log_t[i] * log_x[i];
        }
        
        double denominator = n * sum_log_t_sq - sum_log_t * sum_log_t;
        if (std::abs(denominator) < 1e-10) return result;
        
        result.beta = (n * sum_log_t_log_x - sum_log_t * sum_log_x) / denominator;
        double log_A = (sum_log_x - result.beta * sum_log_t) / n;
        
        // MODIFIED: Preserve sign information in coefficient A
        double A_magnitude = std::exp(log_A);
        if (all_positive) {
            result.A = A_magnitude;
        } else if (all_negative) {
            result.A = -A_magnitude;
        } else {
            // Mixed signs - use magnitude (this case shouldn't happen with consistent bias)
            result.A = A_magnitude;
        }
        
        // Calculate R-squared
        double mean_log_x = sum_log_x / n;
        double ss_tot = 0.0, ss_res = 0.0;
        for (size_t i = 0; i < n; i++) {
            double predicted = log_A + result.beta * log_t[i];
            ss_res += (log_x[i] - predicted) * (log_x[i] - predicted);
            ss_tot += (log_x[i] - mean_log_x) * (log_x[i] - mean_log_x);
        }
        
        result.r_squared = (ss_tot > 0) ? 1.0 - ss_res / ss_tot : 0.0;
        result.success = true;
        
        // Beta validation check
        if (expected_alpha > 0) {
            double beta_diff = std::abs(result.beta - expected_alpha);
            result.beta_valid = (beta_diff <= beta_tolerance);
        } else {
            result.beta_valid = true;
        }
        
        return result;
    }
};

class SystemSizeAnalyzer {
public:
    static double calculate_mean_final_x_position(double t, double alpha, double F, 
                                                  int L, int num_sims) {
        RandomWalker3D walker(L, F);
        double sum_x = 0.0;
        
        #pragma omp parallel for reduction(+:sum_x)
        for (int i = 0; i < num_sims; i++) {
            RandomWalker3D local_walker(L, F);
            double final_x = local_walker.run_single_trajectory(t, alpha);
            sum_x += final_x;
        }
        
        return sum_x / num_sims;
    }
    
    static double measure_A_coefficient(const std::vector<double>& t_values, 
                                        double alpha, double F, int L, int num_sims,
                                        bool* beta_check_passed = nullptr)
    {
        std::vector<double> mean_x_values;
        
        std::cout << "    Measuring L=" << L << " (t_points=" << t_values.size() << ")..." << std::flush;
        
        for (double t : t_values) {
            if (t <= 0) continue;
            double mean_x = calculate_mean_final_x_position(t, alpha, F, L, num_sims);
            // MODIFIED: Use mean_x directly instead of std::abs(mean_x)
            mean_x_values.push_back(mean_x);
        }
        
        // Filter out t=0 for fitting
        std::vector<double> t_filtered;
        for (double t : t_values) {
            if (t > 0) t_filtered.push_back(t);
        }
        
        // Fit ⟨x(t)⟩ = A·t^β directly
        PowerLawFitter::FitResult fit = PowerLawFitter::fit_power_law(t_filtered, mean_x_values, alpha, 0.005);
        
        if (beta_check_passed != nullptr) {
            *beta_check_passed = fit.beta_valid;
        }
        
        if (fit.success && fit.r_squared > 0.8) {
            std::cout << " A=" << std::scientific << std::setprecision(3) << fit.A 
                      << " β=" << std::fixed << std::setprecision(3) << fit.beta
                      << " (R²=" << std::setprecision(3) << fit.r_squared 
                      << ", β-check:" << (fit.beta_valid ? "PASS" : "FAIL") << ")" << std::endl;
            return fit.A;
        } else {
            std::cout << " FAILED FIT (R²=" << std::fixed << std::setprecision(3) 
                      << fit.r_squared;
            if (fit.success) {
                std::cout << ", β=" << std::setprecision(3) << fit.beta 
                          << ", β-check:" << (fit.beta_valid ? "PASS" : "FAIL");
            }
            std::cout << ")" << std::endl;
            return -1.0;
        }
    }
    
    static void analyze_A_vs_system_size(int L_start, int L_finish, int L_step,
                                          const std::vector<double>& t_values,
                                          double alpha, double F, int num_sims,
                                          int num_repetitions, const std::string& filename) {
        std::ofstream file(filename);
        file << "System_Size_L,Mean_A_Coefficient,Std_A_Coefficient,Num_Repetitions,Valid_Fits,Beta_Check_Passed,Beta_Check_Rate\n";
        file << std::scientific << std::setprecision(6);
        
        std::cout << "\nSystem Size Analysis - A Coefficient vs L (Direct Mean Fitting)\n";
        std::cout << "================================================================\n";
        std::cout << "Fitting: ⟨x(t)⟩ = A·t^β (no absolute value)\n";  // MODIFIED: Updated description
        std::cout << "L range: " << L_start << " to " << L_finish << " (step: " << L_step << ")\n";
        std::cout << "Repetitions per L: " << num_repetitions << "\n";
        std::cout << "Simulations per time point: " << num_sims << "\n";
        std::cout << "Expected beta ≈ alpha = " << alpha << " (tolerance: ±0.005)\n";
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";
        
        for (int L = L_start; L <= L_finish; L += L_step) {
            std::cout << "Processing L=" << L << " (" << num_repetitions << " repetitions):\n";
            
            std::vector<double> A_values;
            int beta_checks_passed = 0;
            int total_valid_fits = 0;
            
            for (int rep = 0; rep < num_repetitions; rep++) {
                std::cout << "  Repetition " << (rep + 1) << "/" << num_repetitions << ": ";
                bool beta_check_result = false;
                double A = measure_A_coefficient(t_values, alpha, F, L, num_sims, &beta_check_result);
                if (A != -1.0) {  // MODIFIED: Check for failed fit differently since A can be negative
                    A_values.push_back(A);
                    total_valid_fits++;
                    if (beta_check_result) {
                        beta_checks_passed++;
                    }
                }
            }
            
            if (!A_values.empty()) {
                double mean_A = std::accumulate(A_values.begin(), A_values.end(), 0.0) / A_values.size();
                
                double variance = 0.0;
                for (double A : A_values) {
                    variance += (A - mean_A) * (A - mean_A);
                }
                double std_A = (A_values.size() > 1) ? std::sqrt(variance / (A_values.size() - 1)) : 0.0;
                
                double beta_check_rate = (total_valid_fits > 0) ? (double)beta_checks_passed / total_valid_fits : 0.0;
                
                file << L << "," << mean_A << "," << std_A << "," 
                     << num_repetitions << "," << A_values.size() << ","
                     << beta_checks_passed << "," << beta_check_rate << "\n";
                
                std::cout << "  RESULT L=" << L << ": Mean_A=" << std::scientific << std::setprecision(3) << mean_A
                          << " ± " << std_A << " (valid fits: " << A_values.size() << "/" << num_repetitions 
                          << ", β-checks: " << beta_checks_passed << "/" << total_valid_fits 
                          << " = " << std::fixed << std::setprecision(1) << (beta_check_rate * 100) << "%)\n\n";
            } else {
                std::cout << "  ERROR: No valid fits for L=" << L << "\n\n";
                file << L << ",NaN,NaN," << num_repetitions << ",0,0,0.0\n";
            }
        }
        
        file.close();
        std::cout << "Results saved to: " << filename << std::endl;
        std::cout << "Now fitting ⟨x(t)⟩ = A·t^β directly (preserving sign information)" << std::endl;
        std::cout << "Import this CSV into Excel/Google Sheets to analyze A vs L!" << std::endl;
    }
};

int main() {
    omp_set_num_threads(16);

    double alpha = 0.5;
    double F = 0.01;
    int num_sims = 150000;
    
    double t_start = 100.0;
    double t_finish = 1e5;
    double t_step = 1000.0;
    
    int L_start = 1;
    int L_finish = 3;
    int L_step = 1;

    int num_repetitions = 10;
    
    std::vector<double> t_values;
    for (double t = t_start; t <= t_finish; t += t_step) {
        t_values.push_back(t);
    }
    
    std::cout << "3D Random Walker - System Size Analysis (Direct Mean Fitting)\n";
    std::cout << "Parameters: alpha=" << alpha << ", F=" << F << std::endl;
    std::cout << "Time range: " << t_start << " to " << t_finish 
              << " (step: " << t_step << ", points: " << t_values.size() << ")" << std::endl;
    std::cout << "Fitting model: ⟨x(t)⟩ = A·t^β (no absolute value)" << std::endl;
    
    SystemSizeAnalyzer::analyze_A_vs_system_size(L_start, L_finish, L_step,
                                                  t_values, alpha, F, num_sims,
                                                  num_repetitions, "A_coefficient_vs_L.csv");
    
    return 0;
}