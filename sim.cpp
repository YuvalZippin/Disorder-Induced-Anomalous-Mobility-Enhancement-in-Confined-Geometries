#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

class LevyStableGenerator {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    
public:
    LevyStableGenerator() : rng(std::random_device{}()), uniform_dist(0.0, 1.0) {}
    
    double generate_eta(double alpha) {
        double theta = M_PI * uniform_dist(rng);
        double W = -std::log(uniform_dist(rng));
        
        double sin_theta = std::sin(theta);
        double sin_alpha_theta = std::sin(alpha * theta);
        double sin_1_minus_alpha_theta = std::sin((1.0 - alpha) * theta);
        
        double a_theta = sin_1_minus_alpha_theta 
                        * std::pow(sin_alpha_theta, alpha / (1.0 - alpha))
                        / std::pow(sin_theta, 1.0 / (1.0 - alpha));
        
        double eta = std::pow(a_theta / W, (1.0 - alpha) / alpha);
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
    int W, H;  // Y and Z boundary sizes
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
        y = fmod(y + W, W);
        if (y < 0) y += W;
        
        z = fmod(z + H, H);
        if (z < 0) z += H;
    }
    
public:
    RandomWalker3D(int W_size, int H_size, double force) 
        : rng(std::random_device{}()), uniform_dist(0.0, 1.0), 
          W(W_size), H(H_size), F(force) {
        reset_position();
    }
    
    void reset_position() {
        x = 0.0;
        y = W / 2.0;  // Start at center
        z = H / 2.0;  // Start at center
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

class SimulationRunner {
public:
    static double calculate_mean_final_x_position(double t, double alpha, double F, int W, int H, int num_sims) {
        RandomWalker3D walker(W, H, F);
        double sum_x = 0.0;
        
        for (int i = 0; i < num_sims; i++) {
            double final_x = walker.run_single_trajectory(t, alpha);
            sum_x += final_x;
        }
        return sum_x / num_sims;
    }
    static void analyze_mean_x_vs_time(const std::vector<double>& t_values,
                                         double alpha, double F, int W, int H,
                                         int num_sims, const std::string& filename) {
        std::ofstream file(filename);
        
        // Write CSV header
        file << "Time_t,Mean_X_Position,Num_Simulations\n";
        file << std::fixed << std::setprecision(6);
        
        std::cout << "Running simulations...\n";
        std::cout << "Time_t\tMean_X\tNum_Sims\n";
        std::cout << "------------------------\n";
        
        for (double t : t_values) {
            double mean_x = calculate_mean_final_x_position(t, alpha, F, W, H, num_sims);
            
            // Write to CSV file
            file << t << "," << mean_x << "," << num_sims << "\n";
            
            // Print to console
            std::cout << t << "\t" << mean_x << "\t" << num_sims << "\n";
        }
        
        file.close();
        std::cout << "\nResults saved to: " << filename << std::endl;
        std::cout << "You can now import this CSV file into Excel or Google Sheets!" << std::endl;
    }
};

// Example usage
int main() {
    // Simulation parameters
    double alpha = 0.5;      // Stability parameter
    double F = 1.0;          // Force parameter
    int W = 100;             // Y boundary size
    int H = 100;             // Z boundary size  
    int num_sims = 100000;     // Number of simulations per time point
    
    // Time range parameters
    double t_start = 0.0;    // Starting time value
    double t_finish = 1000.0;  // Ending time value
    double t_step = 100.0;     // Step size for time values

    // Generate time values
    std::vector<double> t_values;
    for (double t = t_start; t <= t_finish; t += t_step) {
        t_values.push_back(t);
    }
    
    std::cout << "3D Random Walker Simulation - X-axis Analysis\n";
    std::cout << "Parameters: alpha=" << alpha << ", F=" << F 
              << ", W=" << W << ", H=" << H << std::endl;
    std::cout << "Time range: " << t_start << " to " << t_finish 
              << " (step: " << t_step << ")" << std::endl;
    std::cout << "Total time points: " << t_values.size() << std::endl;
    std::cout << "Simulations per time point: " << num_sims << "\n\n";
    
    // Run analysis and save to CSV
    SimulationRunner::analyze_mean_x_vs_time(t_values, alpha, F, W, H, num_sims,"random_walker_results_01.csv");

    return 0;
}