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
    
    double get_x_position() {
        return x;
    }
    
    void perform_single_jump() {
        make_jump();
    }
};

class FirstMomentAnalyzer {
public:
    static void analyze_first_moment_vs_jumps(double t, double alpha, double F, 
                                             int W, int H, int measurement_interval,
                                             const std::string& filename) {
        
        // Step 1: Calculate S_alpha for optimal time to get total number of jumps N
        LevyStableGenerator levy_gen;
        double S_alpha = levy_gen.compute_S_alpha(t, alpha);
        long long N = static_cast<long long>(std::round(S_alpha));
        
        std::cout << "=== FIRST MOMENT ANALYSIS ===\n";
        std::cout << "Optimal laboratory time t = " << std::scientific << t << std::endl;
        std::cout << "Parameters: alpha = " << alpha << ", F = " << F << std::endl;
        std::cout << "Calculated S_alpha = " << S_alpha << std::endl;
        std::cout << "Total number of jumps N = " << N << std::endl;
        std::cout << "Measurement every " << measurement_interval << " jumps\n\n";
        
        // Step 2: Initialize random walker
        RandomWalker3D walker(W, H, F);
        
        // Step 3: Open CSV file for output
        std::ofstream file(filename);
        file << "Jump_Number,First_Moment_M,X_Position\n";
        file << std::fixed << std::setprecision(8);
        
        std::cout << "Jump_Number\tFirst_Moment_M\tX_Position\n";
        std::cout << "-------------------------------------------\n";
        
        // Step 4: Perform the walk and calculate first moment M every few jumps
        double running_sum_x = 0.0;
        long long measurement_count = 0;
        
        for (long long jump = 1; jump <= N; jump++) {
            // Perform one jump
            walker.perform_single_jump();
            double current_x = walker.get_x_position();
            
            // Update running sum for first moment calculation
            running_sum_x += current_x;
            
            // Check if it's time to take a measurement
            if (jump % measurement_interval == 0) {
                measurement_count++;
                
                // Calculate first moment M = <X> = (sum of all X positions so far) / (number of jumps so far)
                double first_moment_M = running_sum_x / jump;
                
                // Write to CSV
                file << jump << "," << first_moment_M << "," << current_x << "\n";
                
                // Print to console (only every 10th measurement to avoid spam)
                if (measurement_count % 10 == 0) {
                    std::cout << jump << "\t\t" << first_moment_M << "\t" << current_x << std::endl;
                }
                
                // Flush file every 100 measurements
                if (measurement_count % 100 == 0) {
                    file.flush();
                }
            }
        }
        
        // Final measurement
        double final_first_moment = running_sum_x / N;
        double final_x = walker.get_x_position();
        
        file << N << "," << final_first_moment << "," << final_x << "\n";
        file.close();
        
        std::cout << "\n=== FINAL RESULTS ===\n";
        std::cout << "Total jumps completed: " << N << std::endl;
        std::cout << "Final X position: " << final_x << std::endl;
        std::cout << "Final first moment M = <X> = " << final_first_moment << std::endl;
        std::cout << "Total measurements taken: " << (measurement_count + 1) << std::endl;
        std::cout << "Results saved to: " << filename << std::endl;
    }
    
    // Alternative version that runs multiple trajectories for better statistics
    static void analyze_ensemble_first_moment(double t, double alpha, double F, 
                                            int W, int H, int measurement_interval,
                                            int num_trajectories, const std::string& filename) {
        
        // Calculate total jumps N once
        LevyStableGenerator levy_gen;
        double S_alpha = levy_gen.compute_S_alpha(t, alpha);
        long long N = static_cast<long long>(std::round(S_alpha));
        
        std::cout << "=== ENSEMBLE FIRST MOMENT ANALYSIS ===\n";
        std::cout << "Number of trajectories: " << num_trajectories << std::endl;
        std::cout << "Total jumps per trajectory N = " << N << std::endl;
        std::cout << "Measurement interval: " << measurement_interval << " jumps\n\n";
        
        std::ofstream file(filename);
        file << "Jump_Number,Average_First_Moment,Std_Dev_First_Moment,Num_Trajectories\n";
        file << std::fixed << std::setprecision(8);
        
        // Calculate number of measurement points
        long long num_measurements = N / measurement_interval;
        
        for (long long measurement_point = 1; measurement_point <= num_measurements; measurement_point++) {
            long long current_jump = measurement_point * measurement_interval;
            
            std::vector<double> first_moments;
            
            // Run multiple trajectories up to current_jump
            for (int traj = 0; traj < num_trajectories; traj++) {
                RandomWalker3D walker(W, H, F);
                double sum_x = 0.0;
                
                // Perform jumps up to current_jump
                for (long long jump = 1; jump <= current_jump; jump++) {
                    walker.perform_single_jump();
                    sum_x += walker.get_x_position();
                }
                
                double first_moment = sum_x / current_jump;
                first_moments.push_back(first_moment);
            }
            
            // Calculate statistics
            double sum_moments = 0.0;
            for (double moment : first_moments) {
                sum_moments += moment;
            }
            double avg_first_moment = sum_moments / num_trajectories;
            
            double sum_sq_diff = 0.0;
            for (double moment : first_moments) {
                sum_sq_diff += (moment - avg_first_moment) * (moment - avg_first_moment);
            }
            double std_dev = std::sqrt(sum_sq_diff / num_trajectories);
            
            // Write to CSV
            file << current_jump << "," << avg_first_moment << "," << std_dev << "," << num_trajectories << "\n";
            
            if (measurement_point % 10 == 0) {
                std::cout << "Jump " << current_jump << " - Avg First Moment: " << avg_first_moment << std::endl;
            }
        }
        
        file.close();
        std::cout << "Ensemble analysis complete! Results saved to: " << filename << std::endl;
    }
};

int main() {
    // OPTIMAL PARAMETERS
    double alpha = 0.3;          // Stability parameter
    double F = 0.01;             // Force parameter
    double t = 1e14;             // Optimal laboratory time: 10^14
    
    int W = 100;                 // Y boundary size  
    int H = 100;                 // Z boundary size
    
    // USER CONFIGURABLE PARAMETERS
    int measurement_interval = 1000;  // Take measurement every X jumps
    
    std::cout << "Choose analysis mode:\n";
    std::cout << "1. Single trajectory analysis\n";
    std::cout << "2. Ensemble analysis (multiple trajectories)\n";
    std::cout << "Enter choice (1 or 2): ";
    
    int choice;
    std::cin >> choice;
    
    if (choice == 1) {
        std::cout << "Enter measurement interval (jumps between measurements): ";
        std::cin >> measurement_interval;
        
        FirstMomentAnalyzer::analyze_first_moment_vs_jumps(
            t, alpha, F, W, H, measurement_interval, 
            "first_moment_vs_jumps.csv"
        );
    }
    else if (choice == 2) {
        std::cout << "Enter measurement interval (jumps between measurements): ";
        std::cin >> measurement_interval;
        
        std::cout << "Enter number of trajectories for ensemble: ";
        int num_trajectories;
        std::cin >> num_trajectories;
        
        FirstMomentAnalyzer::analyze_ensemble_first_moment(
            t, alpha, F, W, H, measurement_interval, num_trajectories,
            "ensemble_first_moment_vs_jumps.csv"
        );
    }
    else {
        std::cout << "Invalid choice. Running single trajectory analysis.\n";
        FirstMomentAnalyzer::analyze_first_moment_vs_jumps(
            t, alpha, F, W, H, measurement_interval, 
            "first_moment_vs_jumps.csv"
        );
    }
    
    return 0;
}

// Compilation:
// g++ -o first_moment_sim first_moment_sim.cpp -std=c++11 -O3
// ./first_moment_sim