#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <fstream>

// Global random number generator
std::random_device rd;
std::mt19937 gen(rd());

// --- Utility Functions ---

// Transform a uniform variable into a power-law waiting time.
double func_transform(double x) {
    // Avoid division by zero by ensuring x > 0.
    return std::pow(x, -2.0);
}

// Generate a single waiting time from uniform distribution in (epsilon, 1]
double gen_wait_time(double a = 1e-6, double b = 1.0) {
    std::uniform_real_distribution<> dis(a, b);
    double x = dis(gen);
    return func_transform(x);
}

// Generate a shuffled vector of waiting times, with the middle set to zero.
std::vector<double> generate_waiting_times(int size) {
    std::vector<double> waiting_times;
    waiting_times.reserve(size);
    for (int i = 0; i < size; ++i) {
        waiting_times.push_back(gen_wait_time());
    }
    // Shuffle the waiting times.
    std::shuffle(waiting_times.begin(), waiting_times.end(), gen);
    int mid_index = size / 2;
    waiting_times[mid_index] = 0.0; // Immediate jump at the middle.
    return waiting_times;
}

// Sample a jump (dx, dy) from a discrete distribution.
std::pair<int, int> sample_jump() {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double r = dis(gen);
    if (r < 3.0/8.0)
        return {1, 0};      // RIGHT
    else if (r < 4.0/8.0)
        return {-1, 0};     // LEFT
    else if (r < 6.0/8.0)
        return {0, 1};      // UP
    else
        return {0, -1};     // DOWN
}

// --- Simulation Functions ---

// Simulate a 2D random walk with quenched waiting times and periodic boundaries in y.
// Returns a pair: vector of positions (as pairs of ints) and vector of simulation times.
std::pair<std::vector<std::pair<int,int>>, std::vector<double>>
RW_sim_2d_fixed_wait(int sim_time, int wait_list_size, int Y_min, int Y_max) {
    // Generate waiting time lists for x and y.
    std::vector<double> wait_x = generate_waiting_times(wait_list_size);
    std::vector<double> wait_y = generate_waiting_times(wait_list_size);
    
    int x = 0, y = 0;
    std::vector<std::pair<int,int>> positions;
    std::vector<double> times;
    positions.push_back({x, y});
    times.push_back(0.0);
    
    double current_time = 0.0;
    int x_index = wait_list_size / 2;
    int y_index = wait_list_size / 2;
    
    while (current_time < sim_time) {
        auto jump = sample_jump();
        int dx = jump.first;
        int dy = jump.second;
        x += dx;
        int new_y = y + dy;
        // Periodic boundary conditions in y:
        if (new_y > Y_max)
            y = Y_min;
        else if (new_y < Y_min)
            y = Y_max;
        else
            y = new_y;
        
        // Update waiting time indices (clamping to valid range)
        if (dx != 0) {
            x_index = std::max(0, std::min(wait_list_size - 1, x_index + dx));
        }
        if (dy != 0) {
            y_index = std::max(0, std::min(wait_list_size - 1, y_index + dy));
        }
        
        // Select waiting time.
        double waiting_time = (x == 0 && y == 0) ? 0.0 : std::max(wait_x[x_index], wait_y[y_index]);
        current_time += waiting_time;
        positions.push_back({x, y});
        times.push_back(current_time);
    }
    
    return {positions, times};
}

// Write 3D random walk data (x, y, time) to a CSV file for external plotting.
void save_3d_walk_to_csv(const std::vector<std::pair<int,int>> &positions,
                           const std::vector<double> &times,
                           const std::string &filename = "3d_walk.csv") {
    std::ofstream ofs(filename);
    ofs << "x,y,time\n";
    for (size_t i = 0; i < positions.size(); ++i) {
        ofs << positions[i].first << "," << positions[i].second << "," << times[i] << "\n";
    }
    ofs.close();
    std::cout << "3D walk data saved to " << filename << std::endl;
}

// Compute the first moment (mean x, mean y) for a vector of positions.
std::pair<double,double> calculate_first_moment(const std::vector<std::pair<int,int>> &positions) {
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto &p : positions) {
        sum_x += p.first;
        sum_y += p.second;
    }
    double n = static_cast<double>(positions.size());
    return {sum_x / n, sum_y / n};
}

// Run multiple simulations to compute first moments; returns vectors of ⟨Jx⟩ and ⟨Jy⟩.
void multi_RW_first_moment_fixed_wait(int num_sims, int sim_time, int wait_list_size,
                                        int Y_min, int Y_max,
                                        std::vector<double> &first_moments_x,
                                        std::vector<double> &first_moments_y) {
    first_moments_x.clear();
    first_moments_y.clear();
    for (int i = 0; i < num_sims; ++i) {
        auto result = RW_sim_2d_fixed_wait(sim_time, wait_list_size, Y_min, Y_max);
        auto moment = calculate_first_moment(result.first);
        first_moments_x.push_back(moment.first);
        first_moments_y.push_back(moment.second);
    }
}

// Estimate total runtime given a single run time and total iterations.
void estimate_runtime(double single_run_time, int total_iterations) {
    double estimated_time = single_run_time * total_iterations;
    std::cout << "Estimated total runtime: " << estimated_time << " seconds ("
              << estimated_time / 60.0 << " minutes)" << std::endl;
}

// Compute an “A value” for a given system width W by averaging first moments over multiple tests.
// (Note: The original Python code fit a power law to the time evolution. Here we simply average the computed values.)
double compute_A_for_W(int W, int num_tests, int num_sims,
                         int sim_time_start, int sim_time_finish, int time_step,
                         int wait_list_size) {
    int Y_min = -W / 2;
    int Y_max = W / 2;
    std::vector<double> A_values;
    
    for (int test = 0; test < num_tests; ++test) {
        std::vector<double> mean_first_moments_x;
        for (int sim_time = sim_time_start; sim_time <= sim_time_finish; sim_time += time_step) {
            std::vector<double> test_first_moments;
            for (int i = 0; i < num_sims; ++i) {
                auto result = RW_sim_2d_fixed_wait(sim_time, wait_list_size, Y_min, Y_max);
                auto moment = calculate_first_moment(result.first);
                test_first_moments.push_back(moment.first);
            }
            // Compute mean over num_sims
            double sum = 0.0;
            for (double v : test_first_moments)
                sum += v;
            mean_first_moments_x.push_back(sum / test_first_moments.size());
        }
        // Average over the time series for this test.
        double sum = 0.0;
        for (double v : mean_first_moments_x)
            sum += v;
        A_values.push_back(sum / mean_first_moments_x.size());
    }
    
    // Return average A over tests.
    double sum = 0.0;
    for (double v : A_values)
        sum += v;
    return sum / A_values.size();
}

// Helper: run compute_A_for_W in parallel for different W values.
// (This uses a simple thread-based parallelism.)
std::vector<double> coefficient_vs_width_new(int num_tests, int W_initial, int W_final, int W_step,
                                               int num_sims, int sim_time_start, int sim_time_finish,
                                               int time_step, int wait_list_size) {
    std::vector<int> W_values;
    for (int W = W_initial; W <= W_final; W += W_step)
        W_values.push_back(W);
    std::vector<double> mean_A_values(W_values.size(), 0.0);
    
    std::vector<std::thread> threads;
    std::mutex mtx;
    
    auto worker = [&](int idx, int W) {
        double A_val = compute_A_for_W(W, num_tests, num_sims, sim_time_start, sim_time_finish, time_step, wait_list_size);
        std::lock_guard<std::mutex> lock(mtx);
        mean_A_values[idx] = A_val;
    };
    
    for (size_t i = 0; i < W_values.size(); ++i) {
        threads.emplace_back(worker, i, W_values[i]);
    }
    for (auto &t : threads) {
        t.join();
    }
    
    // Optionally, output the (W, A) values to a CSV for external plotting.
    std::ofstream ofs("coefficient_vs_width.csv");
    ofs << "W,Mean_A\n";
    for (size_t i = 0; i < W_values.size(); ++i) {
        ofs << W_values[i] << "," << mean_A_values[i] << "\n";
    }
    ofs.close();
    std::cout << "Coefficient vs. width data saved to coefficient_vs_width.csv" << std::endl;
    
    return mean_A_values;
}

// --- Menu and Main Function ---

void print_menu() {
    std::cout << "\nMenu:\n";
    std::cout << "1. View Single 2D Random Walk (save data to CSV)\n";
    std::cout << "2. View Histogram Data of Final Positions (printed summary)\n";
    std::cout << "3. View First Moment with Noise (simulate and print sample stats)\n";
    std::cout << "4. View First Moment and Save Data for Power-Law Analysis\n";
    std::cout << "5. Test Relationship Between Coefficient A and Width W (save CSV)\n";
    std::cout << "9. Exit\n";
    std::cout << "Enter your choice: ";
}

void view_single_random_walk() {
    int sim_time = 500;
    int wait_list_size = 250;
    int Y_min = -5, Y_max = 5;
    auto result = RW_sim_2d_fixed_wait(sim_time, wait_list_size, Y_min, Y_max);
    // Save data to CSV so it can be plotted externally.
    save_3d_walk_to_csv(result.first, result.second, "3d_walk.csv");
}

void view_hist_final_positions() {
    int num_sims = 500000;
    int sim_time = 10000;
    int wait_list_size = 250;
    int Y_min = -100, Y_max = 100;
    std::vector<std::pair<int,int>> final_positions;
    for (int i = 0; i < num_sims; ++i) {
        auto result = RW_sim_2d_fixed_wait(sim_time, wait_list_size, Y_min, Y_max);
        final_positions.push_back(result.first.back());
    }
    // Compute simple histogram statistics for x and y.
    double sum_x = 0, sum_y = 0;
    for (auto &p : final_positions) {
        sum_x += p.first;
        sum_y += p.second;
    }
    std::cout << "Average final X: " << sum_x / final_positions.size() << std::endl;
    std::cout << "Average final Y: " << sum_y / final_positions.size() << std::endl;
    std::cout << "Histogram data can be further processed or exported as needed." << std::endl;
}

void view_first_moment_with_noise() {
    int num_sims = 5000;
    int sim_time_start = 0;
    int sim_time_finish = 1000;
    int time_step = 50;
    int wait_list_size = 250;
    int Y_min = -100, Y_max = 100;
    std::vector<double> all_times;
    std::vector<double> all_first_moments_x;
    std::vector<double> all_first_moments_y;
    
    for (int sim_time = sim_time_start; sim_time <= sim_time_finish; sim_time += time_step) {
        std::vector<double> first_moments_x, first_moments_y;
        multi_RW_first_moment_fixed_wait(num_sims, sim_time, wait_list_size, Y_min, Y_max,
                                          first_moments_x, first_moments_y);
        for (size_t i = 0; i < first_moments_x.size(); ++i) {
            all_times.push_back(sim_time);
            all_first_moments_x.push_back(first_moments_x[i]);
            all_first_moments_y.push_back(first_moments_y[i]);
        }
    }
    // For demonstration, print out the first few values.
    std::cout << "Sample data for first moment (Jx):\n";
    for (size_t i = 0; i < std::min(all_first_moments_x.size(), size_t(10)); ++i)
        std::cout << "Time: " << all_times[i] << ", Jx: " << all_first_moments_x[i] << "\n";
    std::cout << "Data for further analysis can be saved to a file if needed.\n";
}

void view_first_moment_powerlaw_data() {
    int num_sims = 25000;
    int sim_time_start = 0;
    int sim_time_finish = 1000;
    int time_step = 50;
    int wait_list_size = 100;
    int Y_min = -5, Y_max = 5;
    std::vector<double> time_values;
    std::vector<double> mean_first_moments_x;
    
    for (int sim_time = sim_time_start; sim_time <= sim_time_finish; sim_time += time_step) {
        std::vector<double> first_moments;
        for (int i = 0; i < num_sims; ++i) {
            auto result = RW_sim_2d_fixed_wait(sim_time, wait_list_size, Y_min, Y_max);
            auto moment = calculate_first_moment(result.first);
            first_moments.push_back(moment.first);
        }
        double sum = 0.0;
        for (double v : first_moments)
            sum += v;
        time_values.push_back(sim_time);
        mean_first_moments_x.push_back(sum / first_moments.size());
    }
    
    // Save the data to a CSV file.
    std::ofstream ofs("first_moment_powerlaw.csv");
    ofs << "Time,Mean_Jx\n";
    for (size_t i = 0; i < time_values.size(); ++i)
        ofs << time_values[i] << "," << mean_first_moments_x[i] << "\n";
    ofs.close();
    std::cout << "First moment vs. time data saved to first_moment_powerlaw.csv" << std::endl;
}

void test_coefficient_vs_width_new() {
    int num_tests = 10;
    int W_initial = 0;
    int W_final = 100;
    int W_step = 25;
    int num_sims = 1000;
    int sim_time_start = 0;
    int sim_time_finish = 1000;
    int time_step = 250;
    int wait_list_size = 50;
    
    // First, estimate runtime for one run.
    auto start = std::chrono::steady_clock::now();
    compute_A_for_W(W_initial, num_tests, num_sims, sim_time_start, sim_time_finish, time_step, wait_list_size);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    estimate_runtime(elapsed.count(), ((W_final - W_initial) / W_step) + 1);
    
    // Run the coefficient vs. width tests in parallel.
    coefficient_vs_width_new(num_tests, W_initial, W_final, W_step, num_sims,
                             sim_time_start, sim_time_finish, time_step, wait_list_size);
}

// --- Main Program ---

int main() {
    while (true) {
        print_menu();
        int choice;
        std::cin >> choice;
        if (choice == 1) {
            view_single_random_walk();
        } else if (choice == 2) {
            view_hist_final_positions();
        } else if (choice == 3) {
            view_first_moment_with_noise();
        } else if (choice == 4) {
            view_first_moment_powerlaw_data();
        } else if (choice == 5) {
            test_coefficient_vs_width_new();
        } else if (choice == 9) {
            break;
        } else {
            std::cout << "Invalid choice. Please try again.\n";
        }
    }
    return 0;
}
