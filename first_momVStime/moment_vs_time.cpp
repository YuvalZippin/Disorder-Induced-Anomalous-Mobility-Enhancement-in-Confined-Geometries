// moment_vs_time.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o moment_vs_time moment_vs_time.cpp
// Run:     ./moment_vs_time
//
// This simulation:
// - 3D simple cubic lattice with periodic BCs in y,z of width w, bias along +x
// - One-sided S_alpha CMS step-generation
// - Proper 3D normalization B(F) = 1 / (2 cosh(F/2) + 4)
// - Outputs both leading-order theory and two-term small-F correction
// - Computes A_est(t) to calibrate amplitude from data
// - Detects convergence for both pre-asymptotic and asymptotic regimes
//
// CSV columns:
//   w,t,F_used,average_x,x_leading,x_two_term,mu_num,mu_theory,A_est,ratio,rel_err,slope_local,steps_max,steps_mean
//
// Reference: User-provided 3D channel derivation (small-F leading behavior)
// ------------------------------------------------------------------------------

#include <bits/stdc++.h>
using namespace std;

// ---------------------- Config ----------------------
struct SimConfig {
    // Simulation controls
    int    N_traj = 10000;           // increase for better statistics
    int    measurement_points = 5;  // log-spaced times
    double t_min = 1e5;             // push lower for transient
    double t_max = 1e20;             // push higher for asymptotic

    // Physics
    double alpha = 0.3;              // 0<alpha<1
    double F_input = 0.001;          // SMALL force; must satisfy F << 2/w^2
    int    w = 5;                    // channel width (periodic in y,z)
    double A = 1.0;                  // trap amplitude factor (will be calibrated)

    // Small-force guard
    double c_small_force = 0.02;     // enforce F <= c/w^2

    // Reproducibility
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;

    // Output
    string csv = "results_moment_vs_time.csv";

    // Convergence criteria
    double conv_tol = 0.10;          // |ratio-1| <= 10%
    double slope_tol = 0.05;         // |slope_local - alpha| <= 0.05
    int    conv_window = 3;          // consecutive points required
};

// ---------------------- Math helpers ----------------------
inline double gamma1p(double a) { return std::tgamma(1.0 + a); }

// 3D normalization B(F)
inline double norm3d_from_halfF(double Fhalf) { return 1.0 / (2.0 * std::cosh(Fhalf) + 4.0); }

// Small-force guard per width
inline double clamp_force_for_width(double F_in, int w, double c_small_force) {
    double bound = c_small_force / (double(w) * double(w));
    return std::min(F_in, bound);
}

// SplitMix64 for per-thread seeding
static inline uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ---------------------- Probabilities ----------------------
struct ProbX  { double p_plus_x, p_minus_x; };
struct ProbYZ { double p_plus_y, p_minus_y, p_plus_z, p_minus_z; };

ProbX compute_probabilities_x(double F) {
    double B = norm3d_from_halfF(F / 2.0);
    ProbX px;
    px.p_plus_x  = B * std::exp(F / 2.0);
    px.p_minus_x = B * std::exp(-F / 2.0);
    return px;
}

ProbYZ compute_probabilities_yz(double F) {
    double B = norm3d_from_halfF(F / 2.0);
    return ProbYZ{B, B, B, B};
}

// ---------------------- CMS one-sided S_alpha ----------------------
uint64_t generate_steps(double alpha, double t, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_real_distribution<double> Utheta(0.0, M_PI);
    double theta = Utheta(rng);
    double x = U(rng);
    double W = -std::log(x);
    double a_num = std::sin((1.0 - alpha) * theta) * std::pow(std::sin(alpha * theta), alpha / (1.0 - alpha));
    double a_den = std::pow(std::sin(theta), 1.0 / (1.0 - alpha));
    double a_theta = a_num / a_den;
    double eta = std::pow(a_theta / W, (1.0 - alpha) / alpha);
    uint64_t s = (uint64_t)std::ceil(std::pow(t / eta, alpha));
    if (s < 1ULL) s = 1ULL;
    return s;
}

// ---------------------- Trajectory simulation ----------------------
struct CumProb { double c1, c2, c3, c4, c5, c6; };

inline CumProb build_cum(const ProbX& px, const ProbYZ& pyz) {
    CumProb c;
    c.c1 = px.p_plus_x;
    c.c2 = c.c1 + px.p_minus_x;
    c.c3 = c.c2 + pyz.p_plus_y;
    c.c4 = c.c3 + pyz.p_minus_y;
    c.c5 = c.c4 + pyz.p_plus_z;
    c.c6 = 1.0;
    return c;
}

int64_t simulate_trajectory(int w, const ProbX &px, const ProbYZ &pyz, double alpha, double t,
                            std::mt19937_64 &rng, uint64_t &steps_out) {
    uint64_t steps = generate_steps(alpha, t, rng);
    steps_out = steps;
    int64_t x = 0;
    int y = 0, z = 0;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    CumProb cp = build_cum(px, pyz);
    int64_t x_sum = 0;

    for (uint64_t s = 0; s < steps; ++s) {
        double r = U(rng);
        if (r < cp.c1) {
            ++x; ++x_sum;
        } else if (r < cp.c2) {
            --x; --x_sum;
        } else if (r < cp.c3) {
            y = (y + 1) % w;
        } else if (r < cp.c4) {
            y = (y - 1 + w) % w;
        } else if (r < cp.c5) {
            z = (z + 1) % w;
        } else {
            z = (z - 1 + w) % w;
        }
    }
    return x_sum;
}

// ---------------------- Worker ----------------------
struct Accum {
    double x_sum = 0.0;
    uint64_t steps_max = 0ULL;
    uint64_t steps_sum = 0ULL;
};

void worker_block(int w, const ProbX &px, const ProbYZ &pyz, double alpha, double t,
                  uint64_t seed, int i0, int i1, Accum &out) {
    std::mt19937_64 rng(seed);
    double local_x = 0.0;
    uint64_t local_steps_max = 0ULL;
    uint64_t local_steps_sum = 0ULL;

    for (int i = i0; i < i1; ++i) {
        uint64_t steps_out = 0ULL;
        int64_t x_disp = simulate_trajectory(w, px, pyz, alpha, t, rng, steps_out);
        local_x += (double)x_disp;
        local_steps_sum += steps_out;
        if (steps_out > local_steps_max) local_steps_max = steps_out;
    }
    out.x_sum = local_x;
    out.steps_max = local_steps_max;
    out.steps_sum = local_steps_sum;
}

// ---------------------- Theory functions ----------------------
inline double mu_theory_3D(double alpha, int w, double A) {
    // μ_th = 6^{-α} / (A Γ^2(1+α)) · w^{-2(1−α)}
    double g = gamma1p(alpha);
    return std::pow(6.0, -alpha) / (A * g * g) * std::pow((double)w, -2.0 * (1.0 - alpha));
}

inline double x_leading(double t, double F, double alpha, int w, double A) {
    // <x>_leading = μ_th · F^α · t^α
    return mu_theory_3D(alpha, w, A) * std::pow(F, alpha) * std::pow(t, alpha);
}

inline double x_two_term(double t, double F, double alpha, int w, double A) {
    // Two-term small-F expansion:
    // <x> ≈ (F/6)/(A Γ^2(1+α)) · [ (w²F/6)^(α-1) - (w²F/6)^α ] · t^α
    double g = gamma1p(alpha);
    double prefac = (F / 6.0) / (A * g * g);
    double eps = (w * w * F) / 6.0;
    double bracket = std::pow(eps, alpha - 1.0) - std::pow(eps, alpha);
    return prefac * bracket * std::pow(t, alpha);
}

inline double A_est_from_mu(double mu_num, double alpha, int w) {
    // A_est = 6^{-α} / (mu_num Γ^2(1+α)) · w^{-2(1−α)}
    double g = gamma1p(alpha);
    return std::pow(6.0, -alpha) / (mu_num * g * g) * std::pow((double)w, -2.0 * (1.0 - alpha));
}

// ---------------------- Main ----------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    SimConfig cfg;

    // Threads
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;

    // Log-spaced times
    vector<double> times(cfg.measurement_points);
    double lgmin = std::log10(cfg.t_min), lgmax = std::log10(cfg.t_max);
    double d = (lgmax - lgmin) / (cfg.measurement_points - 1);
    for (int i = 0; i < cfg.measurement_points; ++i) {
        times[i] = std::pow(10.0, lgmin + d * i);
    }

    // Small-force guard
    double F_use = clamp_force_for_width(cfg.F_input, cfg.w, cfg.c_small_force);

    // Probabilities
    ProbX  px  = compute_probabilities_x(F_use);
    ProbYZ pyz = compute_probabilities_yz(F_use);

    // Theoretical μ_th
    double mu_th = mu_theory_3D(cfg.alpha, cfg.w, cfg.A);

    // Output CSV
    ofstream out(cfg.csv);
    out << "w,t,F_used,average_x,x_leading,x_two_term,mu_num,mu_theory,A_est,ratio,rel_err,slope_local,steps_max,steps_mean\n";

    cout << "========================================\n";
    cout << "3D Channel Random Walk: First Moment vs Time\n";
    cout << "========================================\n";
    cout << "Threads: " << n_threads << ", N_traj: " << cfg.N_traj << "\n";
    cout << "Alpha: " << cfg.alpha << ", w: " << cfg.w << "\n";
    cout << "F_input: " << cfg.F_input << " -> F_used: " << F_use << " (guard F <= c/w^2)\n";
    cout << "Theory mu_th = " << std::setprecision(10) << mu_th << "\n";
    cout << "Small-F condition: F << 2/w^2 = " << 2.0 / (cfg.w * cfg.w) << "\n";
    cout << "========================================\n\n";

    // Storage for diagnostics
    vector<double> t_vec, x_vec, mu_vec, ratio_vec, A_est_vec;

    for (int m = 0; m < cfg.measurement_points; ++m) {
        double t = times[m];

        // Launch threads
        int per = cfg.N_traj / (int)n_threads;
        vector<thread> pool;
        vector<Accum> acc(n_threads);
        uint64_t seed = cfg.base_seed ^ (uint64_t)m;

        for (unsigned int th = 0; th < n_threads; ++th) {
            uint64_t s = seed; 
            for (unsigned int k = 0; k <= th; ++k) s = splitmix64(s);
            int i0 = th * per;
            int i1 = (th == n_threads - 1) ? cfg.N_traj : i0 + per;
            pool.emplace_back(worker_block, cfg.w, cref(px), cref(pyz), cfg.alpha, t, s, i0, i1, ref(acc[th]));
        }
        for (auto &th : pool) th.join();

        // Reduce
        double x_sum = 0.0;
        uint64_t steps_max = 0ULL;
        unsigned __int128 steps_sum128 = 0;

        for (auto &a : acc) {
            x_sum += a.x_sum;
            steps_max = max(steps_max, a.steps_max);
            steps_sum128 += a.steps_sum;
        }
        double avg_steps = (double)steps_sum128 / (double)cfg.N_traj;
        double avg_x = x_sum / (double)cfg.N_traj;

        // Theoretical curves
        double x_lead = x_leading(t, F_use, cfg.alpha, cfg.w, cfg.A);
        double x_two = x_two_term(t, F_use, cfg.alpha, cfg.w, cfg.A);

        // Extract μ_num and A_est
        double mu_num = avg_x / (std::pow(F_use, cfg.alpha) * std::pow(t, cfg.alpha));
        double A_est = A_est_from_mu(mu_num, cfg.alpha, cfg.w);
        double ratio = mu_num / mu_th;
        double rel_err = std::abs(mu_num - mu_th) / std::max(mu_th, 1e-300);

        // Local slope (backward diff)
        double slope_local = NAN;
        if (!t_vec.empty()) {
            double logt2 = std::log(t);
            double logt1 = std::log(t_vec.back());
            double logx2 = std::log(std::max(avg_x, 1e-300));
            double logx1 = std::log(std::max(x_vec.back(), 1e-300));
            slope_local = (logx2 - logx1) / (logt2 - logt1);
        }

        // Save series
        t_vec.push_back(t);
        x_vec.push_back(avg_x);
        mu_vec.push_back(mu_num);
        ratio_vec.push_back(ratio);
        A_est_vec.push_back(A_est);

        // Log
        cout << "t=" << std::scientific << std::setprecision(4) << t
             << "  <x>=" << avg_x
             << "  x_lead=" << x_lead
             << "  x_two=" << x_two
             << "  mu_num=" << mu_num
             << "  A_est=" << A_est
             << "  ratio=" << ratio
             << "  slope=" << slope_local << "\n";

        out << cfg.w << ","
            << std::setprecision(17) << t << ","
            << F_use << ","
            << avg_x << ","
            << x_lead << ","
            << x_two << ","
            << mu_num << ","
            << mu_th << ","
            << A_est << ","
            << ratio << ","
            << rel_err << ","
            << slope_local << ","
            << steps_max << ","
            << avg_steps << "\n";
    }
    out.close();

    cout << "\n========================================\n";
    cout << "Convergence Analysis\n";
    cout << "========================================\n";

    // Detect convergence (ratio within tolerance for conv_window consecutive points)
    int win = std::max(1, cfg.conv_window);
    int conv_idx = -1;
    for (int i = win - 1; i < (int)t_vec.size(); ++i) {
        bool ok = true;
        for (int k = 0; k < win; ++k) {
            if (std::abs(ratio_vec[i - k] - 1.0) > cfg.conv_tol) { ok = false; break; }
        }
        if (!ok) continue;

        // Check slope at i
        double slope_local = NAN;
        if (i > 0) {
            double logt2 = std::log(t_vec[i]);
            double logt1 = std::log(t_vec[i-1]);
            double logx2 = std::log(std::max(x_vec[i], 1e-300));
            double logx1 = std::log(std::max(x_vec[i-1], 1e-300));
            slope_local = (logx2 - logx1) / (logt2 - logt1);
        }

        if (std::isfinite(slope_local) && std::abs(slope_local - cfg.alpha) <= cfg.slope_tol) {
            conv_idx = i;
            break;
        }
    }

    if (conv_idx >= 0) {
        cout << "✓ Convergence detected at t ≈ " << std::scientific << t_vec[conv_idx] << "\n";
        cout << "  ratio = " << ratio_vec[conv_idx] << ", A_est = " << A_est_vec[conv_idx] << "\n";
    } else {
        cout << "✗ Convergence NOT reached within scanned times.\n";
        cout << "  Suggestions:\n";
        cout << "    - Reduce F further (currently F=" << F_use << ", need F << " << 2.0/(cfg.w*cfg.w) << ")\n";
        cout << "    - Increase t_max or N_traj\n";
        cout << "    - Check A_est at largest t for amplitude calibration\n";
    }

    // Report A_est at largest t
    if (!A_est_vec.empty()) {
        double A_est_tail = A_est_vec.back();
        cout << "\nA_est at largest t = " << A_est_tail << " (input A = " << cfg.A << ")\n";
        if (std::abs(A_est_tail - cfg.A) / cfg.A > 0.2) {
            cout << "  ⚠ Consider using A ≈ " << A_est_tail << " for better amplitude alignment.\n";
        }
    }

    cout << "========================================\n";
    cout << "CSV saved: " << cfg.csv << "\n";
    
    return 0;
}
