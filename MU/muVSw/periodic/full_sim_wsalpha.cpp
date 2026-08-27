// full_sim_wsalpha.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o full_sim_wsalpha full_sim_wsalpha.cpp
// Run:     ./full_sim_wsalpha
#include <bits/stdc++.h>
using namespace std;

// ---------------------- Config ----------------------
struct SimConfig {
    int N_traj = 1000;                // increase for production
    int measurement_points = 15;        // log-spaced t points
    double t_min = 1e16;
    double t_max = 1e17;
    double alpha = 0.3;                 // S_alpha
    double F_input = 0.01;              // nominal force
    vector<int> Ws = {1,3,5,7,9};           // widths to test
    bool use_effective_drift = true;    // use F_eff on x only
    double kappa_alpha = 10.0;           // κ_α amplitude factor (start with 1)
    double c_small_force = 0.02;        // enforce F <= c/w^2
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL; // reproducible base seed
    string csv = "results_wsalpha.csv";
};

// ---------------------- Math helpers ----------------------
inline double gamma1p(double a) { return std::tgamma(1.0 + a); } // Γ(1+α)

// K_lead(α) = 6^{-α}/(A Γ^2(1+α)); if A known, compute amplitude
inline double klead_from_A(double alpha, double A) {
    return std::pow(6.0, -alpha) / (A * std::pow(gamma1p(alpha), 2));
}

// Given plateau C_sim, estimate A: A_est = 6^{-α}/(C_sim Γ^2(1+α))
inline double A_est_from_plateau(double alpha, double Csim) {
    return std::pow(6.0, -alpha) / (Csim * std::pow(gamma1p(alpha), 2));
}

// 3D normalization with cosh form
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
struct ProbX { double p_plus_x, p_minus_x; };
struct ProbYZ { double p_plus_y, p_minus_y, p_plus_z, p_minus_z; };

ProbX compute_probabilities_x(double F_eff) {
    double Bx = norm3d_from_halfF(F_eff / 2.0);
    ProbX px;
    px.p_plus_x  = Bx * std::exp(F_eff / 2.0);
    px.p_minus_x = Bx * std::exp(-F_eff / 2.0);
    return px;
}
ProbYZ compute_probabilities_yz() {
    double By = norm3d_from_halfF(0.0);
    return ProbYZ{By, By, By, By};
}

// ---------------------- CMS one-sided S_alpha ----------------------
// Draw η from one-sided α-stable (CMS method variant) and map to steps s = ceil( (t/η)^α )
uint64_t generate_steps(double alpha, double t, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_real_distribution<double> Utheta(0.0, M_PI);
    double theta = Utheta(rng);
    double x = U(rng);
    double W = -std::log(x);
    // CMS constants for one-sided S_alpha:
    // η = (a(theta)/W)^{(1-α)/α}, where
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
    c.c6 = 1.0; // pyz.p_minus_z completes to 1 by construction
    return c;
}

int64_t simulate_trajectory(int w, const ProbX &px, const ProbYZ &pyz, double alpha, double t, std::mt19937_64 &rng, uint64_t &steps_out) {
    uint64_t steps = generate_steps(alpha, t, rng);
    steps_out = steps;
    int64_t x = 0;
    int y = 0, z = 0;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    CumProb cp = build_cum(px, pyz);
    // Safety: ensure probabilities sum to 1
    // (floating error tolerance)
    // assert(std::abs(cp.c6 - 1.0) < 1e-12);
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

// ---------------------- Main ----------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    SimConfig cfg;
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;

    // log-spaced times
    vector<double> times(cfg.measurement_points);
    double lgmin = std::log10(cfg.t_min), lgmax = std::log10(cfg.t_max);
    double d = (lgmax - lgmin) / (cfg.measurement_points - 1);
    for (int i = 0; i < cfg.measurement_points; ++i) times[i] = std::pow(10.0, lgmin + d * i);

    ofstream out(cfg.csv);
    out << "w,t,F_used,average_x,C_sim,steps_max,steps_mean\n";

    cout << "Threads: " << n_threads << ", N_traj: " << cfg.N_traj << "\n";
    cout << "Alpha: " << cfg.alpha << ", F_input: " << cfg.F_input
         << " (effective drift " << (cfg.use_effective_drift ? "ON" : "OFF") << ")\n";

    // Precompute transverse probabilities (unbiased)
    ProbYZ pyz = compute_probabilities_yz();

    for (int w : cfg.Ws) {
        cout << "\nWidth w=" << w << "\n";
        // enforce small-force guard
        double F_use_nom = clamp_force_for_width(cfg.F_input, w, cfg.c_small_force);
        for (int m = 0; m < cfg.measurement_points; ++m) {
            double t = times[m];

            // Effective drift for x
            double F_use = F_use_nom;
            double F_eff = cfg.use_effective_drift
                         ? cfg.kappa_alpha * std::pow(F_use, cfg.alpha) * std::pow((double)w, -2.0 * (1.0 - cfg.alpha))
                         : F_use;

            ProbX px = compute_probabilities_x(F_eff);

            // Launch threads
            int per = cfg.N_traj / (int)n_threads;
            vector<thread> pool;
            vector<Accum> acc(n_threads);
            uint64_t seed = cfg.base_seed ^ (uint64_t)w ^ (uint64_t)m;
            for (unsigned int th = 0; th < n_threads; ++th) {
                uint64_t s = seed; for (unsigned int k = 0; k <= th; ++k) s = splitmix64(s);
                int i0 = th * per;
                int i1 = (th == n_threads - 1) ? cfg.N_traj : i0 + per;
                pool.emplace_back(worker_block, w, cref(px), cref(pyz), cfg.alpha, t, s, i0, i1, ref(acc[th]));
            }
            for (auto &th : pool) th.join();

            // Reduce
            double x_sum = 0.0;
            uint64_t steps_max = 0ULL;
            unsigned __int128 steps_sum128 = 0; // to avoid overflow
            for (auto &a : acc) {
                x_sum += a.x_sum;
                steps_max = max(steps_max, a.steps_max);
                steps_sum128 += a.steps_sum;
            }
            double avg_steps = (double)steps_sum128 / (double)cfg.N_traj;
            double avg_x = x_sum / (double)cfg.N_traj;

            // Plateau observable: C_sim = <x> t^{-α} F^{-α} w^{2(1-α)}
            // If effective drift is ON, F^{-α} uses the physical F_use, not F_eff,
            // because the theoretical law is in terms of the physical F.
            double C_sim = avg_x * std::pow(t, -cfg.alpha)
                                 * std::pow(F_use, -cfg.alpha)
                                 * std::pow((double)w, 2.0 * (1.0 - cfg.alpha));

            cout << "t=" << std::scientific << t
                 << "  <x>=" << avg_x
                 << "  C_sim=" << C_sim
                 << "  steps_max=" << steps_max
                 << "  steps_mean=" << avg_steps << "\n";

            out << w << "," << std::setprecision(17) << t << "," << F_use << ","
                << std::setprecision(17) << avg_x << ","
                << std::setprecision(17) << C_sim << ","
                << steps_max << "," << std::setprecision(17) << avg_steps << "\n";
        }
    }
    out.close();

    cerr << "\nPost-processing tip: At the largest t for each w, average C_sim over small F values (if swept) to estimate A via A_est = 6^{-alpha}/(C_sim * Gamma(1+alpha)^2).\n";
    return 0;
}
