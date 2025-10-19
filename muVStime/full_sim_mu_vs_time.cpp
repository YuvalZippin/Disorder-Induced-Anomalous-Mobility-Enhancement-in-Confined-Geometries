// sim_mu_vs_time_leading.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o sim_mu_vs_time_leading sim_mu_vs_time_leading.cpp
// Run:     ./sim_mu_vs_time_leading

#include <bits/stdc++.h>
using namespace std;

// ---------------------- Config ----------------------
struct SimConfig {
    int N_traj = 10000;
    int measurement_points = 5;
    double t_min = 1e05;
    double t_max = 1e25;
    double alpha = 0.3;
    double F_input = 0.01;
    int w = 5;
    double c_small_force = 0.02;
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;
    string csv = "results_mu_vs_time_leading.csv";
    double A = 1.0; // Set to 1.0 unless you want to use a different amplitude
};

// ---------------------- Math helpers ----------------------
inline double gamma1p(double a) { return std::tgamma(1.0 + a); }

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

inline double norm3d_from_halfF(double Fhalf) { return 1.0 / (2.0 * std::cosh(Fhalf) + 4.0); }

ProbX compute_probabilities_x(double F) {
    double Bx = norm3d_from_halfF(F / 2.0);
    ProbX px;
    px.p_plus_x  = Bx * std::exp(F / 2.0);
    px.p_minus_x = Bx * std::exp(-F / 2.0);
    return px;
}
ProbYZ compute_probabilities_yz() {
    double By = norm3d_from_halfF(0.0);
    return ProbYZ{By, By, By, By};
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

int64_t simulate_trajectory(int w, const ProbX &px, const ProbYZ &pyz, double alpha, double t, std::mt19937_64 &rng, uint64_t &steps_out) {
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
    out << "t,average_x,mu_num,mu_theory,steps_max,steps_mean\n";

    cout << "Threads: " << n_threads << ", N_traj: " << cfg.N_traj << "\n";
    cout << "Alpha: " << cfg.alpha << ", F_input: " << cfg.F_input << "\n";
    cout << "Width w=" << cfg.w << "\n";

    // Precompute probabilities
    ProbYZ pyz = compute_probabilities_yz();
    double F_use = clamp_force_for_width(cfg.F_input, cfg.w, cfg.c_small_force);
    ProbX px = compute_probabilities_x(F_use);

    // Theoretical mu
    double gamma1p_alpha = gamma1p(cfg.alpha);
    double mu_theory = std::pow(6.0, -cfg.alpha) / (cfg.A * gamma1p_alpha * gamma1p_alpha) * std::pow(cfg.w, -2.0 * (1.0 - cfg.alpha));

    for (int m = 0; m < cfg.measurement_points; ++m) {
        double t = times[m];

        // Launch threads
        int per = cfg.N_traj / (int)n_threads;
        vector<thread> pool;
        vector<Accum> acc(n_threads);
        uint64_t seed = cfg.base_seed ^ (uint64_t)m;
        for (unsigned int th = 0; th < n_threads; ++th) {
            uint64_t s = seed; for (unsigned int k = 0; k <= th; ++k) s = splitmix64(s);
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

        // Numerically extract mu from ⟨x⟩ = mu * F^alpha * t^alpha
        double mu_num = avg_x / (std::pow(F_use, cfg.alpha) * std::pow(t, cfg.alpha));

        cout << "t=" << std::scientific << t
             << "  <x>=" << avg_x
             << "  mu_num=" << mu_num
             << "  mu_theory=" << mu_theory
             << "  steps_max=" << steps_max
             << "  steps_mean=" << avg_steps << "\n";

        out << std::setprecision(17) << t << ","
            << std::setprecision(17) << avg_x << ","
            << std::setprecision(17) << mu_num << ","
            << std::setprecision(17) << mu_theory << ","
            << steps_max << ","
            << std::setprecision(17) << avg_steps << "\n";
    }
    out.close();

    cerr << "\nPost-processing: Compare mu_num to mu_theory for scaling check.\n";
    return 0;
}
