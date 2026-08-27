// 2d_angled_sim_wsalpha_2D.cpp
// Usage: Compile with g++ -std=c++17 -O3 -pthread -o 2d_angled_sim_wsalpha_2D 2d_angled_sim_wsalpha_2D.cpp
//        Run: ./2d_angled_sim_wsalpha_2D
#include <bits/stdc++.h>
using namespace std;

struct SimConfig {
    int N_traj = 10000;
    int measurement_points = 15;
    double t_min = 1e16, t_max = 1e17;
    double alpha = 0.3;
    double F_input = 0.01;
    double theta_deg = 45.0;
    vector<int> Ws = {1,3,5,7,9};
    bool use_effective_drift = true;
    double kappa_alpha = 10.0;
    double c_small_force = 0.02;
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;
    string csv = "results_2d_angled.csv";
};

inline double gamma1p(double a) { return std::tgamma(1.0 + a); }
inline double norm2d_angled(double delta_ux, double delta_uy) {
    return 1.0 / (2.0 * (std::cosh(delta_ux / 2.0) + std::cosh(delta_uy / 2.0)));
}
inline double clamp_force_for_width(double F_in, int w, double c_small_force) {
    double bound = c_small_force / (double(w) * double(w));
    return std::min(F_in, bound);
}
static inline uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Prob2D {
    double p_plus_x, p_minus_x, p_plus_y, p_minus_y;
    double delta_ux, delta_uy;
};
Prob2D compute_probabilities_2d_angled(double F_eff, double theta_rad, double a = 1.0) {
    Prob2D p;
    p.delta_ux = a * std::cos(theta_rad) * F_eff;
    p.delta_uy = a * std::sin(theta_rad) * F_eff;
    double B = norm2d_angled(p.delta_ux, p.delta_uy);
    p.p_plus_x  = B * std::exp(p.delta_ux / 2.0);
    p.p_minus_x = B * std::exp(-p.delta_ux / 2.0);
    p.p_plus_y  = B * std::exp(p.delta_uy / 2.0);
    p.p_minus_y = B * std::exp(-p.delta_uy / 2.0);
    return p;
}

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

struct CumProb2D { double c1, c2, c3, c4; };
inline CumProb2D build_cum_2d(const Prob2D& p) {
    CumProb2D c;
    c.c1 = p.p_plus_x;
    c.c2 = c.c1 + p.p_minus_x;
    c.c3 = c.c2 + p.p_plus_y;
    c.c4 = 1.0;
    return c;
}
pair<int64_t, int64_t> simulate_trajectory_2d(int w_y, const Prob2D &p, double alpha, double t,
                                              std::mt19937_64 &rng, uint64_t &steps_out) {
    uint64_t steps = generate_steps(alpha, t, rng);
    steps_out = steps;
    int64_t x = 0; int y = 0;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    CumProb2D cp = build_cum_2d(p);
    int64_t x_sum = 0, y_sum = 0;
    for (uint64_t s = 0; s < steps; ++s) {
        double r = U(rng);
        if (r < cp.c1) { ++x; ++x_sum; }
        else if (r < cp.c2) { --x; --x_sum; }
        else if (r < cp.c3) { y = (y + 1) % w_y; ++y_sum; }
        else { y = (y - 1 + w_y) % w_y; --y_sum; }
    }
    return make_pair(x_sum, y_sum);
}

struct Accum2D {
    double x_sum = 0.0, y_sum = 0.0;
    uint64_t steps_max = 0ULL;
    uint64_t steps_sum = 0ULL;
};
void worker_block_2d(int w_y, const Prob2D &p, double alpha, double t,
                     uint64_t seed, int i0, int i1, Accum2D &out) {
    std::mt19937_64 rng(seed);
    double local_x = 0.0, local_y = 0.0;
    uint64_t local_steps_max = 0ULL, local_steps_sum = 0ULL;
    for (int i = i0; i < i1; ++i) {
        uint64_t steps_out = 0ULL;
        auto [x_disp, y_disp] = simulate_trajectory_2d(w_y, p, alpha, t, rng, steps_out);
        local_x += (double)x_disp;
        local_y += (double)y_disp;
        local_steps_sum += steps_out;
        if (steps_out > local_steps_max) local_steps_max = steps_out;
    }
    out.x_sum = local_x;
    out.y_sum = local_y;
    out.steps_max = local_steps_max;
    out.steps_sum = local_steps_sum;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    SimConfig cfg;
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;
    double theta_rad = cfg.theta_deg * M_PI / 180.0;
    vector<double> times(cfg.measurement_points);
    double lgmin = std::log10(cfg.t_min), lgmax = std::log10(cfg.t_max);
    double d = (lgmax - lgmin) / (cfg.measurement_points - 1);
    for (int i = 0; i < cfg.measurement_points; ++i) times[i] = std::pow(10.0, lgmin + d * i);
    ofstream out(cfg.csv);
    out << "w,t,F_used,theta_deg,average_x,average_y,C_sim_x,C_sim_y,steps_max,steps_mean,delta_ux,delta_uy\n";
    cout << "Threads: " << n_threads << ", N_traj: " << cfg.N_traj << "\n";
    cout << "Alpha: " << cfg.alpha << ", F_input: " << cfg.F_input << ", theta: " << cfg.theta_deg << " degrees\n";
    cout << "Effective drift " << (cfg.use_effective_drift ? "ON" : "OFF") << "\n";
    for (int w : cfg.Ws) {
        cout << "\nWidth w=" << w << "\n";
        double F_use_nom = clamp_force_for_width(cfg.F_input, w, cfg.c_small_force);
        for (int m = 0; m < cfg.measurement_points; ++m) {
            double t = times[m];
            double F_use = F_use_nom;
            double F_eff = cfg.use_effective_drift
                ? cfg.kappa_alpha * std::pow(F_use, cfg.alpha) * std::pow((double)w, -(1.0 - cfg.alpha)) // corrected for 2D
                : F_use;
            Prob2D p = compute_probabilities_2d_angled(F_eff, theta_rad);
            int per = cfg.N_traj / (int)n_threads;
            vector<thread> pool;
            vector<Accum2D> acc(n_threads);
            uint64_t seed = cfg.base_seed ^ (uint64_t)w ^ (uint64_t)m;
            for (unsigned int th = 0; th < n_threads; ++th) {
                uint64_t s = seed; for (unsigned int k = 0; k <= th; ++k) s = splitmix64(s);
                int i0 = th * per; int i1 = (th == n_threads - 1) ? cfg.N_traj : i0 + per;
                pool.emplace_back(worker_block_2d, w, cref(p), cfg.alpha, t, s, i0, i1, ref(acc[th]));
            }
            for (auto &th : pool) th.join();
            double x_sum = 0.0, y_sum = 0.0;
            uint64_t steps_max = 0ULL;
            unsigned __int128 steps_sum128 = 0;
            for (auto &a : acc) {
                x_sum += a.x_sum;
                y_sum += a.y_sum;
                steps_max = max(steps_max, a.steps_max);
                steps_sum128 += a.steps_sum;
            }
            double avg_steps = (double)steps_sum128 / (double)cfg.N_traj;
            double avg_x = x_sum / (double)cfg.N_traj;
            double avg_y = y_sum / (double)cfg.N_traj;
            double cos_theta = std::cos(theta_rad);
            double sin_theta = std::sin(theta_rad);
            // **Key correction for 2D geometry**:
            double C_sim_x = avg_x * std::pow(t, -cfg.alpha)
                  * std::pow(cos_theta * F_use, -cfg.alpha)
                  * std::pow((double)w, 1.0 - cfg.alpha); // <<--- correct for 2D!
            double C_sim_y = avg_y * std::pow(t, -cfg.alpha)
                  * std::pow(sin_theta * F_use, -cfg.alpha)
                  * std::pow((double)w, 1.0 - cfg.alpha); // also correct for confined axis
            cout << "t=" << std::scientific << t << "  <x>=" << avg_x << "  <y>=" << avg_y
                 << "  C_sim_x=" << C_sim_x << "  C_sim_y=" << C_sim_y << "  steps_max=" << steps_max << "\n";
            out << w << "," << std::setprecision(17) << t << "," << F_use << ","
                << cfg.theta_deg << ","
                << std::setprecision(17) << avg_x << ","
                << std::setprecision(17) << avg_y << ","
                << std::setprecision(17) << C_sim_x << ","
                << std::setprecision(17) << C_sim_y << ","
                << steps_max << "," << std::setprecision(17) << avg_steps << ","
                << std::setprecision(17) << p.delta_ux << "," << std::setprecision(17) << p.delta_uy << "\n";
        }
    }
    out.close();
    cerr << "\nTheoretical validation: For 2D channel: K_lead = 4^{-alpha}/(A Gamma(1+alpha)^2)\n";
    cerr << "C_sim_x should plateau for large t and all w, and match K_lead from theory (for correct A)!\n";
    return 0;
}
