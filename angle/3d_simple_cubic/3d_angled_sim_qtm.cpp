// 3d_angled_sim_qtm.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o 3d_angled_sim_qtm 3d_angled_sim_qtm.cpp
#include <bits/stdc++.h>
using namespace std;

struct SimConfig {
    int N_traj = 1000;
    int measurement_points = 15;
    double t_min = 1e16, t_max = 1e17;
    double alpha = 0.3;
    double F_input = 0.01;
    double theta_deg = 60.0; // [0,360]
    double phi_deg = 60.0;   // [0,180]
    vector<int> Ws = {1,3,5,7,9};
    double kappa_alpha = 10.0;    // for F_eff scaling
    double c_small_force = 0.02;
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;
    string csv = "results_3d_angled.csv";
};

inline double norm3d_angled(double dux, double duy, double duz) {
    return 1.0 / (2.0 * (cosh(dux / 2.0) + cosh(duy / 2.0) + cosh(duz / 2.0)));
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

struct Prob3D {
    double p_plus_x, p_minus_x;
    double p_plus_y, p_minus_y;
    double p_plus_z, p_minus_z;
    double delta_ux, delta_uy, delta_uz;
};
Prob3D compute_probabilities_3d_angled(double F_eff, double theta_rad, double phi_rad, double a = 1.0) {
    Prob3D p;
    p.delta_ux = a * cos(theta_rad) * sin(phi_rad) * F_eff;
    p.delta_uy = a * sin(theta_rad) * sin(phi_rad) * F_eff;
    p.delta_uz = a * cos(phi_rad) * F_eff;
    double B = norm3d_angled(p.delta_ux, p.delta_uy, p.delta_uz);
    p.p_plus_x  = B * exp(p.delta_ux / 2.0);
    p.p_minus_x = B * exp(-p.delta_ux / 2.0);
    p.p_plus_y  = B * exp(p.delta_uy / 2.0);
    p.p_minus_y = B * exp(-p.delta_uy / 2.0);
    p.p_plus_z  = B * exp(p.delta_uz / 2.0);
    p.p_minus_z = B * exp(-p.delta_uz / 2.0);
    return p;
}

// CMS one-sided S_alpha waiting time
uint64_t generate_steps(double alpha, double t, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_real_distribution<double> Utheta(0.0, M_PI);
    double theta = Utheta(rng); double x = U(rng); double W = -log(x);
    double a_num = sin((1.0 - alpha) * theta) * pow(sin(alpha * theta), alpha / (1.0 - alpha));
    double a_den = pow(sin(theta), 1.0 / (1.0 - alpha));
    double a_theta = a_num / a_den;
    double eta = pow(a_theta / W, (1.0 - alpha) / alpha);
    uint64_t s = (uint64_t)ceil(pow(t / eta, alpha));
    if (s < 1ULL) s = 1ULL;
    return s;
}

struct CumProb3D { double c1, c2, c3, c4, c5, c6; };
inline CumProb3D build_cum_3d(const Prob3D& p) {
    CumProb3D c;
    c.c1 = p.p_plus_x; c.c2 = c.c1 + p.p_minus_x;
    c.c3 = c.c2 + p.p_plus_y; c.c4 = c.c3 + p.p_minus_y;
    c.c5 = c.c4 + p.p_plus_z; c.c6 = 1.0;
    return c;
}

// Trajectory: X infinite, Y,Z periodic width w
int64_t simulate_trajectory_3d(int w, const Prob3D &p, double alpha, double t, std::mt19937_64 &rng, uint64_t &steps_out) {
    uint64_t steps = generate_steps(alpha, t, rng);
    steps_out = steps;
    int64_t x = 0;
    int y = 0, z = 0;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    CumProb3D cp = build_cum_3d(p);
    int64_t x_sum = 0;
    for (uint64_t s = 0; s < steps; ++s) {
        double r = U(rng);
        if (r < cp.c1) { ++x; ++x_sum; }
        else if (r < cp.c2) { --x; --x_sum; }
        else if (r < cp.c3) { y = (y + 1) % w; }
        else if (r < cp.c4) { y = (y - 1 + w) % w; }
        else if (r < cp.c5) { z = (z + 1) % w; }
        else { z = (z - 1 + w) % w; }
    }
    return x_sum;
}

struct Accum3D {
    double x_sum = 0.0;
    uint64_t steps_max = 0ULL, steps_sum = 0ULL;
};
void worker_block_3d(int w, const Prob3D &p, double alpha, double t,
                     uint64_t seed, int i0, int i1, Accum3D &out) {
    std::mt19937_64 rng(seed);
    double local_x = 0.0;
    uint64_t local_steps_max = 0ULL, local_steps_sum = 0ULL;
    for (int i = i0; i < i1; ++i) {
        uint64_t steps_out = 0ULL;
        int64_t x_disp = simulate_trajectory_3d(w, p, alpha, t, rng, steps_out);
        local_x += (double)x_disp;
        local_steps_sum += steps_out;
        if (steps_out > local_steps_max) local_steps_max = steps_out;
    }
    out.x_sum = local_x;
    out.steps_max = local_steps_max;
    out.steps_sum = local_steps_sum;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    SimConfig cfg;
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;
    double theta_rad = cfg.theta_deg * M_PI / 180.0;
    double phi_rad = cfg.phi_deg * M_PI / 180.0;
    vector<double> times(cfg.measurement_points);
    double lgmin = log10(cfg.t_min), lgmax = log10(cfg.t_max);
    double d = (lgmax - lgmin) / (cfg.measurement_points - 1);
    for (int i = 0; i < cfg.measurement_points; ++i) times[i] = pow(10.0, lgmin + d * i);
    ofstream out(cfg.csv);
    out << "w,t,F_used,theta_deg,phi_deg,average_x,C_sim_x,steps_max,steps_mean,delta_ux,delta_uy,delta_uz\n";
    cout << "Threads: " << n_threads << ", N_traj: " << cfg.N_traj << "\n";
    cout << "Alpha: " << cfg.alpha << ", F_input: " << cfg.F_input << ", theta: " << cfg.theta_deg << " phi: " << cfg.phi_deg << "\n";
    for (int w : cfg.Ws) {
        cout << "\nWidth w=" << w << "\n";
        double F_use_nom = clamp_force_for_width(cfg.F_input, w, cfg.c_small_force);
        for (int m = 0; m < cfg.measurement_points; ++m) {
            double t = times[m];
            double F_eff = cfg.kappa_alpha * pow(F_use_nom, cfg.alpha) * pow((double)w, -(2.0 * (1.0 - cfg.alpha))); // 2D periodic
            Prob3D p = compute_probabilities_3d_angled(F_eff, theta_rad, phi_rad);
            int per = cfg.N_traj / (int)n_threads;
            vector<thread> pool;
            vector<Accum3D> acc(n_threads);
            uint64_t seed = cfg.base_seed ^ (uint64_t)w ^ (uint64_t)m;
            for (unsigned int th = 0; th < n_threads; ++th) {
                uint64_t s = seed; for (unsigned int k = 0; k <= th; ++k) s = splitmix64(s);
                int i0 = th * per; int i1 = (th == n_threads - 1) ? cfg.N_traj : i0 + per;
                pool.emplace_back(worker_block_3d, w, cref(p), cfg.alpha, t, s, i0, i1, ref(acc[th]));
            }
            for (auto &th : pool) th.join();
            double x_sum = 0.0;
            uint64_t steps_max = 0ULL; unsigned __int128 steps_sum128 = 0;
            for (auto &a : acc) { x_sum += a.x_sum; steps_max = max(steps_max, a.steps_max); steps_sum128 += a.steps_sum; }
            double avg_steps = (double)steps_sum128 / (double)cfg.N_traj;
            double avg_x = x_sum / (double)cfg.N_traj;
            // Plateau observable for 3D channel theory:
            double cos_theta = cos(theta_rad);
            double sin_theta = sin(theta_rad);
            double C_sim_x = avg_x * pow(t,-cfg.alpha)
                  * pow(cos_theta * sin(phi_rad) * F_use_nom, -cfg.alpha)
                  * pow((double)w, 2.0 * (1.0 - cfg.alpha)); // 3D channel enhancement
            cout << "t=" << scientific << t << "  <x>=" << avg_x << "  C_sim_x=" << C_sim_x << "  steps_max=" << steps_max << "\n";
            out << w << "," << setprecision(17) << t << "," << F_use_nom << ","
                << cfg.theta_deg << "," << cfg.phi_deg << ","
                << setprecision(17) << avg_x << ","
                << setprecision(17) << C_sim_x << ","
                << steps_max << "," << setprecision(17) << avg_steps << ","
                << setprecision(17) << p.delta_ux << ","
                << setprecision(17) << p.delta_uy << ","
                << setprecision(17) << p.delta_uz << "\n";
        }
    }
    out.close();
    cerr << "\nTheoretical validation: For 3D channel: C_sim_x is expected to plateau and match theory amplitude K_lead = 6^{-alpha}/(A Gamma(1+alpha)^2)\n";
    return 0;
}
