// reflecting/full_sim_reflecting.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o full_sim_reflecting full_sim_reflecting.cpp
// Run:     ./full_sim_reflecting
#include <bits/stdc++.h>
using namespace std;

// ---------------------- Config ----------------------
struct SimConfig {
    int N_traj = 1000;                 // increase for production
    int measurement_points = 15;         // log-spaced t points
    double t_min = 1e16;
    double t_max = 1e17;
    double alpha = 0.3;                  // S_alpha
    double F_input = 0.01;               // nominal (physical) force
    vector<int> Ws = {1,3,5,7,10};       // widths along y,z
    bool use_effective_drift = true;     // x-only F_eff
    double kappa_alpha = 1.0;            // amplitude factor for F_eff
    double c_small_force = 0.02;         // enforce F <= c/w^2 (reflecting has a different constant, keep conservative)
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL; // reproducible base
    string csv = "results_reflecting.csv";
};

// ---------------------- Helpers ----------------------
inline double gamma1p(double a){ return std::tgamma(1.0 + a); } // Γ(1+α) [web:3]
inline double norm3d(double halfF){ return 1.0 / (2.0 * std::cosh(halfF) + 4.0); } // 3D cosh normalization [web:15]
inline double clamp_force_for_width(double F, int w, double c){ return std::min(F, c/(double(w)*double(w))); } // window guard [web:15]

// SplitMix64 for deterministic per-thread seeding
static inline uint64_t splitmix64(uint64_t &x){
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ---------------------- Probabilities ----------------------
struct ProbX { double pp, pm; };
struct ProbYZ{ double py_plus, py_minus, pz_plus, pz_minus; };

ProbX probs_x(double F_eff){
    double B = norm3d(F_eff/2.0);
    return ProbX{B*std::exp(F_eff/2.0), B*std::exp(-F_eff/2.0)}; // p±x [web:15]
}
ProbYZ probs_yz_unbiased(){
    double B0 = norm3d(0.0); // 1/(2*1+4)=1/6 for small bias baseline [web:15]
    return ProbYZ{B0, B0, B0, B0}; // p±y, p±z [web:15]
}

struct Cum { double c1,c2,c3,c4,c5,c6; };
inline Cum cum_from(const ProbX& px, const ProbYZ& pyz){
    Cum c;
    c.c1 = px.pp;
    c.c2 = c.c1 + px.pm;
    c.c3 = c.c2 + pyz.py_plus;
    c.c4 = c.c3 + pyz.py_minus;
    c.c5 = c.c4 + pyz.pz_plus;
    c.c6 = 1.0; // remainder to pz_minus
    return c;
}

// ---------------------- CMS one-sided S_alpha ----------------------
// Chambers–Mallows–Stuck generator for one-sided α-stable operational time [web:25][web:3]
uint64_t generate_steps(double alpha, double t, std::mt19937_64& rng){
    std::uniform_real_distribution<double> U(0.0,1.0), Uth(0.0, M_PI);
    double theta = Uth(rng);
    double x = U(rng);
    double W = -std::log(x);
    double a_num = std::sin((1.0 - alpha)*theta)*std::pow(std::sin(alpha*theta), alpha/(1.0 - alpha));
    double a_den = std::pow(std::sin(theta), 1.0/(1.0 - alpha));
    double a_theta = a_num / a_den;
    double eta = std::pow(a_theta / W, (1.0 - alpha)/alpha);
    uint64_t s = (uint64_t)std::ceil(std::pow(t/eta, alpha));
    return s ? s : 1ULL;
}

// ---------------------- Reflecting BC updates ----------------------
// If w==1: coordinate is fixed at 0; if w>=2: reflect attempted exits back inward (specular) [web:167]
inline void step_reflect(int& coord, int w, bool plus){
    if (w <= 1){ coord = 0; return; }
    if (plus){
        if (coord == w-1) coord = w-2; else ++coord;
    } else {
        if (coord == 0) coord = 1; else --coord;
    }
}

// ---------------------- Trajectory ----------------------
int64_t simulate_traj_reflecting(int w, const ProbX& px, const ProbYZ& pyz,
                                 double alpha, double t, std::mt19937_64& rng,
                                 uint64_t& steps_out){
    uint64_t steps = generate_steps(alpha, t, rng); steps_out = steps; // s ~ t^α [web:25]
    int y=0, z=0; // x unbounded; we only track displacement along x [web:15]
    int64_t xsum = 0;
    auto cp = cum_from(px, pyz);
    std::uniform_real_distribution<double> U(0.0,1.0);
    for (uint64_t k=0;k<steps;++k){
        double r = U(rng);
        if (r < cp.c1){ ++xsum; }
        else if (r < cp.c2){ --xsum; }
        else if (r < cp.c3){ step_reflect(y, w, true); }
        else if (r < cp.c4){ step_reflect(y, w, false); }
        else if (r < cp.c5){ step_reflect(z, w, true); }
        else { step_reflect(z, w, false); }
    }
    return xsum;
}

// ---------------------- Worker ----------------------
struct Accum { double x_sum=0.0; uint64_t smax=0, ssum=0; };

void worker(int w, const ProbX& px, const ProbYZ& pyz, double alpha, double t,
            uint64_t seed, int i0, int i1, Accum& out){
    std::mt19937_64 rng(seed);
    double locx=0.0; uint64_t lmax=0, lsum=0;
    for (int i=i0;i<i1;++i){
        uint64_t so=0;
        int64_t xd = simulate_traj_reflecting(w, px, pyz, alpha, t, rng, so);
        locx += (double)xd; lsum += so; if (so>lmax) lmax=so;
    }
    out.x_sum = locx; out.ssum = lsum; out.smax = lmax;
}

// ---------------------- Main ----------------------
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    SimConfig cfg;
    unsigned nt = std::thread::hardware_concurrency(); if (!nt) nt=1;

    // times (log-spaced)
    vector<double> times(cfg.measurement_points);
    double a = std::log10(cfg.t_min), b = std::log10(cfg.t_max), d = (b-a)/(cfg.measurement_points-1);
    for (int i=0;i<cfg.measurement_points;++i) times[i] = std::pow(10.0, a + d*i);

    ofstream out(cfg.csv);
    out << "w,t,F_used,average_x,C_sim,steps_max,steps_mean\n";

    cout << "Threads="<<nt<<" N_traj="<<cfg.N_traj<<" alpha="<<cfg.alpha
         <<" F_input="<<cfg.F_input<<" (reflecting y,z; x unbounded)\n";

    const ProbYZ pyz = probs_yz_unbiased(); // transverse unbiased [web:15]

    for (int w : cfg.Ws){
        cout << "\nWidth w="<<w<<"\n";
        double F_use_nom = clamp_force_for_width(cfg.F_input, w, cfg.c_small_force); // window guard [web:15]
        for (int m=0;m<cfg.measurement_points;++m){
            double t = times[m];

            // effective small-force drift on x (keeps leading F^α and w^{-2(1-α)}) [web:15]
            double F_use = F_use_nom;
            double F_eff = cfg.use_effective_drift
                         ? cfg.kappa_alpha * std::pow(F_use, cfg.alpha) * std::pow((double)w, -2.0*(1.0 - cfg.alpha))
                         : F_use;
            ProbX px = probs_x(F_eff);

            // threads
            int per = cfg.N_traj / (int)nt;
            vector<thread> pool; vector<Accum> acc(nt);
            uint64_t seed = cfg.base_seed ^ (uint64_t)w ^ (uint64_t)m;
            for (unsigned th=0; th<nt; ++th){
                uint64_t s=seed; for (unsigned k=0;k<=th;++k) s = splitmix64(s);
                int i0 = th*per; int i1 = (th==nt-1 ? cfg.N_traj : i0+per);
                pool.emplace_back(worker, w, cref(px), cref(pyz), cfg.alpha, t, s, i0, i1, ref(acc[th]));
            }
            for (auto& th : pool) th.join();

            // reduce
            double xsum=0.0; uint64_t smax=0; unsigned __int128 ssum128=0;
            for (auto& a : acc){ xsum+=a.x_sum; if (a.smax>smax) smax=a.smax; ssum128+=a.ssum; }
            double avg_x = xsum / (double)cfg.N_traj;
            double smean = (double)ssum128 / (double)cfg.N_traj;

            // plateau observable C_sim = <x> t^{-α} F^{-α} w^{2(1-α)} [web:15]
            double C_sim = avg_x * std::pow(t, -cfg.alpha)
                                 * std::pow(F_use, -cfg.alpha)
                                 * std::pow((double)w, 2.0*(1.0 - cfg.alpha));

            cout << std::scientific
                 << "t="<<t<<"  <x>="<<avg_x<<"  C_sim="<<C_sim
                 << "  steps_max="<<smax<<"  steps_mean="<<smean<<"\n";

            out << w << "," << std::setprecision(17) << t << "," << F_use << ","
                << std::setprecision(17) << avg_x << ","
                << std::setprecision(17) << C_sim << ","
                << smax << "," << std::setprecision(17) << smean << "\n";
        }
    }
    out.close();
    cerr << "\nTip: Use the same plotting scripts to check C_sim plateaus and mu(w) slopes under reflecting BC.\n";
    return 0;
}
