// qtm_firstmoment_periodic_fast.cpp
// Quenched Trap Model (QTM), 3D channel with periodic y,z and infinite x
// Fast version: deterministic quenched field via coordinate hashing, multithreaded workers
// Compile: g++ -std=c++17 -O3 -pthread -o qtm_firstmoment_periodic_fast qtm_firstmoment_periodic_fast.cpp
// Run:     ./qtm_firstmoment_periodic_fast

#include <bits/stdc++.h>
using namespace std;

// ---------------------- Config ----------------------
struct SimConfig {
    int w = 3;                          // channel width (y,z)
    double F = 0.01;                    // field along x
    double alpha = 0.3;                 // trap tail exponent
    int N_traj = 300000;                // trajectories per disorder sample
    int N_disorder = 1;                 // disorder samples (≥1)
    int measurement_points = 100;        // log-spaced time points
    double t_min = 1e5, t_max = 1e17;   // time window
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;
    string csv = "qtm_firstmoment_periodic_fast.csv";
};

// ---------------------- RNG & hashing ----------------------
// SplitMix64 (also used to generate deterministic U(0,1) from coordinates)
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline uint64_t seed_thread(uint64_t base, uint64_t salt) {
    uint64_t s = base ^ (salt + 0x632be59bd9b4e019ULL);
    return splitmix64(s);
}

// Deterministic site->U(0,1) from (x,y,z) and disorder_seed
static inline double U01_from_site(int64_t x, int y, int z, uint64_t disorder_seed) {
    uint64_t h = disorder_seed;
    // fold coordinates (cast y,z to 64-bit)
    h ^= splitmix64((uint64_t)x ^ 0x243f6a8885a308d3ULL);
    h ^= splitmix64((uint64_t)y ^ 0x13198a2e03707344ULL);
    h ^= splitmix64((uint64_t)z ^ 0xa4093822299f31d0ULL);
    // convert to double in (0,1)
    const double inv53 = 1.0 / (double)(1ULL << 53);
    double u = ((h >> 11) * inv53);
    if (u <= 0.0) u = std::ldexp(1.0, -52);
    if (u >= 1.0) u = 1.0 - std::ldexp(1.0, -52);
    return u;
}

// Pareto-like trapped time from U via τ = u^{-1/α} (τ0=1)
static inline double tau_from_site(int64_t x, int y, int z, uint64_t disorder_seed, double alpha) {
    double u = U01_from_site(x, y, z, disorder_seed);
    return std::pow(u, -1.0 / alpha);
}

// ---------------------- Probabilities (periodic y,z) ----------------------
struct ProbVec { double c1, c2, c3, c4, c5; }; // cumulative cutoffs; last branch is else
static inline ProbVec build_cum(double F) {
    double B = 1.0 / (2.0 * std::cosh(F/2.0) + 4.0);
    double pxp = B * std::exp(F/2.0), pxm = B * std::exp(-F/2.0);
    double pyp = B, pym = B, pzp = B, pzm = B;
    ProbVec cp;
    cp.c1 = pxp;
    cp.c2 = cp.c1 + pxm;
    cp.c3 = cp.c2 + pyp;
    cp.c4 = cp.c3 + pym;
    cp.c5 = cp.c4 + pzp; // final "else" implicitly covers pzm
    return cp; // sums to 1 with final else
}

// ---------------------- One trajectory to t_max ----------------------
static inline void simulate_trajectory_periodic(
    const SimConfig& cfg,
    const vector<double>& t_grid,
    vector<double>& x_record,           // output: size = t_grid.size()
    std::mt19937_64& rng,
    uint64_t disorder_seed
){
    ProbVec cp = build_cum(cfg.F);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    int64_t x = 0;
    int y = 0, z = 0;
    double t = 0.0;
    size_t ti = 0, M = t_grid.size();

    while (ti < M) {
        // dwell at current site according to quenched τ(x,y,z)
        double tau = tau_from_site(x, y, z, disorder_seed, cfg.alpha);
        t += tau;

        // fill all output times crossed during this dwell
        while (ti < M && t >= t_grid[ti]) {
            x_record[ti] = (double)x;
            ++ti;
        }
        if (ti >= M) break;

        // step decision
        double r = U(rng);
        if (r < cp.c1) {
            ++x;
        } else if (r < cp.c2) {
            --x;
        } else if (r < cp.c3) {
            y = (y + 1) % cfg.w;
        } else if (r < cp.c4) {
            y = (y - 1 + cfg.w) % cfg.w;
        } else if (r < cp.c5) {
            z = (z + 1) % cfg.w;
        } else {
            z = (z - 1 + cfg.w) % cfg.w;
        }
    }
}

// ---------------------- Worker ----------------------
struct BlockAccum {
    vector<double> x_sum;  // length = M
    BlockAccum(size_t M): x_sum(M, 0.0) {}
};

static inline void worker_block(
    const SimConfig& cfg,
    const vector<double>& t_grid,
    uint64_t thread_seed,
    int traj_begin, int traj_end,
    uint64_t disorder_seed,
    BlockAccum& out
){
    std::mt19937_64 rng(thread_seed);
    const size_t M = t_grid.size();
    vector<double> x_rec(M);

    for (int i = traj_begin; i < traj_end; ++i) {
        // per-trajectory RNG stream advance (cheap decorrelation)
        uint64_t throwaway = splitmix64(thread_seed ^ (uint64_t)i);
        (void)throwaway;

        std::fill(x_rec.begin(), x_rec.end(), 0.0);
        simulate_trajectory_periodic(cfg, t_grid, x_rec, rng, disorder_seed);
        for (size_t k=0; k<M; ++k) out.x_sum[k] += x_rec[k];
    }
}

// ---------------------- Main ----------------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    SimConfig cfg;
    // build time grid
    vector<double> t_grid(cfg.measurement_points);
    double lgmin = std::log10(cfg.t_min), lgmax = std::log10(cfg.t_max);
    double d = (lgmax - lgmin) / (cfg.measurement_points - 1);
    for (int i=0;i<cfg.measurement_points;++i) t_grid[i] = std::pow(10.0, lgmin + d*i);

    unsigned nt = std::thread::hardware_concurrency(); if (!nt) nt=1;
    cerr<<"Threads="<<nt<<" N_traj="<<cfg.N_traj<<" w="<<cfg.w<<" F="<<cfg.F
        <<" alpha="<<cfg.alpha<<" N_disorder="<<cfg.N_disorder<<"\n";

    vector<double> x_avg_total(cfg.measurement_points, 0.0);

    for (int ds=0; ds<cfg.N_disorder; ++ds){
        uint64_t disorder_seed = splitmix64(cfg.base_seed ^ (uint64_t)ds);
        // parallel over trajectories
        int per = cfg.N_traj / (int)nt;
        vector<thread> pool;
        vector<BlockAccum> acc; acc.reserve(nt);
        for (unsigned th=0; th<nt; ++th){
            acc.emplace_back(cfg.measurement_points);
        }
        for (unsigned th=0; th<nt; ++th){
            int i0 = th*per;
            int i1 = (th==nt-1 ? cfg.N_traj : i0+per);
            uint64_t thread_seed = seed_thread(cfg.base_seed ^ (uint64_t)ds, (uint64_t)th);
            pool.emplace_back(worker_block, cref(cfg), cref(t_grid), thread_seed, i0, i1, disorder_seed, ref(acc[th]));
        }
        for (auto& th: pool) th.join();

        // reduce
        vector<double> x_mean(cfg.measurement_points, 0.0);
        for (unsigned th=0; th<nt; ++th)
            for (int k=0;k<cfg.measurement_points;++k)
                x_mean[k] += acc[th].x_sum[k];
        for (int k=0;k<cfg.measurement_points;++k)
            x_mean[k] /= (double)cfg.N_traj;

        // accumulate across disorder samples
        for (int k=0;k<cfg.measurement_points;++k)
            x_avg_total[k] += x_mean[k];
    }

    // final average over disorder samples
    for (int k=0;k<cfg.measurement_points;++k)
        x_avg_total[k] /= (double)cfg.N_disorder;

    ofstream out(cfg.csv);
    out<<"t,average_x\n";
    for (int k=0;k<cfg.measurement_points;++k)
        out<<std::setprecision(17)<<t_grid[k]<<","<<std::setprecision(17)<<x_avg_total[k]<<"\n";
    out.close();

    cerr<<"Done. Wrote "<<cfg.csv<<"\n";
    return 0;
}
