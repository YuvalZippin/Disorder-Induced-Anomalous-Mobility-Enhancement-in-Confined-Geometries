// full_sim_mu_vs_time_up.cpp
// Upgraded μ(t) simulator for 2D/3D periodic strips with CTRW and detailed-balance kernel.

#include <bits/stdc++.h>
using namespace std;

// ---------------------- CLI ----------------------
struct Args {
    int dim = 3;                // 2 or 3
    int w = 5;                  // strip width in transverse direction(s)
    int N_traj = 30000;        // trajectories per time point
    int M = 5;                 // number of time points
    double t_min = 1e1;
    double t_max = 1e17;
    double alpha = 0.3;
    double F = 0.01;            // physical force (used in μ normalization)
    double kappa_alpha = 1.0;   // effective-drift amplitude
    bool use_Feff = true;       // use F_eff on x
    double Klead = 0.19;        // optional theory amplitude for μ∞
    bool use_tail_muinf = false;// estimate μ∞ from tail instead of Klead
    int tail_points = 6;        // points for tail median
    double c_small_force = 0.02;// guard: F <= c/w^2
    string csv = "results_mu_vs_time_up.csv";
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;
    bool verbose_checks = true; // print prob sums, drift, steps/t^alpha
};

static inline bool get_arg(int argc, char** argv, const string& key) {
    for (int i=1;i<argc;i++) if (string(argv[i])==key) return true;
    return false;
}
template<typename T> static inline T get_argv(int argc, char** argv, const string& key, T def) {
    for (int i=1;i<argc-1;i++) if (string(argv[i])==key) {
        if constexpr (is_same<T,string>::value) return string(argv[i+1]);
        else {
            std::istringstream is(argv[i+1]); T v; is>>v; return v;
        }
    }
    return def;
}

Args parse_args(int argc, char** argv){
    Args a;
    a.dim = get_argv<int>(argc,argv,"--dim",a.dim);
    a.w   = get_argv<int>(argc,argv,"--w",a.w);
    a.N_traj = get_argv<int>(argc,argv,"--N",a.N_traj);
    a.M   = get_argv<int>(argc,argv,"--M",a.M);
    a.t_min = get_argv<double>(argc,argv,"--tmin",a.t_min);
    a.t_max = get_argv<double>(argc,argv,"--tmax",a.t_max);
    a.alpha = get_argv<double>(argc,argv,"--alpha",a.alpha);
    a.F = get_argv<double>(argc,argv,"--F",a.F);
    a.kappa_alpha = get_argv<double>(argc,argv,"--kappa",a.kappa_alpha);
    a.use_Feff = !get_arg(argc,argv,"--no-feff");
    a.Klead = get_argv<double>(argc,argv,"--Klead",a.Klead);
    a.use_tail_muinf = get_arg(argc,argv,"--auto-tail");
    a.tail_points = get_argv<int>(argc,argv,"--tail-points",a.tail_points);
    a.c_small_force = get_argv<double>(argc,argv,"--c-small",a.c_small_force);
    a.csv = get_argv<string>(argc,argv,"--csv",a.csv);
    a.base_seed = get_argv<uint64_t>(argc,argv,"--seed",a.base_seed);
    a.verbose_checks = !get_arg(argc,argv,"--quiet");
    return a;
}

// ---------------------- RNG & math ----------------------
inline double gamma1p(double a) { return std::tgamma(1.0 + a); } // Γ(1+α)

static inline uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// CTRW steps via CMS-based one-sided α-stable mapping
uint64_t generate_steps(double alpha, double t, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_real_distribution<double> Utheta(0.0, M_PI);
    double theta = Utheta(rng);
    double x = U(rng);
    double W = -std::log(x);
    double a_num = std::sin((1.0 - alpha) * theta) *
                   std::pow(std::sin(alpha * theta), alpha / (1.0 - alpha));
    double a_den = std::pow(std::sin(theta), 1.0 / (1.0 - alpha));
    double a_theta = a_num / a_den;
    double eta = std::pow(a_theta / W, (1.0 - alpha) / alpha);
    uint64_t s = (uint64_t)std::ceil(std::pow(t / eta, alpha));
    if (s < 1ULL) s = 1ULL;
    return s;
}

// Small-force clamp F <= c/w^2
inline double clamp_force_for_width(double F_in, int w, double c_small_force) {
    double bound = c_small_force / (double(w) * double(w));
    return std::min(F_in, bound);
}

// ---------------------- Probabilities (2D/3D) ----------------------
struct Probs {
    // ordering: +x,-x, +y,-y, +z,-z (z unused in 2D)
    array<double,6> p{};
    int used = 4; // 4 in 2D, 6 in 3D
};
struct CumProb { array<double,6> c{}; int used=4; };

inline Probs compute_probs(int dim, double Feff){
    // neighbors: 2 along x, and 2*(dim-1) transverse
    int used = (dim==2?4:6);
    double B = 1.0 / (2.0*std::cosh(Feff/2.0) + 2.0*(dim-1));
    Probs pr; pr.used = used;
    pr.p[0] = B * std::exp(+Feff/2.0); // +x
    pr.p[1] = B * std::exp(-Feff/2.0); // -x
    pr.p[2] = B;                       // +y
    pr.p[3] = B;                       // -y
    pr.p[4] = (dim==3? B : 0.0);       // +z
    pr.p[5] = (dim==3? B : 0.0);       // -z
    return pr;
}
inline CumProb build_cum(const Probs& pr){
    CumProb cp; cp.used = pr.used;
    double s=0.0;
    for(int i=0;i<pr.used;i++){ s+=pr.p[i]; cp.c[i]=s; }
    if(pr.used==6) cp.c[5] = 1.0;
    else if(pr.used==4) cp.c[3] = 1.0;
    return cp;
}

// ---------------------- Trajectory ----------------------
struct Accum {
    double x_sum = 0.0;
    uint64_t steps_max = 0ULL;
    uint64_t steps_sum = 0ULL;
};

int64_t simulate_traj(int dim, int w, const Probs &pr,
                      double alpha, double t, std::mt19937_64 &rng, uint64_t &steps_out) {
    uint64_t steps = generate_steps(alpha, t, rng);
    steps_out = steps;
    int64_t x_disp = 0;
    int y=0,z=0;

    std::uniform_real_distribution<double> U(0.0,1.0);
    CumProb cp = build_cum(pr);

    for(uint64_t s=0; s<steps; ++s){
        double r = U(rng);
        if(r < cp.c[0]) { ++x_disp; }
        else if(r < cp.c[1]) { --x_disp; }
        else if(r < cp.c[2]) { y = (y+1)%w; }
        else if(r < cp.c[3]) { y = (y-1+w)%w; }
        else if(dim==3 && r < cp.c[4]) { z = (z+1)%w; }
        else if(dim==3) { z = (z-1+w)%w; }
    }
    return x_disp;
}

void worker_block(int dim, int w, const Probs &pr, double alpha, double t,
                  uint64_t seed, int i0, int i1, Accum &out) {
    std::mt19937_64 rng(seed);
    double local_x = 0.0;
    uint64_t local_steps_max = 0ULL;
    uint64_t local_steps_sum = 0ULL;
    for(int i=i0;i<i1;++i){
        uint64_t steps_out=0ULL;
        int64_t xd = simulate_traj(dim,w,pr,alpha,t,rng,steps_out);
        local_x += (double)xd;
        local_steps_sum += steps_out;
        if(steps_out > local_steps_max) local_steps_max = steps_out;
    }
    out.x_sum = local_x;
    out.steps_max = local_steps_max;
    out.steps_sum = local_steps_sum;
}

// ---------------------- Main ----------------------
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Args cfg = parse_args(argc,argv);
    if(cfg.dim!=2 && cfg.dim!=3){ cerr<<"dim must be 2 or 3\n"; return 1; }

    unsigned int n_threads = std::thread::hardware_concurrency();
    if(n_threads==0) n_threads=1;

    // log-spaced times
    vector<double> times(cfg.M);
    double lgmin = std::log10(cfg.t_min), lgmax = std::log10(cfg.t_max);
    double d = (cfg.M>1? (lgmax - lgmin)/(cfg.M - 1) : 0.0);
    for(int i=0;i<cfg.M;i++) times[i] = std::pow(10.0, lgmin + d*i);

    // small-force clamp
    double F_use = clamp_force_for_width(cfg.F, cfg.w, cfg.c_small_force);

    // μ∞ from Klead or estimated later from tail
    double mu_inf_from_K = cfg.Klead * std::pow((double)cfg.w, -2.0*(1.0 - cfg.alpha));

    cout<<"=== Upgraded μ(t) sim ===\n";
    cout<<"dim="<<cfg.dim<<" w="<<cfg.w<<" N_traj="<<cfg.N_traj<<" threads="<<n_threads<<"\n";
    cout<<"alpha="<<cfg.alpha<<" F_in="<<cfg.F<<" F_use="<<F_use<<" use_Feff="<<(cfg.use_Feff?"ON":"OFF")<<"\n";
    cout<<"Klead="<<cfg.Klead<<" -> mu_inf(K)="<<mu_inf_from_K<<"  auto_tail="<<(cfg.use_tail_muinf?"ON":"OFF")<<"\n\n";

    ofstream out(cfg.csv);
    out<<"dim,w,t,F_used,F_eff,average_x,mu_sim,mu_inf,R_of_t,steps_max,steps_mean,prob_sum,delta_x\n";

    vector<double> mu_tail; mu_tail.reserve(cfg.M);

    for(int m=0;m<cfg.M;m++){
        double t = times[m];

        // Effective drift for x
        double F_eff = cfg.use_Feff
            ? cfg.kappa_alpha * std::pow(F_use, cfg.alpha) * std::pow((double)cfg.w, -2.0*(1.0 - cfg.alpha))
            : F_use;

        // Probabilities (2D/3D) with one B for all neighbors
        Probs pr = compute_probs(cfg.dim, F_eff);

        // Diagnostics: sum and delta
        double p_sum = 0.0; for(int i=0;i<pr.used;i++) p_sum += pr.p[i];
        double delta_x = pr.p[0] - pr.p[1];

        if(cfg.verbose_checks){
            cout<<fixed<<setprecision(6)
                <<"t="<<t<<"  B-sum="<<p_sum<<"  delta_x="<<delta_x<<"\n";
        }

        // Launch threads
        int per = cfg.N_traj / (int)n_threads;
        vector<thread> pool;
        vector<Accum> acc(n_threads);
        uint64_t seed = cfg.base_seed ^ (uint64_t)m;
        for(unsigned int th=0; th<n_threads; ++th){
            uint64_t s=seed; for(unsigned int k=0;k<=th;++k) s=splitmix64(s);
            int i0 = th*per;
            int i1 = (th==n_threads-1)? cfg.N_traj : i0+per;
            pool.emplace_back(worker_block, cfg.dim, cfg.w, cref(pr), cfg.alpha, t, s, i0, i1, ref(acc[th]));
        }
        for(auto& th: pool) th.join();

        // Reduce
        double x_sum=0.0; uint64_t steps_max=0ULL; unsigned __int128 steps_sum128=0;
        for(auto &a: acc){ x_sum+=a.x_sum; steps_max=max(steps_max,a.steps_max); steps_sum128+=a.steps_sum; }
        double avg_steps = (double)steps_sum128 / (double)cfg.N_traj;
        double avg_x = x_sum / (double)cfg.N_traj;

        // μ_sim
        double mu_sim = avg_x / (std::pow(F_use, cfg.alpha) * std::pow(t, cfg.alpha));
        mu_tail.push_back(mu_sim);

        // choose μ∞ source
        double mu_inf = mu_inf_from_K;
        if(cfg.use_tail_muinf){
            int n = std::min(cfg.tail_points, (int)mu_tail.size());
            vector<double> tail(mu_tail.end()-n, mu_tail.end());
            std::nth_element(tail.begin(), tail.begin()+n/2, tail.end());
            mu_inf = tail[n/2];
        }
        double R_of_t = (mu_inf>0.0)? (mu_sim/mu_inf) : std::numeric_limits<double>::quiet_NaN();

        if(cfg.verbose_checks){
            cout<<scientific<<setprecision(4)
                <<"   <x>="<<avg_x<<"  mu="<<mu_sim<<"  R="<<R_of_t
                <<"  steps/t^a="<<(avg_steps/std::pow(t,cfg.alpha))<<"\n";
        }

        out<<cfg.dim<<","<<cfg.w<<","<<setprecision(17)<<t<<","
           <<F_use<<","<<F_eff<<","<<avg_x<<","<<mu_sim<<","
           <<mu_inf<<","<<R_of_t<<","<<steps_max<<","<<avg_steps<<","
           <<p_sum<<","<<delta_x<<"\n";
    }

    out.close();
    cout<<"\nSaved CSV: "<<cfg.csv<<"\n";
    return 0;
}
