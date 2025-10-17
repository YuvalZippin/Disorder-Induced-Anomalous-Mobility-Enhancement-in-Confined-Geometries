// full_sim_mu_vs_time.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o full_sim_mu_vs_time full_sim_mu_vs_time.cpp
// Run:     ./full_sim_mu_vs_time --w 5 --alpha 0.3 --F 0.01 --Klead 0.19 --N 30000 --M 17 --tmin 1e4 --tmax 1e15

#include <bits/stdc++.h>
using namespace std;

struct Args {
    int w = 5;
    int N_traj = 300000;
    int M = 10;
    double t_min = 1e7, t_max = 1e17;
    double alpha = 0.3;
    double F = 0.01;                 // physical F (used in μ normalization)
    double kappa_alpha = 1.0;        // amplitude in F_eff
    bool use_Feff = true;            // apply effective drift on x moves only
    double Klead = 0.19;             // for μ∞ = Klead * w^{-2(1-α)}
    bool auto_tail_muinf = false;    // estimate μ∞ from tail instead of Klead
    int tail_points = 6;
    double c_small_force = 0.02;     // small-force guard: F <= c/w^2
    string csv = "results_mu_vs_time_3d.csv";
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;
    bool quiet = false;
};

static inline bool has_flag(int argc, char** argv, const string& k){
    for(int i=1;i<argc;i++) if(string(argv[i])==k) return true; return false;
}
template<typename T>
static inline T get_opt(int argc, char** argv, const string& k, T def){
    for(int i=1;i<argc-1;i++) if(string(argv[i])==k){
        if constexpr (is_same<T,string>::value) return string(argv[i+1]);
        T v; std::istringstream(string(argv[i+1]))>>v; return v;
    } return def;
}

Args parse_args(int argc, char** argv){
    Args a;
    a.w = get_opt<int>(argc,argv,"--w",a.w);
    a.N_traj = get_opt<int>(argc,argv,"--N",a.N_traj);
    a.M = get_opt<int>(argc,argv,"--M",a.M);
    a.t_min = get_opt<double>(argc,argv,"--tmin",a.t_min);
    a.t_max = get_opt<double>(argc,argv,"--tmax",a.t_max);
    a.alpha = get_opt<double>(argc,argv,"--alpha",a.alpha);
    a.F = get_opt<double>(argc,argv,"--F",a.F);
    a.kappa_alpha = get_opt<double>(argc,argv,"--kappa",a.kappa_alpha);
    a.use_Feff = !has_flag(argc,argv,"--no-feff");
    a.Klead = get_opt<double>(argc,argv,"--Klead",a.Klead);
    a.auto_tail_muinf = has_flag(argc,argv,"--auto-tail");
    a.tail_points = get_opt<int>(argc,argv,"--tail-points",a.tail_points);
    a.c_small_force = get_opt<double>(argc,argv,"--c-small",a.c_small_force);
    a.csv = get_opt<string>(argc,argv,"--csv",a.csv);
    a.base_seed = get_opt<uint64_t>(argc,argv,"--seed",a.base_seed);
    a.quiet = has_flag(argc,argv,"--quiet");
    return a;
}

// RNG and math
static inline uint64_t splitmix64(uint64_t &x){
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z>>30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z>>27)) * 0x94d049bb133111ebULL;
    return z ^ (z>>31);
}
inline double clamp_force_for_width(double F, int w, double c){
    double bound = c / (double(w)*double(w)); return std::min(F, bound);
}
// One-sided alpha-stable CMS mapping to CTRW step count
uint64_t generate_steps(double alpha, double t, std::mt19937_64 &rng){
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::uniform_real_distribution<double> Utheta(0.0, M_PI);
    double theta = Utheta(rng), x = U(rng), W = -std::log(x);
    double a_num = std::sin((1.0-alpha)*theta)*std::pow(std::sin(alpha*theta), alpha/(1.0-alpha));
    double a_den = std::pow(std::sin(theta), 1.0/(1.0-alpha));
    double a_theta = a_num/a_den;
    double eta = std::pow(a_theta/W, (1.0-alpha)/alpha);
    uint64_t s = (uint64_t)std::ceil(std::pow(t/eta, alpha));
    return (s<1ULL)?1ULL:s;
}

// 3D kernel: one B for all 6 moves
struct Probs6 { double pxp, pxm, pyp, pym, pzp, pzm; };
struct Cum6 { double c1,c2,c3,c4,c5,c6; };
inline Probs6 probs3d(double Feff){
    double B = 1.0 / (2.0*std::cosh(Feff/2.0) + 4.0);
    double pxp = B*std::exp(+Feff/2.0), pxm = B*std::exp(-Feff/2.0), p0 = B;
    return Probs6{pxp,pxm,p0,p0,p0,p0};
}
inline Cum6 build_cum(const Probs6& p){
    Cum6 c;
    c.c1 = p.pxp;
    c.c2 = c.c1 + p.pxm;
    c.c3 = c.c2 + p.pyp;
    c.c4 = c.c3 + p.pym;
    c.c5 = c.c4 + p.pzp;
    c.c6 = 1.0;
    return c;
}

// Trajectory
struct Accum{ double x_sum=0.0; uint64_t steps_max=0ULL; uint64_t steps_sum=0ULL; };
int64_t simulate_traj(int w, const Probs6& pr, double alpha, double t, std::mt19937_64& rng, uint64_t& steps_out){
    uint64_t steps = generate_steps(alpha,t,rng); steps_out = steps;
    int64_t x = 0; int y=0,z=0;
    Cum6 cp = build_cum(pr);
    std::uniform_real_distribution<double> U(0.0,1.0);
    for(uint64_t s=0;s<steps;++s){
        double r=U(rng);
        if(r<cp.c1){ ++x; }
        else if(r<cp.c2){ --x; }
        else if(r<cp.c3){ y = (y+1)%w; }
        else if(r<cp.c4){ y = (y-1+w)%w; }
        else if(r<cp.c5){ z = (z+1)%w; }
        else            { z = (z-1+w)%w; }
    }
    return x;
}
void worker(int w, const Probs6& pr, double alpha, double t, uint64_t seed, int i0, int i1, Accum& out){
    std::mt19937_64 rng(seed);
    double lx=0.0; uint64_t smax=0ULL, ssum=0ULL;
    for(int i=i0;i<i1;++i){
        uint64_t so=0ULL; int64_t xd = simulate_traj(w,pr,alpha,t,rng,so);
        lx += (double)xd; ssum += so; if(so>smax) smax=so;
    }
    out.x_sum=lx; out.steps_max=smax; out.steps_sum=ssum;
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    Args cfg = parse_args(argc,argv);

    unsigned int n_threads = std::thread::hardware_concurrency(); if(!n_threads) n_threads=1;

    vector<double> times(cfg.M);
    double lgmin=std::log10(cfg.t_min), lgmax=std::log10(cfg.t_max);
    double d = (cfg.M>1? (lgmax-lgmin)/(cfg.M-1):0.0);
    for(int i=0;i<cfg.M;i++) times[i]=std::pow(10.0, lgmin+d*i);

    double F_use = clamp_force_for_width(cfg.F, cfg.w, cfg.c_small_force);
    double mu_inf_from_K = cfg.Klead * std::pow((double)cfg.w, -2.0*(1.0 - cfg.alpha));

    if(!cfg.quiet){
        cout<<"3D μ(t) sim | w="<<cfg.w<<" N="<<cfg.N_traj<<" threads="<<n_threads<<"\n";
        cout<<"alpha="<<cfg.alpha<<" F_in="<<cfg.F<<" F_use="<<F_use
            <<" Klead="<<cfg.Klead<<" mu_inf(K)="<<mu_inf_from_K<<"\n\n";
    }

    ofstream out(cfg.csv);
    out<<"w,t,F_used,F_eff,average_x,mu,mu_inf,R_of_t,steps_mean,steps_max,prob_sum,delta_x\n";

    vector<double> tail; tail.reserve(cfg.M);

    for(int m=0;m<cfg.M;m++){
        double t = times[m];
        double F_eff = cfg.use_Feff? (cfg.kappa_alpha*std::pow(F_use, cfg.alpha)*std::pow((double)cfg.w, -2.0*(1.0 - cfg.alpha)))
                                   : F_use;

        Probs6 pr = probs3d(F_eff);
        double prob_sum = pr.pxp+pr.pxm+pr.pyp+pr.pym+pr.pzp+pr.pzm;
        double delta_x  = pr.pxp - pr.pxm;
        if(!cfg.quiet){
            cout<<fixed<<setprecision(6)<<"t="<<t<<" prob_sum="<<prob_sum<<" delta_x="<<delta_x<<"\n";
        }

        int per = cfg.N_traj/(int)n_threads;
        vector<thread> pool; vector<Accum> acc(n_threads);
        uint64_t seed = cfg.base_seed ^ (uint64_t)m ^ (uint64_t)cfg.w;
        for(unsigned int th=0; th<n_threads; ++th){
            uint64_t s=seed; for(unsigned int k=0;k<=th;++k) s=splitmix64(s);
            int i0=th*per, i1=(th==n_threads-1? cfg.N_traj : i0+per);
            pool.emplace_back(worker, cfg.w, cref(pr), cfg.alpha, t, s, i0, i1, ref(acc[th]));
        }
        for(auto &th: pool) th.join();

        double x_sum=0.0; uint64_t smax=0ULL; unsigned __int128 ssum128=0;
        for(auto &a: acc){ x_sum+=a.x_sum; ssum128+=a.steps_sum; if(a.steps_max>smax) smax=a.steps_max; }
        double avg_x = x_sum/(double)cfg.N_traj;
        double steps_mean = (double)ssum128/(double)cfg.N_traj;

        double mu = avg_x / (std::pow(F_use, cfg.alpha) * std::pow(t, cfg.alpha)); // physical F in denominator
        tail.push_back(mu);

        double mu_inf = mu_inf_from_K;
        if(cfg.auto_tail_muinf){
            int n = std::min(cfg.tail_points, (int)tail.size());
            vector<double> last(tail.end()-n, tail.end());
            last.erase(remove_if(last.begin(), last.end(),
                                 [](double v){ return !std::isfinite(v) || v<=0.0; }),
                       last.end());
            if(!last.empty()){
                std::nth_element(last.begin(), last.begin()+last.size()/2, last.end());
                mu_inf = last[last.size()/2];
            }
        }
        double R = (mu_inf>0.0)? (mu/mu_inf) : std::numeric_limits<double>::quiet_NaN();

        if(!cfg.quiet){
            cout<<scientific<<setprecision(4)
                <<"   <x>="<<avg_x<<" mu="<<mu<<" R="<<R
                <<" steps/t^a="<<(steps_mean/std::pow(t,cfg.alpha))<<"\n";
        }

        out<<cfg.w<<","<<std::setprecision(17)<<t<<","<<F_use<<","<<F_eff<<","
           <<avg_x<<","<<mu<<","<<mu_inf<<","<<R<<","
           <<steps_mean<<","<<smax<<","<<prob_sum<<","<<delta_x<<"\n";
    }
    out.close();
    if(!cfg.quiet) cout<<"\nSaved CSV: "<<cfg.csv<<"\n";
    return 0;
}
