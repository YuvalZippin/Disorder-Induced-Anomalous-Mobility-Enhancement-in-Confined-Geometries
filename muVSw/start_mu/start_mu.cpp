// start_mu/start_mu.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o start_mu start_mu.cpp
// Run example (periodic):   ./start_mu --bc=periodic --alpha=0.3 --F=0.01 --w=5
// Run example (reflecting): ./start_mu --bc=reflecting --alpha=0.3 --F=0.01 --w=5
#include <bits/stdc++.h>
using namespace std;

// ---------------- CLI ----------------
struct CLI {
    string bc = "periodic";     // "periodic" or "reflecting"
    double alpha = 0.3;
    double F_input = 0.01;
    int w = 5;
    int t_points = 5000;          // dense t sampling
    double t_min = 1e10, t_max = 1e17;
    int N0 = 1000;            // initial trajectories per t
    int Nmax = 5000;         // cap
    double rel_se_target = 0.05;// target SE[C_sim]/C_sim
    int max_batches = 6;        // safety cap for doubling
    int slope_k = 7;            // sliding window points
    double c_small_force = 0.02;// F <= c/w^2 guard
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL;
    string csv = "start_mu_results.csv";
};

static inline uint64_t splitmix64(uint64_t &x){
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline double norm3d(double halfF){ return 1.0 / (2.0*std::cosh(halfF) + 4.0); } // [web:15]
inline double clamp_force_for_width(double F, int w, double c){ return std::min(F, c/(double(w)*double(w))); } // [web:15]

// CMS S_alpha steps (one-sided) [web:3]
uint64_t generate_steps(double alpha, double t, std::mt19937_64& rng){
    std::uniform_real_distribution<double> U(0.0,1.0), Uth(0.0,M_PI);
    double theta = Uth(rng), x=U(rng), W=-std::log(x);
    double a_num = std::sin((1.0-alpha)*theta) * std::pow(std::sin(alpha*theta), alpha/(1.0-alpha));
    double a_den = std::pow(std::sin(theta), 1.0/(1.0-alpha));
    double a_theta = a_num / a_den;
    double eta = std::pow(a_theta / W, (1.0-alpha)/alpha);
    uint64_t s = (uint64_t)std::ceil(std::pow(t/eta, alpha));
    return s? s:1ULL;
}

// probabilities
struct ProbX{ double pp, pm; };
struct ProbYZ{ double py_p, py_m, pz_p, pz_m; };

ProbX probs_x(double F_eff){
    double B = norm3d(F_eff/2.0);
    return ProbX{B*std::exp(F_eff/2.0), B*std::exp(-F_eff/2.0)};
}
ProbYZ probs_yz_unbiased(){
    double B0 = norm3d(0.0);
    return ProbYZ{B0,B0,B0,B0};
}
struct Cum{ double c1,c2,c3,c4,c5,c6; };
inline Cum cum_from(const ProbX& px, const ProbYZ& pyz){
    Cum c;
    c.c1=px.pp; c.c2=c.c1+px.pm;
    c.c3=c.c2+pyz.py_p; c.c4=c.c3+pyz.py_m;
    c.c5=c.c4+pyz.pz_p; c.c6=1.0;
    return c;
}

// reflecting step for y or z [web:15]
inline void step_reflect(int& coord, int w, bool plus){
    if (w<=1){ coord=0; return; }
    if (plus){ coord = (coord==w-1)? w-2 : coord+1; }
    else     { coord = (coord==0)? 1 : coord-1; }
}

// simulate one trajectory given BC
int64_t simulate_traj(int w, const string& bc, const ProbX& px, const ProbYZ& pyz,
                      double alpha, double t, std::mt19937_64& rng, uint64_t& steps_out){
    uint64_t steps = generate_steps(alpha, t, rng); steps_out=steps;
    int64_t xsum=0; int y=0,z=0;
    auto cp = cum_from(px,pyz);
    std::uniform_real_distribution<double> U(0.0,1.0);
    for (uint64_t k=0;k<steps;++k){
        double r = U(rng);
        if (r < cp.c1){ ++xsum; }
        else if (r < cp.c2){ --xsum; }
        else if (r < cp.c3){
            if (bc=="periodic") y = (y+1)%w; else step_reflect(y,w,true);
        } else if (r < cp.c4){
            if (bc=="periodic") y = (y-1+w)%w; else step_reflect(y,w,false);
        } else if (r < cp.c5){
            if (bc=="periodic") z = (z+1)%w; else step_reflect(z,w,true);
        } else {
            if (bc=="periodic") z = (z-1+w)%w; else step_reflect(z,w,false);
        }
    }
    return xsum;
}

// block bootstrap SE for mean (trajectory-level) [web:222][web:224]
double bootstrap_se_mean(const vector<double>& samples, int B=200){
    if (samples.empty()) return NAN;
    std::mt19937_64 rng(1234567);
    std::uniform_int_distribution<int> U(0,(int)samples.size()-1);
    vector<double> boots; boots.reserve(B);
    for (int b=0;b<B;++b){
        double s=0.0; for (size_t i=0;i<samples.size();++i) s+=samples[U(rng)];
        boots.push_back(s / (double)samples.size());
    }
    double m = std::accumulate(boots.begin(), boots.end(), 0.0)/boots.size();
    double v = 0.0; for (double v_i: boots){ double d=v_i-m; v+=d*d; }
    v /= (boots.size()>1? boots.size()-1:1);
    return std::sqrt(v);
}

// sliding window slope α_fit around index i (k points total) [web:223]
double local_loglog_slope(const vector<double>& t, const vector<double>& x, int i, int k){
    int n = (int)t.size(); if (n<2) return NAN;
    int half = k/2;
    int L = std::max(0, i-half), R = std::min(n-1, L + k - 1);
    L = std::max(0, R - k + 1);
    int m = R-L+1; if (m<2) return NAN;
    double sx=0, sy=0, sxx=0, sxy=0;
    for (int j=L;j<=R;++j){
        double X = std::log(t[j]);
        double Y = std::log(std::max(x[j], 1e-300));
        sx+=X; sy+=Y; sxx+=X*X; sxy+=X*Y;
    }
    double denom = m*sxx - sx*sx; if (std::abs(denom)<1e-300) return NAN;
    return (m*sxy - sx*sy)/denom;
}

int main(int argc, char** argv){
    CLI cfg;
    // minimal CLI parsing
    for (int i=1;i<argc;++i){
        string a=argv[i];
        auto next=[&](double& v){ v=stod(argv[++i]); };
        auto nexts=[&](string& v){ v=string(argv[++i]); };
        if (a=="--bc") nexts(cfg.bc);
        else if (a=="--alpha") next(cfg.alpha);
        else if (a=="--F") next(cfg.F_input);
        else if (a=="--w") { double tmp; next(tmp); cfg.w=(int)llround(tmp); }
        else if (a=="--tmin") next(cfg.t_min);
        else if (a=="--tmax") next(cfg.t_max);
        else if (a=="--tpoints") { double tmp; next(tmp); cfg.t_points=(int)llround(tmp); }
        else if (a=="--N0") { double tmp; next(tmp); cfg.N0=(int)llround(tmp); }
        else if (a=="--Nmax") { double tmp; next(tmp); cfg.Nmax=(int)llround(tmp); }
        else if (a=="--relse") next(cfg.rel_se_target);
        else if (a=="--k") { double tmp; next(tmp); cfg.slope_k=(int)llround(tmp); }
        else if (a=="--csv") nexts(cfg.csv);
    }
    if (cfg.bc!="periodic" && cfg.bc!="reflecting"){
        cerr<<"--bc must be periodic or reflecting\n"; return 1;
    }

    double F_use = clamp_force_for_width(cfg.F_input, cfg.w, cfg.c_small_force); // [web:15]
    cerr<<"BC="<<cfg.bc<<" alpha="<<cfg.alpha<<" w="<<cfg.w<<" F_input="<<cfg.F_input
        <<" -> F_used="<<F_use<<"\n";

    // times
    vector<double> times(cfg.t_points);
    double a = std::log10(cfg.t_min), b=std::log10(cfg.t_max), d=(b-a)/(cfg.t_points-1);
    for (int i=0;i<cfg.t_points;++i) times[i] = std::pow(10.0, a + d*i);

    ofstream out(cfg.csv);
    out<<"bc,w,F_used,t,N_traj,avg_x,C_sim,se_C_sim,alpha_fit,steps_max,steps_mean\n";

    ProbYZ pyz = probs_yz_unbiased();

    // For local slope, we need avg_x per t; we will compute in two passes:
    vector<double> avgX(cfg.t_points, 0.0);

    // First pass: compute averages with adaptive N
    for (int i=0;i<cfg.t_points;++i){
        double t = times[i];
        // effective small-force drift on x (same as validated sims)
        double F_eff = std::pow(F_use, cfg.alpha) * std::pow((double)cfg.w, -2.0*(1.0 - cfg.alpha));
        ProbX px = probs_x(F_eff);

        // adaptive batches
        int N = cfg.N0; int batches=0;
        vector<double> xs; xs.reserve(N);
        uint64_t global_seed = cfg.base_seed ^ (uint64_t)cfg.w ^ (uint64_t)i;
        while (true){
            // launch threads
            unsigned nt = std::thread::hardware_concurrency(); if (!nt) nt=1;
            int per = N / (int)nt;
            vector<thread> pool; vector<double> locals(nt,0.0);
            vector<uint64_t> smax(nt,0); vector<uint64_t> ssum(nt,0);
            for (unsigned th=0; th<nt; ++th){
                uint64_t s = global_seed; for (unsigned k=0;k<=th;++k) s=splitmix64(s);
                pool.emplace_back([&,th,s](){
                    std::mt19937_64 rng(s);
                    double loc=0.0; uint64_t lmax=0, lsum=0;
                    int i0 = th*per, i1 = (th==nt-1? N : i0+per);
                    for (int j=i0;j<i1;++j){
                        uint64_t so=0;
                        int64_t xd = simulate_traj(cfg.w, cfg.bc, px, pyz, cfg.alpha, t, rng, so);
                        loc += (double)xd; lsum += so; if (so>lmax) lmax=so;
                    }
                    locals[th]=loc; smax[th]=lmax; ssum[th]=lsum;
                });
            }
            for (auto& th: pool) th.join();
            double x_sum=0.0; uint64_t steps_max=0; unsigned __int128 steps_sum=0;
            for (unsigned th=0; th<locals.size(); ++th){
                x_sum += locals[th];
                if (smax[th]>steps_max) steps_max=smax[th];
                steps_sum += ssum[th];
            }
            double avg_x_batch = x_sum / (double)N;
            double smean = (double)steps_sum / (double)N;
            xs.push_back(avg_x_batch);

            // compute C_sim and SE over batches (simple batching)
            vector<double> Cb; Cb.reserve(xs.size());
            for (double xb : xs){
                double C = xb * std::pow(t, -cfg.alpha)
                              * std::pow(F_use, -cfg.alpha)
                              * std::pow((double)cfg.w, 2.0*(1.0 - cfg.alpha));
                Cb.push_back(C);
            }
            double m = std::accumulate(Cb.begin(),Cb.end(),0.0)/Cb.size();
            double v=0.0; for (double c: Cb){ double d=c-m; v+=d*d; }
            v /= (Cb.size()>1? Cb.size()-1:1);
            double se = std::sqrt(v / (double)Cb.size());

            // write intermediate (last batch only kept in file); keep compact output per t
            avgX[i] = std::accumulate(xs.begin(),xs.end(),0.0)/xs.size();
            double C_sim = avgX[i] * std::pow(t, -cfg.alpha)
                                    * std::pow(F_use, -cfg.alpha)
                                    * std::pow((double)cfg.w, 2.0*(1.0 - cfg.alpha));

            out<<cfg.bc<<","<<cfg.w<<","<<std::setprecision(17)<<F_use<<","<<t<<","
               <<(int)(xs.size()*N)<<","<<std::setprecision(17)<<avgX[i]<<","
               <<std::setprecision(17)<<C_sim<<","<<std::setprecision(17)<<se<<","
               <<0.0<<"," // placeholder for alpha_fit (second pass)
               <<steps_max<<","<<std::setprecision(17)<<smean<<"\n";

            // stopping rule on relative SE
            if (m>0 && se/m <= cfg.rel_se_target) break;
            if (++batches >= cfg.max_batches) break;
            N = std::min(cfg.Nmax, 2*N);
        }
    }

    // Reopen and add alpha_fit per row (simplify by writing a new file)
    out.close();
    // read back rows
    ifstream in(cfg.csv);
    string header; getline(in, header);
    struct Row{ string bc; int w; double F,t; long long Ntraj; double avgx,C,SE,alpha_fit; uint64_t smax; double smean; };
    vector<Row> rows;
    string line;
    while (getline(in,line)){
        if (line.empty()) continue;
        stringstream ss(line);
        string bc; string wS,tS,FS,Ns,avgS,CS,SEs,afS,smaxS,smeanS;
        getline(ss, bc, ',');
        getline(ss, wS, ','); getline(ss, FS, ','); getline(ss, tS, ',');
        getline(ss, Ns, ','); getline(ss, avgS, ','); getline(ss, CS, ','); getline(ss, SEs, ',');
        getline(ss, afS, ','); getline(ss, smaxS, ','); getline(ss, smeanS, ',');
        Row r;
        r.bc=bc; r.w=stoi(wS); r.F=stod(FS); r.t=stod(tS);
        r.Ntraj=stoll(Ns); r.avgx=stod(avgS); r.C=stod(CS); r.SE=stod(SEs);
        r.alpha_fit=0.0; r.smax=(uint64_t)stoull(smaxS); r.smean=stod(smeanS);
        rows.push_back(r);
    }
    in.close();

    // compute alpha_fit(t) from local window using avgX
    vector<double> alpha_local(times.size(), NAN);
    for (int i=0;i<(int)times.size(); ++i){
        alpha_local[i] = local_loglog_slope(times, avgX, i, cfg.slope_k); // [web:223]
    }

    // write final with alpha_fit
    ofstream out2(cfg.csv);
    out2<<"bc,w,F_used,t,N_traj,avg_x,C_sim,se_C_sim,alpha_fit,steps_max,steps_mean\n";
    size_t idx=0;
    for (int i=0;i<(int)times.size(); ++i){
        // find row with same t (one per t)
        Row& r = rows[idx++];
        r.alpha_fit = alpha_local[i];
        out2<<r.bc<<","<<r.w<<","<<std::setprecision(17)<<r.F<<","<<std::setprecision(17)<<r.t<<","
            <<r.Ntraj<<","<<std::setprecision(17)<<r.avgx<<","<<std::setprecision(17)<<r.C<<","
            <<std::setprecision(17)<<r.SE<<","<<std::setprecision(17)<<r.alpha_fit<<","
            <<r.smax<<","<<std::setprecision(17)<<r.smean<<"\n";
    }
    out2.close();

    cerr<<"\nWrote "<<cfg.csv<<". Use Python to plot C_sim(t) with plateau band and alpha_fit(t) to extract t*.\n";
    return 0;
}
