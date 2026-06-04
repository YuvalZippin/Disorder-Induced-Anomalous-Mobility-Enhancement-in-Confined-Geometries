// full_sim_hexagonal.cpp
// Compile: g++ -std=c++17 -O3 -pthread -o full_sim_hexagonal full_sim_hexagonal.cpp
// Run:     ./full_sim_hexagonal

#include <bits/stdc++.h>
using namespace std;

// ---------------------- Config ----------------------
struct SimConfig {
    int N_traj = 1000;                   // מספר מסלולים (הגדל לתוצאות חלקות יותר)
    double t_target = 1e17;              // זמן מטרה קבוע
    double alpha = 0.3;                  // מעריך אנומלי
    vector<double> F_inputs = {0.01, 0.02, 0.05}; // שלושת הכוחות הפיזיקליים
    vector<int> Rs = {5, 10, 15}; // רדיוסים של הצינור המשושה
    bool use_effective_drift = true;     // מיפוי הגיאומטריה לכוח אפקטיבי
    double kappa_alpha = 1.0;            // פקטור אמפליטודה
    uint64_t base_seed = 0x9e3779b97f4a7c15ULL; // גרעין רנדומיזציה בסיסי
    string csv = "results_hexagonal_t1e17.csv";
};

// ---------------------- Helpers ----------------------
inline double gamma1p(double a){ return std::tgamma(1.0 + a); } 

// SplitMix64 למחולל מספרים אקראיים לכל ת'רד
static inline uint64_t splitmix64(uint64_t &x){
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ---------------------- Probabilities ----------------------
struct CumProb { double c[8]; };

inline CumProb build_cum_probs(double F_eff){
    // פונקציית חלוקה לסריג משושה 3D
    double Z = 2.0 * std::cosh(F_eff / 2.0) + 6.0;
    CumProb cp;
    
    // ציר אורכי
    cp.c[0] = std::exp(F_eff / 2.0) / Z;                 // +x
    cp.c[1] = cp.c[0] + std::exp(-F_eff / 2.0) / Z;      // -x
    
    // מישור רוחבי (6 כיוונים)
    double p_trans = 1.0 / Z;
    for(int i = 2; i < 8; ++i) {
        cp.c[i] = cp.c[i-1] + p_trans;
    }
    cp.c[7] = 1.0; 
    
    return cp;
}

// ---------------------- CMS one-sided S_alpha ----------------------
uint64_t generate_steps(double alpha, double t, std::mt19937_64& rng){
    std::uniform_real_distribution<double> U(0.0, 1.0), Uth(0.0, M_PI);
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

// ---------------------- Trajectory ----------------------
int64_t simulate_traj_hex_reflecting(int R, const CumProb& cp, 
                                     double alpha, double t, std::mt19937_64& rng,
                                     uint64_t& steps_out){
    uint64_t steps = generate_steps(alpha, t, rng); 
    steps_out = steps; 
    
    int64_t xsum = 0;
    int q = 0, r_coord = 0; 
    
    std::uniform_real_distribution<double> U(0.0, 1.0);
    
    for (uint64_t k = 0; k < steps; ++k){
        double rand_val = U(rng);
        
        if (rand_val < cp.c[0]) { ++xsum; }
        else if (rand_val < cp.c[1]) { --xsum; }
        else {
            int dq = 0, dr = 0;
            if      (rand_val < cp.c[2]) { dq =  1; dr =  0; }
            else if (rand_val < cp.c[3]) { dq = -1; dr =  0; }
            else if (rand_val < cp.c[4]) { dq =  0; dr =  1; }
            else if (rand_val < cp.c[5]) { dq =  0; dr = -1; }
            else if (rand_val < cp.c[6]) { dq =  1; dr = -1; }
            else                         { dq = -1; dr =  1; }
            
            int nq = q + dq;
            int nr = r_coord + dr;
            
            // תנאי שפה רפלקטיבי למשושה
            if (std::max({std::abs(nq), std::abs(nr), std::abs(nq + nr)}) <= R) {
                q = nq;
                r_coord = nr;
            }
        }
    }
    return xsum;
}

// ---------------------- Worker ----------------------
struct Accum { double x_sum=0.0; uint64_t smax=0, ssum=0; };

void worker(int R, const CumProb& cp, double alpha, double t,
            uint64_t seed, int i0, int i1, Accum& out){
    std::mt19937_64 rng(seed);
    double locx = 0.0; 
    uint64_t lmax = 0, lsum = 0;
    
    for (int i = i0; i < i1; ++i){
        uint64_t so = 0;
        int64_t xd = simulate_traj_hex_reflecting(R, cp, alpha, t, rng, so);
        locx += (double)xd; 
        lsum += so; 
        if (so > lmax) lmax = so;
    }
    out.x_sum = locx; out.ssum = lsum; out.smax = lmax;
}

// ---------------------- Main ----------------------
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    SimConfig cfg;
    unsigned nt = std::thread::hardware_concurrency(); if (!nt) nt=1;

    ofstream out(cfg.csv);
    out << "R,F_input,t,average_x,C_sim,steps_max,steps_mean\n";

    cout << "Threads=" << nt << " N_traj=" << cfg.N_traj << " alpha=" << cfg.alpha
         << " t=" << cfg.t_target << "\n(Simple Hexagonal 3D; Reflecting trans; x unbounded)\n\n";

    for (int R : cfg.Rs){
        for (double F_in : cfg.F_inputs){
            
            // המרת הכוח לפי הסקיילינג הגיאומטרי של חתך המשושה
            double F_eff = cfg.use_effective_drift
                         ? cfg.kappa_alpha * std::pow(F_in, cfg.alpha) * std::pow((double)R, -2.0*(1.0 - cfg.alpha))
                         : F_in;
                         
            CumProb cp = build_cum_probs(F_eff);

            // חלוקת עבודה לת'רדים
            int per = cfg.N_traj / (int)nt;
            vector<thread> pool; vector<Accum> acc(nt);
            // מפתח רנדומיזציה שונה לכל שילוב של R ו-F
            uint64_t seed_mix = (uint64_t)R ^ (uint64_t)(F_in * 1000); 
            uint64_t seed = cfg.base_seed ^ seed_mix;
            
            for (unsigned th = 0; th < nt; ++th){
                uint64_t s = seed; for (unsigned k=0; k<=th; ++k) s = splitmix64(s);
                int i0 = th * per; int i1 = (th == nt-1 ? cfg.N_traj : i0 + per);
                pool.emplace_back(worker, R, cref(cp), cfg.alpha, cfg.t_target, s, i0, i1, ref(acc[th]));
            }
            for (auto& th : pool) th.join();

            // איסוף תוצאות
            double xsum=0.0; uint64_t smax=0; unsigned __int128 ssum128=0;
            for (auto& a_res : acc){ 
                xsum += a_res.x_sum; 
                if (a_res.smax > smax) smax = a_res.smax; 
                ssum128 += a_res.ssum; 
            }
            
            double avg_x = xsum / (double)cfg.N_traj;
            double smean = (double)ssum128 / (double)cfg.N_traj;

            // חישוב הפלטו הקבוע התיאורטי C_sim לבדיקת החוק האוניברסלי
            double C_sim = avg_x * std::pow(cfg.t_target, -cfg.alpha)
                                 * std::pow(F_in, -cfg.alpha)
                                 * std::pow((double)R, 2.0*(1.0 - cfg.alpha));

            cout << std::scientific
                 << "R=" << setw(2) << R 
                 << " | F=" << fixed << setprecision(3) << F_in << scientific 
                 << " | <x>=" << avg_x 
                 << " | C_sim=" << C_sim << "\n";

            out << R << "," << std::setprecision(17) << F_in << "," 
                << cfg.t_target << ","
                << std::setprecision(17) << avg_x << ","
                << std::setprecision(17) << C_sim << ","
                << smax << "," << std::setprecision(17) << smean << "\n";
        }
        cout << "--------------------------------------------------------\n";
    }
    
    out.close();
    cerr << "\nSimulation complete. Results written to: " << cfg.csv << "\n";
    return 0;
}