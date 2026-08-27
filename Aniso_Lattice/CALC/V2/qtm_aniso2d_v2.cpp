// =====================================================================
//  qtm_aniso2d_v2.cpp   --  optimized engine
//
//  Same physics as v1 (nothing about the transport law is assumed), plus:
//    [1] direct-mapped tau cache      -> removes ~95% of the pow() calls
//    [2] multi-T recording in ONE walk -> a full T-convergence scan free
//    [3] common random numbers (CRN)  -> ratios between geometries get
//                                        ~10x more precise for free
//    [4] regime diagnostics           -> tells you if the point is valid
//
//  Build:
//    g++ -std=c++17 -O3 -march=native -fopenmp -o qtm2 qtm_aniso2d_v2.cpp
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint64_t mix64(uint64_t z) noexcept {
    z ^= z >> 33; z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 33; z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 33; return z;
}

struct Rng {                                   // xoshiro256++
    uint64_t s[4];
    void seed(uint64_t k) {
        for (int i = 0; i < 4; ++i) { k += 0x9E3779B97F4A7C15ULL; s[i] = mix64(k); }
        if (!(s[0]|s[1]|s[2]|s[3])) s[0] = 1;
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
};

// ---------------------------------------------------------------------
//  [1] Direct-mapped tau cache.
//  tau_r stays a deterministic function of (disorder seed, x, y): the
//  cache only memoises it, so quenched disorder is exact, not approximated.
//  The disorder seed is folded into the key, so no reset is needed between
//  trajectories -- stale entries simply cannot match.
// ---------------------------------------------------------------------
struct alignas(64) TauCache {
    #ifndef TCBITS
#define TCBITS 13
#endif
    static constexpr int    BITS = TCBITS;
    static constexpr size_t SIZE = size_t(1) << BITS;
    static constexpr size_t MASK = SIZE - 1;
    struct Slot { uint64_t key; double tau; };      // 16 B: one cache line fetch
    std::vector<Slot> s;
    uint64_t hits = 0, misses = 0;
    TauCache() : s(SIZE, Slot{0ULL, 0.0}) {}
};

static inline double tau_of_site(TauCache& C, int64_t x, int32_t y, uint64_t dseed,
                                 double neg_inv_alpha, double scale) noexcept {
    const uint64_t site = (uint64_t)(x * 0x9E3779B97F4A7C15ULL) + (uint64_t)(uint32_t)y * 0xC2B2AE3D27D4EB4FULL;
    const uint64_t h    = mix64(site + dseed);             // single finalizer
    TauCache::Slot& sl  = C.s[(size_t)h & TauCache::MASK];
    if (sl.key == h) { ++C.hits; return sl.tau; }
    const double u = (double)((h >> 11) + 1) * 0x1.0p-53;   // (0,1]
    const double v = scale * std::pow(u, neg_inv_alpha);
    sl.key = h; sl.tau = v; ++C.misses;
    return v;
}

// ---------------------------------------------------------------------
enum RateModel { RM_ISO = 0, RM_INVD2 = 1 };
enum Boundary  { BC_PBC = 0, BC_REFLECT = 1 };

struct Params {
    double a = 1.0, b = 1.0;
    int    w = 5;
    double F = 0.01;
    double alpha = 0.3;
    double A = 1.0;
    RateModel model = RM_INVD2;
    Boundary  bc    = BC_PBC;
};

static void weights(const Params& P, double& wxp, double& wxm, double& wy) {
    const double e = std::exp(P.F * P.a / 2.0);         // F . e_x = F * a
    if (P.model == RM_ISO) { wxp = e;             wxm = 1.0 / e;             wy = 1.0; }
    else                   { wxp = e/(P.a*P.a);   wxm = 1.0/(e*P.a*P.a);     wy = 1.0/(P.b*P.b); }
}

// ---------------------------------------------------------------------
//  Analytic reference (Eqs. 6, 10, 33, 34). Comparison only.
// ---------------------------------------------------------------------
static double polylog_neg(double alpha, double z) {
    if (z <= 0.0) return 0.0;
    double s = 0.0, zj = z;
    for (long j = 1; j < 200000000L; ++j) {
        const double term = std::pow((double)j, alpha) * zj;
        s += term; zj *= z;
        if (j > 32 && term < 1e-16 * s) break;
    }
    return s;
}

struct Theory { double v_par, D_par, D_perp, eps, Lambda, N_star; };

static Theory theory(const Params& P) {
    double wxp, wxm, wy; weights(P, wxp, wxm, wy);
    const double Z = wxp + wxm + 2.0 * wy;
    const double pxp = wxp/Z, pxm = wxm/Z, py = wy/Z;
    Theory th{};
    th.v_par  = P.a * (pxp - pxm);
    th.D_par  = 0.5 * P.a * P.a * (pxp + pxm);
    th.D_perp = P.b * P.b * py;
    th.eps    = (double)P.w * th.v_par / P.a;                 // Eq. (33)
    const double Q0 = 1.0 - th.eps;
    th.Lambda = (Q0 > 0.0 && Q0 < 1.0)
              ? (1.0-Q0)*(1.0-Q0)*polylog_neg(P.alpha, Q0)/Q0 // Eq. (6)
              : NAN;
    // drift must dominate diffusion before Eq.(33) holds:
    th.N_star = 2.0 * th.D_par / (th.v_par * th.v_par);
    return th;
}
static double x_eq10(const Params& P, const Theory& th, double T) {
    return th.v_par * std::pow(T, P.alpha) / (P.A * std::tgamma(1.0+P.alpha) * th.Lambda);
}
static double x_eq34(const Params& P, const Theory& th, double T) {
    const double G1 = std::tgamma(1.0 + P.alpha);
    return th.v_par * std::pow(th.eps, P.alpha-1.0) * std::pow(T, P.alpha) / (P.A * G1 * G1);
}

// ---------------------------------------------------------------------
//  [2] One walk, recorded at every T in an ascending list.
//      The clock is monotone, so x(T) for all T costs one pass to max(T).
// ---------------------------------------------------------------------
static void run_traj(const Params& P, uint64_t traj_seed, TauCache& C,
                     double c1, double c2, double c3,
                     double neg_inv_alpha, double tau_scale, uint64_t NCAP,
                     const std::vector<double>& Tl,
                     std::vector<int64_t>& xrec, std::vector<uint64_t>& Nrec,
                     bool& capped)
{
    Rng rng; rng.seed(mix64(traj_seed ^ 0x5851F42D4C957F2DULL));
    const uint64_t dseed = mix64(traj_seed ^ 0xA24BAED4963EE407ULL);
    const int K = (int)Tl.size(), w = P.w;

    int64_t x = 0; int32_t y = 0; uint64_t N = 0; capped = false;
    double t = tau_of_site(C, 0, 0, dseed, neg_inv_alpha, tau_scale);

    for (int ti = 0; ti < K; ) {
        if (t >= Tl[ti]) { xrec[ti] = x; Nrec[ti] = N; ++ti; continue; }
        if (N >= NCAP)   { capped = true;
                           for (; ti < K; ++ti) { xrec[ti] = x; Nrec[ti] = N; }
                           break; }
        const double r = (double)((rng.next() >> 11) + 1) * 0x1.0p-53;
        if      (r < c1) { ++x; }
        else if (r < c2) { --x; }
        else if (r < c3) { if (P.bc == BC_PBC) y = (y+1 == w) ? 0 : y+1; else if (y+1 < w) ++y; }
        else             { if (P.bc == BC_PBC) y = (y == 0) ? w-1 : y-1; else if (y > 0)   --y; }
        ++N;
        t += tau_of_site(C, x, y, dseed, neg_inv_alpha, tau_scale);
    }
}

// Diagnostic walk: measures 1-Q0 = distinct/N and Lambda = S_alpha/N at max(T).
struct DiagOut { uint64_t N, distinct; double S_alpha; };
static DiagOut run_traj_diag(const Params& P, uint64_t traj_seed, TauCache& C,
                             double c1, double c2, double c3,
                             double neg_inv_alpha, double tau_scale,
                             double T, uint64_t NCAP)
{
    Rng rng; rng.seed(mix64(traj_seed ^ 0x5851F42D4C957F2DULL));
    const uint64_t dseed = mix64(traj_seed ^ 0xA24BAED4963EE407ULL);
    int64_t x = 0; int32_t y = 0; const int w = P.w;
    std::unordered_map<uint64_t,uint32_t> vis; vis.reserve(1u<<16);
    auto key = [](int64_t xx, int32_t yy){
        return ((uint64_t)(uint32_t)yy << 40) ^ (uint64_t)(xx + (int64_t(1)<<39)); };

    double t = tau_of_site(C, 0, 0, dseed, neg_inv_alpha, tau_scale);
    vis[key(0,0)] = 1; double S = 1.0; uint64_t N = 0;

    while (t < T && N < NCAP) {
        const double r = (double)((rng.next() >> 11) + 1) * 0x1.0p-53;
        if      (r < c1) { ++x; }
        else if (r < c2) { --x; }
        else if (r < c3) { if (P.bc == BC_PBC) y = (y+1 == w) ? 0 : y+1; else if (y+1 < w) ++y; }
        else             { if (P.bc == BC_PBC) y = (y == 0) ? w-1 : y-1; else if (y > 0)   --y; }
        ++N;
        uint32_t& n = vis[key(x,y)];
        const double n0 = (double)n; n += 1u;
        S += std::pow(n0+1.0, P.alpha) - (n0 > 0.0 ? std::pow(n0, P.alpha) : 0.0);
        t += tau_of_site(C, x, y, dseed, neg_inv_alpha, tau_scale);
    }
    return { N, (uint64_t)vis.size(), S };
}

// ---------------------------------------------------------------------
static std::vector<double> parse_list(const std::string& s) {
    std::vector<double> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::atof(tok.c_str()));
    return v;
}
static std::string arg_of(int argc, char** argv, const char* k, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], k)) return argv[i+1];
    return def;
}
static double   arg_d(int c, char** v, const char* k, const char* d){ return std::atof(arg_of(c,v,k,d).c_str()); }
static uint64_t arg_u(int c, char** v, const char* k, const char* d){ return strtoull(arg_of(c,v,k,d).c_str(),nullptr,10); }

int main(int argc, char** argv) {
    Params P;
    const std::vector<double> as = parse_list(arg_of(argc,argv,"--a","1.0"));
    const std::vector<double> bs = parse_list(arg_of(argc,argv,"--b","1.0"));
    const std::vector<double> ws = parse_list(arg_of(argc,argv,"--w","5"));
    const std::vector<double> Fs = parse_list(arg_of(argc,argv,"--F","0.01"));
    std::vector<double> Tl       = parse_list(arg_of(argc,argv,"--T","1e10,1e11,1e12"));
    std::sort(Tl.begin(), Tl.end());

    P.alpha = arg_d(argc,argv,"--alpha","0.3");
    P.A     = arg_d(argc,argv,"--amp","1.0");
    const uint64_t NTRAJ = arg_u(argc,argv,"--ntraj","100000");
    const uint64_t NDIAG = arg_u(argc,argv,"--diag","0");
    const uint64_t NCAP  = arg_u(argc,argv,"--ncap","500000000");
    const uint64_t MSEED = arg_u(argc,argv,"--seed","20260827");
    const bool     CRN   = arg_u(argc,argv,"--crn","1") != 0;
    const std::string out = arg_of(argc,argv,"--out","qtm_scan.csv");
    const std::string ms  = arg_of(argc,argv,"--model","invd2");
    const std::string bcs = arg_of(argc,argv,"--bc","pbc");
    P.model = (ms=="iso") ? RM_ISO : RM_INVD2;
    P.bc    = (bcs=="reflect") ? BC_REFLECT : BC_PBC;

    const double neg_inv_alpha = -1.0 / P.alpha;
    const double A_raw = std::tgamma(1.0 - P.alpha);
    const double tau_scale = std::pow(P.A / A_raw, 1.0 / P.alpha);
    const int K = (int)Tl.size();

#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
#else
    const int nthr = 1;
#endif
    std::printf("# QTM 2D anisotropic | model=%s bc=%s alpha=%g A=%g crn=%d threads=%d ntraj=%llu\n",
        (P.model==RM_ISO?"iso":"invd2"), (P.bc==BC_PBC?"pbc":"reflect"),
        P.alpha, P.A, (int)CRN, nthr, (unsigned long long)NTRAJ);
    std::printf("# tau = %.6f * u^(-1/alpha)  |  T list:", tau_scale);
    for (double T : Tl) std::printf(" %.1e", T);
    std::printf("\n# regime is valid when eps << 1 AND <N>/N* >> 1\n");
    std::printf("# %-5s %-5s %-3s %-7s %-8s | %-11s %-10s %-9s | %-11s | %-8s %-8s\n",
        "a","b","w","F","T","<x>_sim","stderr","<N>","<x>_Eq10","eps","<N>/N*");

    std::ofstream csv(out);
    csv << "model,bc,alpha,A,a,b,w,Omega,F,T,ntraj,x_idx_mean,x_phys_mean,x_phys_stderr,"
           "N_mean,frac_stuck,capped,v_par,D_par,D_perp,eps_theory,Lambda_theory,N_star,"
           "N_over_Nstar,x_theory_eq10,x_theory_eq34,eps_meas,Lambda_meas,ndiag,cache_hit_rate\n";

    for (double a : as) for (double b : bs) for (double wd : ws) for (double F : Fs) {
        P.a=a; P.b=b; P.w=(int)std::llround(wd); P.F=F;
        const auto t0 = std::chrono::high_resolution_clock::now();

        double wxp,wxm,wy; weights(P,wxp,wxm,wy);
        const double Z = wxp+wxm+2.0*wy;
        const double c1 = wxp/Z, c2 = c1+wxm/Z, c3 = c2+wy/Z;
        const Theory th = theory(P);

        // [3] CRN: seed depends ONLY on trajectory index, so every geometry
        //     sees the same random stream and the same disorder realisation.
        //     Differences between geometries are then far better resolved.
        const uint64_t pid = CRN ? MSEED : mix64(MSEED
              ^ mix64((uint64_t)(a*1e9)  * 0x100000001B3ULL)
              ^ mix64((uint64_t)(b*1e9)  * 0x9E3779B97F4A7C15ULL)
              ^ mix64((uint64_t)P.w      * 0xC2B2AE3D27D4EB4FULL)
              ^ mix64((uint64_t)(F*1e12) * 0xD1B54A32D192ED03ULL));

        std::vector<long double> sx(K,0.0L), sxx(K,0.0L), sN(K,0.0L);
        std::vector<uint64_t>    stuck(K,0);
        uint64_t capped_n = 0, chits = 0, cmiss = 0;

        #pragma omp parallel
        {
            TauCache C;                                  // per-thread, persists
            std::vector<int64_t>  xr(K,0);
            std::vector<uint64_t> Nr(K,0);
            std::vector<long double> lsx(K,0.0L), lsxx(K,0.0L), lsN(K,0.0L);
            std::vector<uint64_t> lstuck(K,0);
            uint64_t lcap = 0;

            #pragma omp for schedule(dynamic,32) nowait
            for (long long n = 0; n < (long long)NTRAJ; ++n) {
                bool cap=false;
                run_traj(P, mix64(pid ^ (uint64_t)(n+1)*0x9E3779B97F4A7C15ULL), C,
                         c1,c2,c3, neg_inv_alpha, tau_scale, NCAP, Tl, xr, Nr, cap);
                if (cap) ++lcap;
                for (int k=0;k<K;++k) {
                    const long double xp = (long double)xr[k]*(long double)P.a;
                    lsx[k]+=xp; lsxx[k]+=xp*xp; lsN[k]+=(long double)Nr[k];
                    if (Nr[k]==0) ++lstuck[k];
                }
            }
            #pragma omp critical
            {
                for (int k=0;k<K;++k){ sx[k]+=lsx[k]; sxx[k]+=lsxx[k]; sN[k]+=lsN[k]; stuck[k]+=lstuck[k]; }
                capped_n += lcap; chits += C.hits; cmiss += C.misses;
            }
        }

        double eps_meas = NAN, lam_meas = NAN;
        if (NDIAG > 0) {
            long double sD=0.0L, sS=0.0L, sNd=0.0L;
            #pragma omp parallel reduction(+:sD,sS,sNd)
            {
                TauCache C;
                #pragma omp for schedule(dynamic,8)
                for (long long n=0;n<(long long)NDIAG;++n){
                    const DiagOut d = run_traj_diag(P, mix64(pid ^ 0xBEEFULL ^ (uint64_t)(n+1)*0x9E3779B97F4A7C15ULL),
                                                    C, c1,c2,c3, neg_inv_alpha, tau_scale, Tl.back(), NCAP);
                    if (d.N>0){ sD+=(long double)d.distinct; sS+=(long double)d.S_alpha; sNd+=(long double)d.N; }
                }
            }
            if (sNd>0){ eps_meas=(double)(sD/sNd); lam_meas=(double)(sS/sNd); }
        }

        const double secs = std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now()-t0).count();
        const double hitrate = (chits+cmiss) ? (double)chits/(double)(chits+cmiss) : 0.0;

        for (int k=0;k<K;++k){
            const double xm  = (double)(sx[k]/(long double)NTRAJ);
            const double xv  = (double)(sxx[k]/(long double)NTRAJ) - xm*xm;
            const double xse = std::sqrt(std::max(0.0,xv)/(double)NTRAJ);
            const double Nm  = (double)(sN[k]/(long double)NTRAJ);
            const double e10 = x_eq10(P,th,Tl[k]), e34 = x_eq34(P,th,Tl[k]);
            std::printf("  %-5.3g %-5.3g %-3d %-7.4g %-8.1e | %-11.4e %-10.2e %-9.3e | %-11.4e | %-8.2e %-8.1f%s\n",
                a,b,P.w,F,Tl[k],xm,xse,Nm,e10,th.eps,Nm/th.N_star,
                (k==K-1?"":""));
            csv << (P.model==RM_ISO?"iso":"invd2") << "," << (P.bc==BC_PBC?"pbc":"reflect") << ","
                << P.alpha << "," << P.A << "," << a << "," << b << "," << P.w << "," << P.w << ","
                << F << "," << Tl[k] << "," << NTRAJ << ","
                << (xm/P.a) << "," << xm << "," << xse << "," << Nm << ","
                << (double)stuck[k]/(double)NTRAJ << "," << capped_n << ","
                << th.v_par << "," << th.D_par << "," << th.D_perp << "," << th.eps << ","
                << th.Lambda << "," << th.N_star << "," << (Nm/th.N_star) << ","
                << e10 << "," << e34 << ","
                << (k==K-1?eps_meas:NAN) << "," << (k==K-1?lam_meas:NAN) << ","
                << (k==K-1?NDIAG:0ULL) << "," << hitrate << "\n";
        }
        std::printf("      [%.1fs, cache hit %.2f%%]\n", secs, 100.0*hitrate);
        std::fflush(stdout); csv.flush();
    }
    std::printf("# done -> %s\n", out.c_str());
    return 0;
}