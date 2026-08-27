// =====================================================================
//  qtm_omega_scan.cpp
//
//  First moment <x_par(t)> versus cross-section Omega, for a 2D quenched
//  trap model channel with a SINGLE anisotropic pair of lattice constants
//      a = spacing along the open longitudinal axis (x)
//      b = spacing along the confined transverse axis (y)
//  In d = 2 the cross-section is one axis, so Omega = w (number of
//  transverse sites). The scan variable is w.
//
//  Nothing about the transport law is assumed. The bare force enters only
//  through detailed balance, p(e_nu) ~ exp(F . e_nu / 2); tau_r is quenched;
//  F^alpha, Omega^-(1-alpha) and Lambda are MEASURED.
//
//  Prediction under test (paper Eq. 35, generalised):
//      <x_par> ~ (D0 F)^alpha (a/Omega)^(1-alpha) t^alpha / [A Gamma^2(1+alpha)]
//  with   D0 = a^2/4                      (model 'iso',   rates distance-blind)
//         D0 = a^2 b^2 / [2(a^2+b^2)]      (model 'invd2', rates ~ 1/d^2)
//  The claim is that a and b move only the PREFACTOR and leave the
//  Omega-exponent -(1-alpha) untouched. That is what the scan tests.
//
//  Build:
//    g++ -std=c++17 -O3 -march=native -fopenmp -DTCBITS=12 -o qtm_omega qtm_omega_scan.cpp
//  Run:
//    ./qtm_omega --a 1 --b 2 --w 2,3,4,6,8,12,16 --N 100000 --auto 1 --out omega.csv
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
#include <numeric>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef TCBITS
#define TCBITS 12
#endif

// ---------------------------------------------------------------------
static inline uint64_t mix64(uint64_t z) noexcept {
    z ^= z >> 33; z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 33; z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 33; return z;
}

struct Rng {                                        // xoshiro256++
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
    inline double u01() { return (double)((next() >> 11) + 1) * 0x1.0p-53; }   // (0,1]
};

// ---------------------------------------------------------------------
//  Quenched tau_r, memoised. tau_r stays a deterministic function of
//  (disorder seed, x, y), so a return to a site reproduces it exactly;
//  the cache only avoids recomputing pow(). Disorder seed is folded into
//  the key, so stale entries from earlier trajectories cannot match and
//  no reset is needed.
// ---------------------------------------------------------------------
struct alignas(64) TauCache {
    static constexpr size_t SIZE = size_t(1) << TCBITS;
    static constexpr size_t MASK = SIZE - 1;
    struct Slot { uint64_t key; double tau; };
    std::vector<Slot> s;
    uint64_t hits = 0, misses = 0;
    TauCache() : s(SIZE, Slot{0ULL, 0.0}) {}
};

static inline double tau_of_site(TauCache& C, int64_t x, int32_t y, uint64_t dseed,
                                 double neg_inv_alpha, double scale) noexcept {
    const uint64_t site = (uint64_t)(x * 0x9E3779B97F4A7C15ULL)
                        + (uint64_t)(uint32_t)y * 0xC2B2AE3D27D4EB4FULL;
    const uint64_t h = mix64(site + dseed);
    TauCache::Slot& sl = C.s[(size_t)h & TauCache::MASK];
    if (sl.key == h) { ++C.hits; return sl.tau; }
    const double u = (double)((h >> 11) + 1) * 0x1.0p-53;
    const double v = scale * std::pow(u, neg_inv_alpha);
    sl.key = h; sl.tau = v; ++C.misses;
    return v;
}

// ---------------------------------------------------------------------
enum RateModel { RM_ISO = 0, RM_INVD2 = 1 };
enum Boundary  { BC_PBC = 0, BC_REFLECT = 1 };

struct Params {
    double a = 1.0, b = 2.0;
    int    w = 5;                       // Omega in d = 2
    double F = 0.005;
    double alpha = 0.3, A = 1.0;
    RateModel model = RM_INVD2;
    Boundary  bc    = BC_PBC;
};

static void weights(const Params& P, double& wxp, double& wxm, double& wy) {
    const double e = std::exp(P.F * P.a / 2.0);                 // F . e_x = F * a
    if (P.model == RM_ISO) { wxp = e;             wxm = 1.0/e;             wy = 1.0; }
    else                   { wxp = e/(P.a*P.a);   wxm = 1.0/(e*P.a*P.a);   wy = 1.0/(P.b*P.b); }
}

// ---------------------------------------------------------------------
//  Analytic reference (Eqs. 6, 10, 33, 34). Comparison only -- never fed back.
// ---------------------------------------------------------------------
static double polylog_neg(double alpha, double z) {              // Li_{-alpha}(z)
    if (z <= 0.0) return 0.0;
    double s = 0.0, zj = z;
    for (long j = 1; j < 400000000L; ++j) {
        const double term = std::pow((double)j, alpha) * zj;
        s += term; zj *= z;
        if (j > 32 && term < 1e-16 * s) break;
    }
    return s;
}

struct Theory { double v_par, D_par, D_perp, eps, Lambda, N_star; };

static Theory theory(const Params& P) {
    double wxp, wxm, wy; weights(P, wxp, wxm, wy);
    const double Z = wxp + wxm + 2.0*wy;
    const double pxp = wxp/Z, pxm = wxm/Z, py = wy/Z;
    Theory th{};
    th.v_par  = P.a * (pxp - pxm);                               // Eq. (19)
    th.D_par  = 0.5 * P.a * P.a * (pxp + pxm);                   // Eq. (19)
    th.D_perp = P.b * P.b * py;                                  // Eq. (19)
    th.eps    = (double)P.w * th.v_par / P.a;                    // Eq. (33)
    const double Q0 = 1.0 - th.eps;
    th.Lambda = (Q0 > 0.0 && Q0 < 1.0)
              ? (1.0-Q0)*(1.0-Q0)*polylog_neg(P.alpha, Q0)/Q0    // Eq. (6)
              : NAN;
    th.N_star = 2.0 * th.D_par / (th.v_par * th.v_par);          // drift beats diffusion
    return th;
}
static double x_eq10(const Params& P, const Theory& th, double T) {
    return th.v_par * std::pow(T, P.alpha) / (P.A * std::tgamma(1.0+P.alpha) * th.Lambda);
}
static double x_eq34(const Params& P, const Theory& th, double T) {
    const double G = std::tgamma(1.0+P.alpha);
    return th.v_par * std::pow(th.eps, P.alpha-1.0) * std::pow(T,P.alpha) / (P.A*G*G);
}
// bare diffusion coefficient at zero force, for the record
static double D0_of(const Params& P) {
    Params Q = P; Q.F = 0.0; return theory(Q).D_par;
}

// ---------------------------------------------------------------------
//  One walk, recorded at every T in the ascending list (single pass).
// ---------------------------------------------------------------------
static void run_traj(const Params& P, uint64_t traj_seed, TauCache& C,
                     double c1, double c2, double c3,
                     double neg_inv_alpha, double tau_scale, uint64_t NCAP,
                     const std::vector<double>& Tl,
                     std::vector<int64_t>& xrec, std::vector<uint64_t>& Nrec, bool& capped)
{
    Rng rng; rng.seed(mix64(traj_seed ^ 0x5851F42D4C957F2DULL));
    const uint64_t dseed = mix64(traj_seed ^ 0xA24BAED4963EE407ULL);
    const int K = (int)Tl.size(), w = P.w;

    int64_t x = 0; int32_t y = 0; uint64_t N = 0; capped = false;
    double t = tau_of_site(C, 0, 0, dseed, neg_inv_alpha, tau_scale);

    for (int ti = 0; ti < K; ) {
        if (t >= Tl[ti]) { xrec[ti] = x; Nrec[ti] = N; ++ti; continue; }
        if (N >= NCAP) { capped = true;
                         for (; ti < K; ++ti) { xrec[ti] = x; Nrec[ti] = N; }
                         break; }
        const double r = rng.u01();
        if      (r < c1) { ++x; }
        else if (r < c2) { --x; }
        else if (r < c3) { if (P.bc == BC_PBC) y = (y+1 == w) ? 0 : y+1; else if (y+1 < w) ++y; }
        else             { if (P.bc == BC_PBC) y = (y == 0) ? w-1 : y-1; else if (y > 0)   --y; }
        ++N;
        t += tau_of_site(C, x, y, dseed, neg_inv_alpha, tau_scale);
    }
}

// Diagnostic walk: measures 1-Q0 = distinct/N and Lambda = S_alpha/N.
struct DiagOut { uint64_t N, distinct; double S_alpha; };
static DiagOut run_traj_diag(const Params& P, uint64_t traj_seed, TauCache& C,
                             double c1, double c2, double c3,
                             double neg_inv_alpha, double tau_scale, double T, uint64_t NCAP)
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
        const double r = rng.u01();
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
//  Heavy-tail-aware statistics. <x> is an average over a broad,
//  non-self-averaging distribution, so the plain standard error can
//  understate the uncertainty. We add a bootstrap CI on the mean and the
//  share of <x> carried by the top 1% of walkers as a reliability flag.
// ---------------------------------------------------------------------
struct Stats { double mean, se, ci_lo, ci_hi, median, top1; };

static Stats summarise(std::vector<double>& v, uint64_t seed, int nboot) {
    Stats S{};
    const size_t n = v.size();
    if (!n) return S;
    long double s = 0.0L, ss = 0.0L;
    for (double x : v) { s += x; ss += (long double)x * x; }
    S.mean = (double)(s / (long double)n);
    const double var = (double)(ss/(long double)n) - S.mean*S.mean;
    S.se = std::sqrt(std::max(0.0, var) / (double)n);

    std::vector<double> srt(v);
    std::sort(srt.begin(), srt.end());
    S.median = (n & 1) ? srt[n/2] : 0.5*(srt[n/2-1] + srt[n/2]);
    const size_t k = std::max<size_t>(1, n/100);
    double tail = 0.0; for (size_t i = n-k; i < n; ++i) tail += srt[i];
    S.top1 = (s != 0.0L) ? tail / (double)s : NAN;

    if (nboot > 1) {
        std::vector<double> bm((size_t)nboot);
        Rng r; r.seed(seed);
        for (int t = 0; t < nboot; ++t) {
            long double acc = 0.0L;
            for (size_t i = 0; i < n; ++i) acc += v[(size_t)(r.next() % n)];
            bm[(size_t)t] = (double)(acc / (long double)n);
        }
        std::sort(bm.begin(), bm.end());
        S.ci_lo = bm[(size_t)(0.025*nboot)];
        S.ci_hi = bm[(size_t)std::min<double>(nboot-1, 0.975*nboot)];
    } else { S.ci_lo = S.mean - 1.96*S.se; S.ci_hi = S.mean + 1.96*S.se; }
    return S;
}

// ---------------------------------------------------------------------
static std::vector<double> parse_list(const std::string& s) {
    std::vector<double> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::atof(tok.c_str()));
    return v;
}
static bool has_arg(int c, char** v, const char* k) {
    for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], k)) return true; return false;
}
static std::string arg_of(int c, char** v, const char* k, const char* d) {
    for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], k)) return v[i+1]; return d;
}
static double   arg_d(int c, char** v, const char* k, const char* d){ return std::atof(arg_of(c,v,k,d).c_str()); }
static uint64_t arg_u(int c, char** v, const char* k, const char* d){ return strtoull(arg_of(c,v,k,d).c_str(),nullptr,10); }

int main(int argc, char** argv) {
    Params P;
    P.a     = arg_d(argc,argv,"--a","1.0");
    P.b     = arg_d(argc,argv,"--b","2.0");
    P.alpha = arg_d(argc,argv,"--alpha","0.3");
    P.A     = arg_d(argc,argv,"--amp","1.0");
    const std::string ms  = arg_of(argc,argv,"--model","invd2");
    const std::string bcs = arg_of(argc,argv,"--bc","pbc");
    P.model = (ms=="iso") ? RM_ISO : RM_INVD2;
    P.bc    = (bcs=="reflect") ? BC_REFLECT : BC_PBC;

    std::vector<double> wl = parse_list(arg_of(argc,argv,"--w","2,3,4,6,8,12,16"));
    std::sort(wl.begin(), wl.end());
    wl.erase(std::unique(wl.begin(), wl.end()), wl.end());

    // number of walkers: --N, or --ntraj
    uint64_t NW = arg_u(argc,argv,"--N","0");
    if (!NW) NW = arg_u(argc,argv,"--ntraj","100000");

    const double  EPSMAX = arg_d(argc,argv,"--epsmax","0.03");
    const double  SAFETY = arg_d(argc,argv,"--safety","10.0");
    const uint64_t NDIAG = arg_u(argc,argv,"--diag","0");
    const uint64_t NCAP  = arg_u(argc,argv,"--ncap","2000000000");
    const uint64_t MSEED = arg_u(argc,argv,"--seed","20260827");
    const bool     CRN   = arg_u(argc,argv,"--crn","1") != 0;
    const int      NBOOT = (int)arg_u(argc,argv,"--nboot","400");
    const std::string out = arg_of(argc,argv,"--out","omega_scan.csv");

    const double neg_inv_alpha = -1.0 / P.alpha;
    const double A_raw = std::tgamma(1.0 - P.alpha);
    const double tau_scale = std::pow(P.A / A_raw, 1.0 / P.alpha);
    const double G1 = std::tgamma(1.0 + P.alpha);

    // -----------------------------------------------------------------
    //  Operating point.
    //  Both validity conditions bind at the LARGEST Omega:
    //    eps(w) = w v/a grows linearly in w, while N* = 2D/v^2 is w-independent
    //    and <N>(w) ~ eps^-(1-alpha) shrinks with w.
    //  So fix F from eps(w_max) = epsmax, then fix T from <N>(w_max) = safety*N*.
    //  Cost is then dominated by the SMALLEST w, and scales as w_max^2/epsmax^2.
    // -----------------------------------------------------------------
    std::vector<double> Tl;
    const bool AUTO = !has_arg(argc,argv,"--F") || arg_u(argc,argv,"--auto","0") != 0;
    if (AUTO) {
        P.w = (int)std::llround(wl.back());
        const double D0 = D0_of(P);
        P.F = EPSMAX * P.a / ((double)P.w * D0);            // first guess
        for (int it = 0; it < 40; ++it) {                   // refine against exact v(F)
            const Theory t = theory(P);
            P.F *= EPSMAX / t.eps;
        }
        const Theory t = theory(P);
        const double Nreq = SAFETY * t.N_star;
        const double Tmax = std::pow(Nreq * P.A * G1 * t.Lambda, 1.0 / P.alpha);
        Tl = { Tmax/1000.0, Tmax/100.0, Tmax/10.0, Tmax };
    } else {
        P.F = arg_d(argc,argv,"--F","0.005");
        Tl  = parse_list(arg_of(argc,argv,"--T","1e14"));
        std::sort(Tl.begin(), Tl.end());
    }
    const int K = (int)Tl.size();

#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
#else
    const int nthr = 1;
#endif
    const double D0 = D0_of(P);
    std::printf("# ===== <x> vs Omega, anisotropic 2D QTM channel =====\n");
    std::printf("# a=%g  b=%g  (a/b=%g)  model=%s  bc=%s  alpha=%g  A=%g\n",
                P.a, P.b, P.a/P.b, ms.c_str(), bcs.c_str(), P.alpha, P.A);
    std::printf("# D0 = %.6f   (iso: a^2/4 = %.6f ; invd2: a^2b^2/2(a^2+b^2) = %.6f)\n",
                D0, P.a*P.a/4.0, P.a*P.a*P.b*P.b/(2.0*(P.a*P.a+P.b*P.b)));
    std::printf("# F = %.6g  %s   walkers N = %llu   crn=%d   threads=%d\n",
                P.F, (AUTO?"(auto, from eps(Omega_max))":"(user)"), (unsigned long long)NW,
                (int)CRN, nthr);
    std::printf("# T =");
    for (double T : Tl) std::printf(" %.3e", T);
    std::printf("   %s\n", AUTO ? "(auto; smaller T are free and give the convergence check)" : "(user)");
    std::printf("# tau = %.6f * u^(-1/alpha)   [A_raw = Gamma(1-alpha) = %.6f]\n", tau_scale, A_raw);
    std::printf("# prediction: <x> ~ (D0 F)^a (a/Omega)^(1-a) T^a / [A Gamma^2(1+a)]"
                "  ->  slope %.3f on log-log vs Omega\n", P.alpha - 1.0);
    std::printf("# valid row requires eps <= %.3g AND <N>/N* >= %.3g\n", EPSMAX, SAFETY);
    std::printf("# %-5s %-9s | %-11s %-10s %-9s | %-11s %-9s %-8s %-5s\n",
                "Omega","T","<x>_sim","stderr","<N>","<x>_Eq10","eps","<N>/N*","ok");

    std::ofstream csv(out);
    csv << "model,bc,alpha,A,a,b,D0,w,Omega,F,T,N_walkers,"
           "x_mean,x_stderr,x_ci_lo,x_ci_hi,x_median,top1pct_share,"
           "N_steps_mean,frac_stuck,capped,"
           "v_par,D_par,D_perp,eps_theory,Lambda_theory,N_star,N_over_Nstar,valid,"
           "x_theory_eq10,x_theory_eq34,eps_meas,Lambda_meas,ndiag,cache_hit_rate\n";

    // largest Omega is cheapest -> run it first so results appear early
    std::vector<double> order(wl.rbegin(), wl.rend());
    double elapsed_tot = 0.0;

    for (double wd : order) {
        P.w = (int)std::llround(wd);
        const auto t0 = std::chrono::high_resolution_clock::now();

        double wxp,wxm,wy; weights(P,wxp,wxm,wy);
        const double Z = wxp+wxm+2.0*wy;
        const double c1 = wxp/Z, c2 = c1+wxm/Z, c3 = c2+wy/Z;
        const Theory th = theory(P);

        // CRN: seed depends only on walker index, so every Omega sees the
        // same random stream and the same disorder -- differences resolve better.
        const uint64_t pid = CRN ? MSEED
                                 : mix64(MSEED ^ mix64((uint64_t)P.w * 0xC2B2AE3D27D4EB4FULL));

        std::vector<double>   xall((size_t)K * NW, 0.0);
        std::vector<long double> sN((size_t)K, 0.0L);
        std::vector<uint64_t> stuck((size_t)K, 0);
        uint64_t capped_n = 0, chits = 0, cmiss = 0;

        #pragma omp parallel
        {
            TauCache C;
            std::vector<int64_t>  xr((size_t)K,0);
            std::vector<uint64_t> Nr((size_t)K,0);
            std::vector<long double> lsN((size_t)K,0.0L);
            std::vector<uint64_t> lstuck((size_t)K,0);
            uint64_t lcap = 0;

            #pragma omp for schedule(dynamic,32) nowait
            for (long long n = 0; n < (long long)NW; ++n) {
                bool cap=false;
                run_traj(P, mix64(pid ^ (uint64_t)(n+1)*0x9E3779B97F4A7C15ULL), C,
                         c1,c2,c3, neg_inv_alpha, tau_scale, NCAP, Tl, xr, Nr, cap);
                if (cap) ++lcap;
                for (int k=0;k<K;++k) {
                    xall[(size_t)k*NW + (size_t)n] = (double)xr[k] * P.a;   // physical length
                    lsN[k] += (long double)Nr[k];
                    if (Nr[k]==0) ++lstuck[k];
                }
            }
            #pragma omp critical
            {
                for (int k=0;k<K;++k){ sN[k]+=lsN[k]; stuck[k]+=lstuck[k]; }
                capped_n += lcap; chits += C.hits; cmiss += C.misses;
            }
        }

        double eps_meas = NAN, lam_meas = NAN;
        if (NDIAG > 0) {
            long double sD=0.0L,sS=0.0L,sNd=0.0L;
            #pragma omp parallel reduction(+:sD,sS,sNd)
            {
                TauCache C;
                #pragma omp for schedule(dynamic,8)
                for (long long n=0;n<(long long)NDIAG;++n){
                    const DiagOut d = run_traj_diag(P, mix64(pid ^ 0xBEEFULL ^ (uint64_t)(n+1)*0x9E3779B97F4A7C15ULL),
                                                    C,c1,c2,c3,neg_inv_alpha,tau_scale,Tl.back(),NCAP);
                    if (d.N>0){ sD+=(long double)d.distinct; sS+=(long double)d.S_alpha; sNd+=(long double)d.N; }
                }
            }
            if (sNd>0){ eps_meas=(double)(sD/sNd); lam_meas=(double)(sS/sNd); }
        }

        const double secs = std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now()-t0).count();
        elapsed_tot += secs;
        const double hit = (chits+cmiss) ? (double)chits/(double)(chits+cmiss) : 0.0;

        for (int k=0;k<K;++k) {
            std::vector<double> col(xall.begin()+(size_t)k*NW, xall.begin()+(size_t)(k+1)*NW);
            const Stats S = summarise(col, mix64(pid ^ (uint64_t)(k+7)*0xABCDEF01ULL), NBOOT);
            const double Nm  = (double)(sN[k]/(long double)NW);
            const double rat = Nm / th.N_star;
            const bool   ok  = (th.eps <= EPSMAX*1.05) && (rat >= SAFETY*0.9) && (th.eps < 1.0);

            std::printf("  %-5d %-9.2e | %-11.4e %-10.2e %-9.3e | %-11.4e %-9.2e %-8.1f %-5s\n",
                P.w, Tl[k], S.mean, S.se, Nm, x_eq10(P,th,Tl[k]), th.eps, rat, ok?"yes":"NO");

            csv << ms << "," << bcs << "," << P.alpha << "," << P.A << ","
                << P.a << "," << P.b << "," << D0 << "," << P.w << "," << P.w << ","
                << P.F << "," << Tl[k] << "," << NW << ","
                << S.mean << "," << S.se << "," << S.ci_lo << "," << S.ci_hi << ","
                << S.median << "," << S.top1 << ","
                << Nm << "," << (double)stuck[k]/(double)NW << "," << capped_n << ","
                << th.v_par << "," << th.D_par << "," << th.D_perp << "," << th.eps << ","
                << th.Lambda << "," << th.N_star << "," << rat << "," << (ok?1:0) << ","
                << x_eq10(P,th,Tl[k]) << "," << x_eq34(P,th,Tl[k]) << ","
                << (k==K-1?eps_meas:NAN) << "," << (k==K-1?lam_meas:NAN) << ","
                << (k==K-1?NDIAG:0ULL) << "," << hit << "\n";
        }
        std::printf("      [Omega=%d done in %.1fs, cache hit %.2f%%, total %.1fs]\n",
                    P.w, secs, 100.0*hit, elapsed_tot);
        std::fflush(stdout); csv.flush();
    }
    std::printf("# done -> %s   (%.1f s total)\n", out.c_str(), elapsed_tot);
    return 0;
}