// =====================================================================
//  qtm_omega_scan_v2.cpp
//
//  V2 of qtm_omega_scan.cpp.
//
//  Goal: the first moment <x_par> as a function of the cross-section
//  Omega, measured at a FIXED observation time and for SEVERAL fixed
//  forces, so that the two exponents in
//
//      <x_par> ~ (D0 F)^alpha (a/Omega)^(1-alpha) t^alpha /[A Gamma^2(1+alpha)]
//                 \______/     \___________/
//                  F-scaling    Omega-scaling
//
//  can be read off independently:
//    * slope of log<x> vs log(Omega) at fixed F  ->  should be alpha-1
//    * ratio <x>(F2)/<x>(F1) at fixed Omega      ->  should be (F2/F1)^alpha
//
//  Differences from V1:
//    1. --F now takes a LIST of forces (default 0.001,0.005) and the code
//       loops over them; there is no auto-F mode any more, since the whole
//       point is to compare two prescribed forces.
//    2. --T defaults to a single time, 1e17. Smaller checkpoint times
//       (T/1e3, T/1e2, T/10) are recorded in the same pass for free and
//       serve as a convergence check; --nocheck turns them off.
//    3. Default Omega grid is 5 points (2,3,4,6,8), as requested.
//    4. A weighted log-log fit of the Omega-exponent is done per force,
//       and a cross-force ratio table is printed at the end.
//    5. A cost model runs BEFORE the simulation: <N>_pred = T^a /(A G1 L)
//       per walker is printed per (F,Omega) together with the total step
//       budget, and --dryrun stops right there. The step cap defaults to
//       a multiple of <N>_pred instead of a fixed 2e9.
//
//  Everything else (quenched tau, detailed balance, measured Lambda,
//  heavy-tail statistics) is unchanged from V1.
//
//  Build:
//    g++ -std=c++17 -O3 -march=native -fopenmp -DTCBITS=12 \
//        -o qtm_omega_v2 qtm_omega_scan_v2.cpp
//  Run (the requested scan):
//    ./qtm_omega_v2 --F 0.001,0.005 --T 1e17 --w 2,3,4,6,8 \
//                   --a 1 --b 2 --model invd2 --N 20000 --out omega_v2.csv
//  Plan only, no simulation:
//    ./qtm_omega_v2 --dryrun 1
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
// expected number of steps per walker up to time T: <N> = T^alpha /(A G1 Lambda)
static double N_pred(const Params& P, const Theory& th, double T) {
    return std::pow(T, P.alpha) / (P.A * std::tgamma(1.0+P.alpha) * th.Lambda);
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
//  Heavy-tail-aware statistics.
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
//  Weighted least squares of log(y) on log(x): the Omega-exponent.
//  sigma_i is the standard error of y_i, propagated as sigma_i/y_i.
// ---------------------------------------------------------------------
struct Fit { double slope, slope_se, intercept, r2; int n; };

static Fit fit_loglog(const std::vector<double>& X, const std::vector<double>& Y,
                      const std::vector<double>& SE) {
    Fit f{NAN,NAN,NAN,NAN,0};
    std::vector<double> lx, ly, wt;
    for (size_t i = 0; i < X.size(); ++i) {
        if (!(X[i] > 0.0) || !(Y[i] > 0.0)) continue;             // log needs positives
        const double sl = (SE[i] > 0.0) ? SE[i]/Y[i] : 1e-3;      // sigma of log y
        lx.push_back(std::log(X[i])); ly.push_back(std::log(Y[i]));
        wt.push_back(1.0/(sl*sl));
    }
    f.n = (int)lx.size();
    if (f.n < 2) return f;
    double Sw=0, Sx=0, Sy=0, Sxx=0, Sxy=0;
    for (int i=0;i<f.n;++i){ Sw+=wt[i]; Sx+=wt[i]*lx[i]; Sy+=wt[i]*ly[i];
                             Sxx+=wt[i]*lx[i]*lx[i]; Sxy+=wt[i]*lx[i]*ly[i]; }
    const double det = Sw*Sxx - Sx*Sx;
    if (std::fabs(det) < 1e-300) return f;
    f.slope     = (Sw*Sxy - Sx*Sy)/det;
    f.intercept = (Sxx*Sy - Sx*Sxy)/det;
    f.slope_se  = std::sqrt(Sw/det);
    double ybar = Sy/Sw, sstot=0, ssres=0;
    for (int i=0;i<f.n;++i){
        const double pred = f.intercept + f.slope*lx[i];
        sstot += wt[i]*(ly[i]-ybar)*(ly[i]-ybar);
        ssres += wt[i]*(ly[i]-pred)*(ly[i]-pred);
    }
    f.r2 = (sstot > 0.0) ? 1.0 - ssres/sstot : NAN;
    return f;
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

// one finished (F, Omega) measurement at the largest time
struct Row { double F, w, T, x, se, ci_lo, ci_hi, Nmean, eps, ratio; bool ok; };

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

    // ---- the two scan axes: forces and cross-sections -----------------
    std::vector<double> Fl = parse_list(arg_of(argc,argv,"--F","0.001,0.005"));
    std::sort(Fl.begin(), Fl.end());
    Fl.erase(std::unique(Fl.begin(), Fl.end()), Fl.end());

    std::vector<double> wl = parse_list(arg_of(argc,argv,"--w","2,3,4,6,8"));
    std::sort(wl.begin(), wl.end());
    wl.erase(std::unique(wl.begin(), wl.end()), wl.end());

    uint64_t NW = arg_u(argc,argv,"--N","0");
    if (!NW) NW = arg_u(argc,argv,"--ntraj","20000");

    const double  EPSMAX = arg_d(argc,argv,"--epsmax","0.05");
    const double  SAFETY = arg_d(argc,argv,"--safety","10.0");
    const uint64_t NDIAG = arg_u(argc,argv,"--diag","0");
    const uint64_t MSEED = arg_u(argc,argv,"--seed","20260827");
    const bool     CRN   = arg_u(argc,argv,"--crn","1") != 0;
    const int      NBOOT = (int)arg_u(argc,argv,"--nboot","400");
    const bool     DRY   = arg_u(argc,argv,"--dryrun","0") != 0;
    const bool     NOCHK = arg_u(argc,argv,"--nocheck","0") != 0;
    const double   CAPMUL= arg_d(argc,argv,"--capmul","200.0");   // NCAP = capmul * <N>_pred
    const std::string out = arg_of(argc,argv,"--out","omega_scan_v2.csv");

    const double neg_inv_alpha = -1.0 / P.alpha;
    const double A_raw = std::tgamma(1.0 - P.alpha);
    const double tau_scale = std::pow(P.A / A_raw, 1.0 / P.alpha);
    const double G1 = std::tgamma(1.0 + P.alpha);

    // ---- observation times: one target time + free early checkpoints ---
    std::vector<double> Tl = parse_list(arg_of(argc,argv,"--T","1e17"));
    std::sort(Tl.begin(), Tl.end());
    if (!NOCHK && Tl.size() == 1) {
        const double Tm = Tl[0];
        Tl = { Tm/1000.0, Tm/100.0, Tm/10.0, Tm };
    }
    const int K = (int)Tl.size();
    const double Tmax = Tl.back();

#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
#else
    const int nthr = 1;
#endif
    const double D0 = D0_of(P);

    std::printf("# ===== V2: <x> vs Omega at fixed T, for several forces =====\n");
    std::printf("# a=%g  b=%g  (a/b=%g)  model=%s  bc=%s  alpha=%g  A=%g\n",
                P.a, P.b, P.a/P.b, ms.c_str(), bcs.c_str(), P.alpha, P.A);
    std::printf("# D0 = %.6f   (iso: a^2/4 = %.6f ; invd2: a^2b^2/2(a^2+b^2) = %.6f)\n",
                D0, P.a*P.a/4.0, P.a*P.a*P.b*P.b/(2.0*(P.a*P.a+P.b*P.b)));
    std::printf("# forces F =");
    for (double f : Fl) std::printf(" %g", f);
    std::printf("   Omega =");
    for (double w : wl) std::printf(" %d", (int)std::llround(w));
    std::printf("\n# T =");
    for (double T : Tl) std::printf(" %.3e", T);
    std::printf("   (last one is the target; earlier ones are free convergence checks)\n");
    std::printf("# walkers N = %llu   crn=%d   threads=%d   seed=%llu\n",
                (unsigned long long)NW, (int)CRN, nthr, (unsigned long long)MSEED);
    std::printf("# tau = %.6f * u^(-1/alpha)   [A_raw = Gamma(1-alpha) = %.6f]\n", tau_scale, A_raw);
    std::printf("# prediction: <x> ~ (D0 F)^a (a/Omega)^(1-a) T^a /[A Gamma^2(1+a)]\n");
    std::printf("#   -> log-log slope vs Omega = %.3f at fixed F\n", P.alpha - 1.0);
    if (Fl.size() >= 2)
        std::printf("#   -> <x>(F=%g)/<x>(F=%g) = (F2/F1)^a = %.4f at fixed Omega\n",
                    Fl.back(), Fl.front(), std::pow(Fl.back()/Fl.front(), P.alpha));
    std::printf("# valid row requires eps <= %.3g AND <N>/N* >= %.3g\n", EPSMAX, SAFETY);

    // ---- cost model, before anything expensive starts ------------------
    std::printf("\n# ---- cost plan (theory) ----\n");
    std::printf("# %-8s %-6s %-10s %-11s %-11s %-9s %-8s\n",
                "F","Omega","eps","Lambda","<N>_pred","N*","<N>/N*");
    long double total_steps = 0.0L;
    double Nmax_pred = 0.0;
    for (double f : Fl) for (double wd : wl) {
        Params Q = P; Q.F = f; Q.w = (int)std::llround(wd);
        const Theory t = theory(Q);
        const double Np = N_pred(Q, t, Tmax);
        total_steps += (long double)Np * (long double)NW;
        Nmax_pred = std::max(Nmax_pred, Np);
        std::printf("  %-8g %-6d %-10.3e %-11.4e %-11.3e %-9.2e %-8.2f%s\n",
                    f, Q.w, t.eps, t.Lambda, Np, t.N_star, Np/t.N_star,
                    (t.eps <= EPSMAX && Np/t.N_star >= SAFETY) ? "" : "   <-- marginal");
    }
    std::printf("# total step budget ~ %.3Le steps (%.1f Gsteps); at ~1e8 steps/s/thread\n"
                "#   that is ~%.0f s of wall time on %d threads.\n",
                total_steps, (double)(total_steps/1e9L),
                (double)(total_steps/1e8L)/(double)nthr, nthr);

    uint64_t NCAP = arg_u(argc,argv,"--ncap","0");
    if (!NCAP) NCAP = (uint64_t)std::max(1e7, CAPMUL * Nmax_pred);
    std::printf("# step cap per walker NCAP = %llu  (%.0f x the largest predicted <N>)\n",
                (unsigned long long)NCAP, (double)NCAP/std::max(1.0,Nmax_pred));
    if (DRY) { std::printf("# --dryrun set: stopping before the simulation.\n"); return 0; }

    std::ofstream csv(out);
    csv << "model,bc,alpha,A,a,b,D0,w,Omega,F,T,N_walkers,"
           "x_mean,x_stderr,x_ci_lo,x_ci_hi,x_median,top1pct_share,"
           "N_steps_mean,frac_stuck,capped,"
           "v_par,D_par,D_perp,eps_theory,Lambda_theory,N_star,N_over_Nstar,valid,"
           "x_theory_eq10,x_theory_eq34,eps_meas,Lambda_meas,ndiag,cache_hit_rate\n";

    std::vector<Row> rows;                    // target-time results, all (F,Omega)
    double elapsed_tot = 0.0;

    // =================================================================
    //  outer loop: force.  inner loop: Omega, largest first (cheapest).
    // =================================================================
    for (double f : Fl) {
        P.F = f;
        std::printf("\n# ================ F = %g ================\n", f);
        std::printf("# %-5s %-9s | %-11s %-10s %-9s | %-11s %-9s %-8s %-5s\n",
                    "Omega","T","<x>_sim","stderr","<N>","<x>_Eq10","eps","<N>/N*","ok");

        std::vector<double> order(wl.rbegin(), wl.rend());
        for (double wd : order) {
            P.w = (int)std::llround(wd);
            const auto t0 = std::chrono::high_resolution_clock::now();

            double wxp,wxm,wy; weights(P,wxp,wxm,wy);
            const double Z = wxp+wxm+2.0*wy;
            const double c1 = wxp/Z, c2 = c1+wxm/Z, c3 = c2+wy/Z;
            const Theory th = theory(P);

            // CRN: the walker seed depends only on the walker index, so every
            // (F, Omega) sees the same random stream and the same disorder.
            // Differences between columns of the scan then resolve much better.
            const uint64_t pid = CRN ? MSEED
                                     : mix64(MSEED ^ mix64((uint64_t)P.w * 0xC2B2AE3D27D4EB4FULL)
                                                   ^ (uint64_t)(f*1e9));

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
                                                        C,c1,c2,c3,neg_inv_alpha,tau_scale,Tmax,NCAP);
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

                if (k == K-1)
                    rows.push_back(Row{P.F,(double)P.w,Tl[k],S.mean,S.se,S.ci_lo,S.ci_hi,
                                       Nm,th.eps,rat,ok});
            }
            std::printf("      [F=%g Omega=%d done in %.1fs, cache hit %.2f%%, capped %llu, total %.1fs]\n",
                        f, P.w, secs, 100.0*hit, (unsigned long long)capped_n, elapsed_tot);
            std::fflush(stdout); csv.flush();
        }

        // ---- Omega-exponent for this force ---------------------------
        std::vector<double> X, Y, E, Xo, Yo, Eo;
        for (const Row& r : rows) if (r.F == f) {
            X.push_back(r.w); Y.push_back(r.x); E.push_back(r.se);
            if (r.ok) { Xo.push_back(r.w); Yo.push_back(r.x); Eo.push_back(r.se); }
        }
        const Fit fa = fit_loglog(X,Y,E);
        const Fit fo = fit_loglog(Xo,Yo,Eo);
        std::printf("# fit log<x> vs log(Omega) at T=%.2e, all %d points : slope = %.4f +- %.4f"
                    "  (predicted %.4f)  R2=%.4f\n",
                    Tmax, fa.n, fa.slope, fa.slope_se, P.alpha-1.0, fa.r2);
        if (fo.n >= 2 && fo.n != fa.n)
            std::printf("# fit restricted to the %d rows flagged ok  : slope = %.4f +- %.4f  R2=%.4f\n",
                        fo.n, fo.slope, fo.slope_se, fo.r2);
    }

    // =================================================================
    //  cross-force check of the F-exponent at fixed Omega
    // =================================================================
    if (Fl.size() >= 2) {
        const double F1 = Fl.front(), F2 = Fl.back();
        const double pred = std::pow(F2/F1, P.alpha);
        std::printf("\n# ---- F-scaling at fixed Omega (T = %.2e) ----\n", Tmax);
        std::printf("# predicted <x>(%g)/<x>(%g) = (F2/F1)^alpha = %.4f\n", F2, F1, pred);
        std::printf("# %-6s %-12s %-12s %-10s %-10s\n","Omega","<x>(F1)","<x>(F2)","ratio","rel.dev");
        for (double wd : wl) {
            const int w = (int)std::llround(wd);
            double x1=NAN,x2=NAN;
            for (const Row& r : rows) {
                if ((int)r.w != w) continue;
                if (r.F == F1) x1 = r.x;
                if (r.F == F2) x2 = r.x;
            }
            if (std::isfinite(x1) && std::isfinite(x2) && x1 != 0.0)
                std::printf("  %-6d %-12.4e %-12.4e %-10.4f %+9.1f%%\n",
                            w, x1, x2, x2/x1, 100.0*((x2/x1)/pred - 1.0));
        }
    }

    std::printf("\n# done -> %s   (%.1f s total)\n", out.c_str(), elapsed_tot);
    return 0;
}