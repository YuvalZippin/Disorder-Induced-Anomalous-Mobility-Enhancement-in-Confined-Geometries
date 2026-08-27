// =====================================================================
//  qtm_aniso2d.cpp
//  Direct Monte-Carlo of the 2D Quenched Trap Model in a restricted
//  channel with INDEPENDENT lattice constants:
//      a  = spacing along the open longitudinal axis  (x)
//      b  = spacing along the confined transverse axis (y), w sites
//
//  Nothing about the transport law is assumed. The walker hops on the
//  lattice with the BARE force F entering only through detailed balance,
//  p(e_nu) ~ exp(F . e_nu / 2), and the clock advances by the QUENCHED
//  trapping time of each site.  F^alpha, Omega^-(1-alpha) and Lambda are
//  MEASURED, not imposed.
//
//  Quenched disorder is realised by deriving tau_r from a deterministic
//  hash of (disorder_seed, x, y).  Returning to a site therefore yields
//  the identical tau_r by construction, with O(1) memory and no map.
//
//  Build:
//    g++ -std=c++17 -O3 -march=native -fopenmp -o qtm_aniso2d qtm_aniso2d.cpp
//  Run (example):
//    ./qtm_aniso2d --model invd2 --a 0.5,0.7,1,1.4,2 --b 0.5,0.7,1,1.4,2 \
//                  --w 5 --F 0.01 --alpha 0.3 --T 1e14 --ntraj 100000 \
//                  --diag 2000 --out scan_invd2.csv
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

// ---------------------------------------------------------------------
//  Bit mixing / RNG
// ---------------------------------------------------------------------
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
    // strictly in (0,1] : never 0, so log/pow can never blow up
    inline double u01() { return (double)((next() >> 11) + 1) * 0x1.0p-53; }
};

// ---------------------------------------------------------------------
//  Quenched trapping time of a site  --  tau_r = scale * u_r^(-1/alpha)
//  u_r is a deterministic hash of (disorder seed, x, y): identical on
//  every return to r, which is exactly what "quenched" means.
// ---------------------------------------------------------------------
static inline double tau_of_site(int64_t x, int32_t y, uint64_t dseed,
                                 double neg_inv_alpha, double scale) noexcept {
    uint64_t h = dseed ^ mix64((uint64_t)x * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    h = mix64(h ^ ((uint64_t)(uint32_t)y * 0xC2B2AE3D27D4EB4FULL + 0x165667B19E3779F9ULL));
    const double u = (double)((h >> 11) + 1) * 0x1.0p-53;   // (0,1]
    return scale * std::pow(u, neg_inv_alpha);
}

// ---------------------------------------------------------------------
//  Parameters
// ---------------------------------------------------------------------
enum RateModel { RM_ISO = 0, RM_INVD2 = 1 };
enum Boundary  { BC_PBC = 0, BC_REFLECT = 1 };

struct Params {
    double a = 1.0, b = 1.0;      // lattice constants (longitudinal, transverse)
    int    w = 5;                 // transverse sites  -> Omega = w  in 2D
    double F = 0.01;              // BARE force  F = f/T
    double alpha = 0.3;
    double A = 1.0;               // disorder amplitude (paper uses A = 1)
    double T = 1e1;              // observation time
    RateModel model = RM_INVD2;
    Boundary  bc    = BC_PBC;
};

// jump weights (unnormalised). Force acts along +x only, so the two
// transverse weights are equal and v_perp = 0 identically.
static void weights(const Params& P, double& wxp, double& wxm, double& wy) {
    const double e = std::exp(P.F * P.a / 2.0);   // F . e_x = F * a
    if (P.model == RM_ISO) { wxp = e;               wxm = 1.0 / e;               wy = 1.0; }
    else                   { wxp = e / (P.a*P.a);   wxm = 1.0 / (e*P.a*P.a);     wy = 1.0 / (P.b*P.b); }
}

// ---------------------------------------------------------------------
//  Analytic reference (paper, Eqs. 6, 10, 33, 34) -- for comparison only.
//  Nothing here feeds back into the simulation.
// ---------------------------------------------------------------------
static double polylog_neg(double alpha, double z) {      // Li_{-alpha}(z) = sum_j j^alpha z^j
    if (z <= 0.0) return 0.0;
    double s = 0.0, zj = z;
    for (long j = 1; j < 200000000L; ++j) {
        const double term = std::pow((double)j, alpha) * zj;
        s += term; zj *= z;
        if (j > 32 && term < 1e-16 * s) break;
    }
    return s;
}

struct Theory { double v_par, D_par, D_perp, eps, Lambda, x_exact, x_asym; };

static Theory theory(const Params& P) {
    double wxp, wxm, wy; weights(P, wxp, wxm, wy);
    const double Z = wxp + wxm + 2.0 * wy;
    const double pxp = wxp / Z, pxm = wxm / Z, py = wy / Z;

    Theory th{};
    th.v_par  = P.a * (pxp - pxm);                       // Eq. (19), longitudinal drift
    th.D_par  = 0.5 * P.a * P.a * (pxp + pxm);           // Eq. (19), D_||
    th.D_perp = 0.5 * P.b * P.b * (2.0 * py);            // Eq. (19), D_perp
    th.eps    = (double)P.w * th.v_par / P.a;            // Eq. (33): 1 - Q0 = Omega v/a

    const double Q0 = 1.0 - th.eps;
    const double G1 = std::tgamma(1.0 + P.alpha);
    const double ta = std::pow(P.T, P.alpha);

    if (Q0 > 0.0 && Q0 < 1.0) {
        th.Lambda  = (1.0 - Q0) * (1.0 - Q0) * polylog_neg(P.alpha, Q0) / Q0;   // Eq. (6)
        th.x_exact = th.v_par * ta / (P.A * G1 * th.Lambda);                    // Eq. (10)
    } else { th.Lambda = NAN; th.x_exact = NAN; }
    th.x_asym = th.v_par * std::pow(th.eps, P.alpha - 1.0) * ta / (P.A * G1 * G1);  // Eq. (34)
    return th;
}

// ---------------------------------------------------------------------
//  One trajectory.  Returns net longitudinal displacement in LATTICE
//  INDEX units (multiply by a for physical length).
// ---------------------------------------------------------------------
struct TrajOut { int64_t x; uint64_t N; bool capped; };

static TrajOut run_traj(const Params& P, uint64_t traj_seed,
                        const double c1, const double c2, const double c3,
                        double neg_inv_alpha, double tau_scale, uint64_t NCAP)
{
    Rng rng;  rng.seed(mix64(traj_seed ^ 0x5851F42D4C957F2DULL));
    const uint64_t dseed = mix64(traj_seed ^ 0xA24BAED4963EE407ULL);   // disorder stream

    int64_t x = 0; int32_t y = 0;
    const int w = P.w;

    double t = tau_of_site(0, 0, dseed, neg_inv_alpha, tau_scale);     // initial trapping
    uint64_t N = 0; bool capped = false;

    while (t < P.T) {
        if (N >= NCAP) { capped = true; break; }
        const double r = rng.u01();
        if      (r < c1) { ++x; }
        else if (r < c2) { --x; }
        else if (r < c3) {                                   // +y
            if (P.bc == BC_PBC)      { y = (y + 1 == w) ? 0 : y + 1; }
            else if (y + 1 < w)      { y = y + 1; }          // reflecting: else stay
        } else {                                             // -y
            if (P.bc == BC_PBC)      { y = (y == 0) ? w - 1 : y - 1; }
            else if (y > 0)          { y = y - 1; }
        }
        ++N;
        t += tau_of_site(x, y, dseed, neg_inv_alpha, tau_scale);
    }
    return { x, N, capped };
}

// Diagnostic variant: also measures  1 - Q0 = distinct/N  and  Lambda = S_alpha/N.
struct DiagOut { int64_t x; uint64_t N; uint64_t distinct; double S_alpha; };

static DiagOut run_traj_diag(const Params& P, uint64_t traj_seed,
                             const double c1, const double c2, const double c3,
                             double neg_inv_alpha, double tau_scale, uint64_t NCAP)
{
    Rng rng;  rng.seed(mix64(traj_seed ^ 0x5851F42D4C957F2DULL));
    const uint64_t dseed = mix64(traj_seed ^ 0xA24BAED4963EE407ULL);

    int64_t x = 0; int32_t y = 0; const int w = P.w;
    std::unordered_map<uint64_t, uint32_t> vis;
    vis.reserve(1u << 16);

    auto key = [](int64_t xx, int32_t yy) {
        return ((uint64_t)(uint32_t)yy << 40) ^ (uint64_t)(xx + (1LL << 39));
    };

    double t = tau_of_site(0, 0, dseed, neg_inv_alpha, tau_scale);
    vis[key(0,0)] = 1;
    double S = 1.0; uint64_t N = 0;

    while (t < P.T && N < NCAP) {
        const double r = rng.u01();
        if      (r < c1) { ++x; }
        else if (r < c2) { --x; }
        else if (r < c3) { if (P.bc == BC_PBC) y = (y + 1 == w) ? 0 : y + 1; else if (y + 1 < w) ++y; }
        else             { if (P.bc == BC_PBC) y = (y == 0) ? w - 1 : y - 1; else if (y > 0) --y; }
        ++N;
        uint32_t& n = vis[key(x,y)];
        const double n0 = (double)n; n += 1u;
        S += std::pow(n0 + 1.0, P.alpha) - (n0 > 0.0 ? std::pow(n0, P.alpha) : 0.0);
        t += tau_of_site(x, y, dseed, neg_inv_alpha, tau_scale);
    }
    return { x, N, (uint64_t)vis.size(), S };
}

// ---------------------------------------------------------------------
//  CLI helpers
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
static double   arg_d(int argc, char** argv, const char* k, const char* d) {
    return std::atof(arg_of(argc, argv, k, d).c_str());
}
static uint64_t arg_u(int argc, char** argv, const char* k, const char* d) {
    return strtoull(arg_of(argc, argv, k, d).c_str(), nullptr, 10);
}

// ---------------------------------------------------------------------
int main(int argc, char** argv) {
    Params P;
    const std::vector<double> as = parse_list(arg_of(argc, argv, "--a", "1.0"));
    const std::vector<double> bs = parse_list(arg_of(argc, argv, "--b", "1.0"));
    const std::vector<double> ws = parse_list(arg_of(argc, argv, "--w", "5"));
    const std::vector<double> Fs = parse_list(arg_of(argc, argv, "--F", "0.01"));
    P.alpha        = arg_d(argc, argv, "--alpha", "0.3");
    P.A            = arg_d(argc, argv, "--amp",   "1.0");
    P.T            = arg_d(argc, argv, "--T",     "1e14");
    const uint64_t NTRAJ = arg_u(argc, argv, "--ntraj", "100000");
    const uint64_t NDIAG = arg_u(argc, argv, "--diag", "0");
    const uint64_t NCAP  = arg_u(argc, argv, "--ncap", "200000000");
    const uint64_t MSEED = arg_u(argc, argv, "--seed", "20260827");
    const std::string out = arg_of(argc, argv, "--out", "qtm_scan.csv");
    const std::string ms  = arg_of(argc, argv, "--model", "invd2");
    const std::string bcs = arg_of(argc, argv, "--bc",    "pbc");
    P.model = (ms == "iso") ? RM_ISO : RM_INVD2;
    P.bc    = (bcs == "reflect") ? BC_REFLECT : BC_PBC;

    // tau normalisation: tau = u^(-1/alpha) gives A = Gamma(1-alpha).
    // Rescale so that the realised amplitude equals the requested A.
    const double neg_inv_alpha = -1.0 / P.alpha;
    const double A_raw   = std::tgamma(1.0 - P.alpha);
    const double tau_scale = std::pow(P.A / A_raw, 1.0 / P.alpha);

#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
#else
    const int nthr = 1;
#endif
    std::printf("# QTM 2D anisotropic channel | model=%s  bc=%s  alpha=%g  A=%g  T=%.3e\n",
                (P.model==RM_ISO?"iso":"invd2"), (P.bc==BC_PBC?"pbc":"reflect"), P.alpha, P.A, P.T);
    std::printf("# tau = %.6f * u^(-1/alpha)   (A_raw=Gamma(1-a)=%.6f)   threads=%d  ntraj=%llu\n",
                tau_scale, A_raw, nthr, (unsigned long long)NTRAJ);
    std::printf("# %-5s %-5s %-4s %-7s | %-11s %-11s %-9s | %-11s %-11s | %-9s\n",
                "a","b","w","F","<x>_sim","stderr","<N>","<x>_Eq10","<x>_Eq34","eps");

    std::ofstream csv(out);
    csv << "model,bc,alpha,A,T,a,b,w,Omega,F,ntraj,"
           "x_idx_mean,x_phys_mean,x_phys_stderr,N_mean,frac_stuck,capped,"
           "v_par,D_par,D_perp,eps_theory,Lambda_theory,x_theory_eq10,x_theory_eq34,"
           "eps_meas,Lambda_meas,ndiag\n";

    for (double a : as) for (double b : bs) for (double wd : ws) for (double F : Fs) {
        P.a = a; P.b = b; P.w = (int)std::llround(wd); P.F = F;
        const auto t0 = std::chrono::high_resolution_clock::now();

        double wxp, wxm, wy; weights(P, wxp, wxm, wy);
        const double Z = wxp + wxm + 2.0 * wy;
        const double c1 = wxp / Z, c2 = c1 + wxm / Z, c3 = c2 + wy / Z;
        const Theory th = theory(P);

        // unique, collision-free point id
        const uint64_t pid = mix64(MSEED
              ^ mix64((uint64_t)(a*1e9)  * 0x100000001B3ULL)
              ^ mix64((uint64_t)(b*1e9)  * 0x9E3779B97F4A7C15ULL)
              ^ mix64((uint64_t)P.w      * 0xC2B2AE3D27D4EB4FULL)
              ^ mix64((uint64_t)(F*1e12) * 0xD1B54A32D192ED03ULL));

        long double sx = 0.0L, sxx = 0.0L, sN = 0.0L;
        uint64_t stuck = 0, capped = 0;

        #pragma omp parallel for schedule(dynamic,64) \
                reduction(+:sx,sxx,sN,stuck,capped)
        for (long long n = 0; n < (long long)NTRAJ; ++n) {
            const TrajOut r = run_traj(P, mix64(pid ^ (uint64_t)(n + 1) * 0x9E3779B97F4A7C15ULL),
                                       c1, c2, c3, neg_inv_alpha, tau_scale, NCAP);
            const long double xp = (long double)r.x * (long double)P.a;
            sx += xp; sxx += xp * xp; sN += (long double)r.N;
            if (r.N == 0) ++stuck;
            if (r.capped) ++capped;
        }

        const double xm  = (double)(sx / (long double)NTRAJ);
        const double xv  = (double)(sxx / (long double)NTRAJ) - xm * xm;
        const double xse = std::sqrt(std::max(0.0, xv) / (double)NTRAJ);
        const double Nm  = (double)(sN / (long double)NTRAJ);

        // ---- optional diagnostics: measure 1-Q0 and Lambda directly ----
        double eps_meas = NAN, lam_meas = NAN;
        if (NDIAG > 0) {
            long double sD = 0.0L, sS = 0.0L, sNd = 0.0L;
            #pragma omp parallel for schedule(dynamic,16) reduction(+:sD,sS,sNd)
            for (long long n = 0; n < (long long)NDIAG; ++n) {
                const DiagOut d = run_traj_diag(P, mix64(pid ^ 0xBEEFULL ^ (uint64_t)(n+1) * 0x9E3779B97F4A7C15ULL),
                                                c1, c2, c3, neg_inv_alpha, tau_scale, NCAP);
                if (d.N > 0) { sD += (long double)d.distinct; sS += (long double)d.S_alpha; sNd += (long double)d.N; }
            }
            if (sNd > 0) { eps_meas = (double)(sD / sNd); lam_meas = (double)(sS / sNd); }
        }

        const double secs = std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now() - t0).count();

        std::printf("  %-5.3g %-5.3g %-4d %-7.4g | %-11.4e %-11.2e %-9.3e | %-11.4e %-11.4e | %-9.3e  [%.1fs]\n",
                    a, b, P.w, F, xm, xse, Nm, th.x_exact, th.x_asym, th.eps, secs);
        std::fflush(stdout);

        csv << (P.model==RM_ISO?"iso":"invd2") << "," << (P.bc==BC_PBC?"pbc":"reflect") << ","
            << P.alpha << "," << P.A << "," << P.T << ","
            << a << "," << b << "," << P.w << "," << P.w << "," << F << "," << NTRAJ << ","
            << (xm / P.a) << "," << xm << "," << xse << "," << Nm << ","
            << (double)stuck / (double)NTRAJ << "," << capped << ","
            << th.v_par << "," << th.D_par << "," << th.D_perp << "," << th.eps << ","
            << th.Lambda << "," << th.x_exact << "," << th.x_asym << ","
            << eps_meas << "," << lam_meas << "," << NDIAG << "\n";
        csv.flush();
    }
    std::printf("# done -> %s\n", out.c_str());
    return 0;
}