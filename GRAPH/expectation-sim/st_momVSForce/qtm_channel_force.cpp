// =====================================================================================
//  qtm_channel_force.cpp
//
//  Quenched Trap Model on a restricted channel -- MEAN DISPLACEMENT vs APPLIED FORCE.
//  Unified engine for both lattices of the paper:
//      * simple cubic      (q = 6), square cross-section  Omega = w^2,       PBC walls
//      * simple hexagonal  (q = 8), hexagonal patch       Omega = 3w^2+3w+1, reflecting
//
//  Compile:
//      g++ -std=c++17 -O3 -march=native -fopenmp -o qtm_force qtm_channel_force.cpp
//      (on the Xeon E5-2683 v4 use -march=broadwell instead of -march=native)
//  Run:
//      export OMP_NUM_THREADS=64 && ./qtm_force
//      ./qtm_force --traj 500000 --out run2.csv        (options: see parse_args)
//
// -------------------------------------------------------------------------------------
//  WHAT CHANGED WITH RESPECT TO engine3_graphA_server.cpp / engine3_hex_server.cpp
// -------------------------------------------------------------------------------------
//  (1) The old engines did NOT simulate the model. They drew N from a one-sided Levy
//      subordinator with Lambda = 1 (annealed CTRW) and then injected
//          F_eff = KAPPA * F^alpha * Omega^(alpha-1)
//      i.e. the very scaling law of Eq. (19) was fed in as an INPUT. The resulting
//      figure is an algebraic identity, not a measurement, and it reproduces the
//      prediction for any epsilon. There is therefore no "validity window" to be in.
//
//      Here the walk is simulated microscopically: a real nearest-neighbour walk on the
//      real channel, with a genuinely QUENCHED trap at every site, accumulating
//      t = sum_r n_r tau_r until t >= T. Lambda, Q_0 and the Omega^-(1-alpha) penalty
//      are OUTPUTS, never inputs. The only things put in are q and Omega.
//
//  (2) Quenched disorder is realised by a stateless hash of the site coordinates,
//      tau_r = tau0 * u(r)^(-1/alpha),  u(r) = hash(r, disorder_seed) / 2^64.
//      A return to r therefore always yields the identical tau_r, at O(1) memory,
//      for the ~10^8 distinct sites a walker can reach.
//
//  (3) Amplitude calibration. The paper fixes A = 1. For a Pareto law
//      psi(tau) = alpha tau0^alpha tau^(-1-alpha) the Laplace transform gives
//      A = Gamma(1-alpha) tau0^alpha, so A = 1 requires
//          tau0 = Gamma(1-alpha)^(-1/alpha)     (= 0.41909 at alpha = 0.3).
//      Using tau0 = 1 would silently shift every prefactor by Gamma(1-alpha)^(-1),
//      which is fatal for a "nothing is fitted" comparison.
//
//  (4) Parameters are inside the window. Validity requires eps = 1 - Q_0 = Omega*v_par
//      to be small. The old runs reached eps ~ 10^2 (meaningless: eps is a probability).
//      Here max eps = 0.046, attained only at the widest channel and the largest force.
//
//  (5) Common random numbers: walker n uses the same disorder seed and the same thermal
//      stream at every force and in every channel. Errors across the F-axis are then
//      strongly correlated, so the log-log slope is measured far more precisely than the
//      per-point error bar suggests, and the curves come out smooth.
//
//  (6) A cheap diagnostic pass (few walkers, exact visit histogram) measures
//      1 - Q_0 = D/N and Lambda = S_alpha/N directly, so the run itself certifies that
//      it sits in the nearly recurrent regime. These are the numbers to quote when a
//      referee asks whether the asymptotics were respected.
// =====================================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <omp.h>

// -------------------------------------------------------------------------------------
//  Configuration
// -------------------------------------------------------------------------------------
struct Cfg {
    double   alpha = 0.3;          // disorder index  T/T_g
    double   T     = 1e17;         // measurement time
    uint64_t traj  = 100000;       // walkers (= independent disorder realisations)
    uint64_t diag  = 128;          // walkers used for the Q_0 / Lambda diagnostic
    double   fmin  = 4.0e-5;       // force grid, log spaced
    double   fmax  = 4.0e-3;
    int      nf    = 13;
    uint64_t seed  = 0x5DEECE66DULL;
    std::string out = "qtm_force.csv";
};

struct Channel { bool hex; int w; };

// --- The four channels of the figure -------------------------------------------------
//   #  lattice     w   Omega   D0     eps at F = 4e-3
//   1  cubic       3     9    1/6      0.0060
//   2  hexagonal   2    19    1/8      0.0095
//   3  cubic       7    49    1/6      0.0327
//   4  hexagonal   5    91    1/8      0.0455   <-- binding constraint
//
//   Omega spans 9 -> 91, i.e. a vertical spread of (91/9)^0.7 = 5.1 between the top and
//   bottom curves, against 10^(2*0.3) = 4.0 of variation along the force axis: the
//   figure comes out roughly square in log units and all four curves stay resolved.
//
//   The width/cross-section inversion of the paper survives: the hexagonal channel of
//   width w = 5 is NARROWER than the cubic channel of width w = 7 yet holds MORE sites
//   (91 against 49), and must therefore carry the walker LESS far.
// -------------------------------------------------------------------------------------
static const std::vector<Channel> CHANNELS = {
    { false, 3 },   // simple cubic,      Omega = 9
    { true,  2 },   // simple hexagonal,  Omega = 19
    { false, 7 },   // simple cubic,      Omega = 49
    { true,  5 }    // simple hexagonal,  Omega = 91
};

static inline int omega_of(bool hex, int w) { return hex ? (3*w*w + 3*w + 1) : (w*w); }
static inline int coord_of(bool hex)        { return hex ? 8 : 6; }

// -------------------------------------------------------------------------------------
//  RNG
// -------------------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t z) noexcept {
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27; z *= 0x94D049BB133111EBULL;
    z ^= z >> 31; return z;
}

struct Xoshiro {
    uint64_t s[4];
    static inline uint64_t rotl(uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() noexcept {
        const uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;    s[3]  = rotl(s[3], 45);
        return r;
    }
    inline void seed(uint64_t x) noexcept {
        for (int i = 0; i < 4; ++i) { x += 0x9E3779B97F4A7C15ULL; s[i] = splitmix64(x); }
        for (int i = 0; i < 8; ++i) next();   // warm-up
    }
};

// -------------------------------------------------------------------------------------
//  Quenched waiting time,  tau = tau0 * u^(-1/alpha),  A = 1
//
//  pow() would dominate the run time, so u^(-1/alpha) is evaluated exactly through the
//  binary exponent of the random 64-bit word and a 4096-entry table for the mantissa
//  (linear interpolation, relative error < 1e-6). ~8 operations instead of ~40 cycles,
//  and no table can truncate the heavy tail: the exponent branch is exact.
// -------------------------------------------------------------------------------------
struct TauGen {
    static constexpr int LOGT = 12;
    static constexpr int TSZ  = 1 << LOGT;
    double mant[TSZ + 1];
    double pow2a[65];
    double tau0 = 1.0;

    void init(double alpha) {
        const double inv = -1.0 / alpha;
        for (int i = 0; i <= TSZ; ++i)
            mant[i] = std::pow(1.0 + (double)i / (double)TSZ, inv);
        for (int k = 0; k <= 64; ++k) {
            const double v = std::pow(2.0, (double)k / alpha);
            pow2a[k] = std::isfinite(v) ? v : 1e300;
        }
        // A = Gamma(1-alpha) * tau0^alpha  ==  1
        tau0 = std::pow(1.0 / std::tgamma(1.0 - alpha), 1.0 / alpha);
    }

    inline double operator()(uint64_t h) const noexcept {
        if (h == 0) h = 1;
        const int      k     = __builtin_clzll(h);            // u in [2^-(k+1), 2^-k)
        const uint64_t m53   = (h << k) >> 11;                // implicit 1 at bit 52
        const uint64_t frac  = m53 & ((1ULL << 52) - 1);
        const uint32_t idx   = (uint32_t)(frac >> (52 - LOGT));
        const uint64_t lowm  = (1ULL << (52 - LOGT)) - 1;
        const double   f     = (double)(frac & lowm) * (1.0 / (double)(1ULL << (52 - LOGT)));
        const double   mv    = mant[idx] + (mant[idx + 1] - mant[idx]) * f;
        return tau0 * pow2a[k + 1] * mv;
    }
};

// -------------------------------------------------------------------------------------
//  Site hashing.  The linear form L = x*CL + y*CA + z*CB is maintained incrementally
//  (one add per step); the quenched uniform is splitmix64(L + disorder_seed).
// -------------------------------------------------------------------------------------
static constexpr uint64_t CL = 0x9E3779B97F4A7C15ULL;   // along the channel
static constexpr uint64_t CA = 0xC2B2AE3D27D4EB4FULL;   // transverse axis 1
static constexpr uint64_t CB = 0x165667B19E3779F9ULL;   // transverse axis 2

// triangular-lattice (axial) neighbours, used for the hexagonal cross-section
static const int HEX_DP[6] = { 1, -1,  0,  0,  1, -1 };
static const int HEX_DQ[6] = { 0,  0,  1, -1, -1,  1 };

// -------------------------------------------------------------------------------------
//  Accumulators
// -------------------------------------------------------------------------------------
struct alignas(64) Acc {
    double   sx = 0, sx2 = 0, sN = 0, sN2 = 0;
    double   slam = 0, sesc = 0;
    uint64_t nlam = 0;
    uint64_t steps = 0;
    char pad[8];
};

struct PointResult {
    double mean_x = 0, sem_x = 0, std_x = 0;
    double mean_N = 0, sem_N = 0;
    double lam_meas = 0, esc_meas = 0;
    double v_par = 0;
    double secs = 0;
    double steps = 0;
};

// -------------------------------------------------------------------------------------
//  One (channel, force) point.
//    HEX  = lattice
//    DIAG = also build the exact visit histogram n_r, giving
//              1 - Q_0 = D/N       (D = number of distinct sites)
//              Lambda  = S_alpha/N,  S_alpha = sum_r n_r^alpha
// -------------------------------------------------------------------------------------
template <bool HEX, bool DIAG>
static void run_batch(const Cfg& c, int w, double F, const TauGen& tg,
                      uint64_t ntraj, uint64_t traj_offset, std::vector<Acc>& acc)
{
    const int    qn   = HEX ? 8 : 6;
    const int    ntr  = qn - 2;
    const double Fh   = 0.5 * F;
    const double nrm  = 1.0 / (2.0 * std::cosh(Fh) + (double)ntr);
    const double pp   = nrm * std::exp( Fh);
    const double pm   = nrm * std::exp(-Fh);
    const double TWO64 = 18446744073709551616.0;
    const uint64_t TH1 = (uint64_t)(pp * TWO64);
    const uint64_t TH2 = (uint64_t)((pp + pm) * TWO64);
    const double T     = c.T;
    const double alpha = c.alpha;

    // diagnostic hash table (open addressing, generation-stamped so no clearing is needed)
    constexpr size_t TCAP  = 1u << 18;
    constexpr size_t TMASK = TCAP - 1;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Acc a{};

        std::vector<uint64_t> tkey;
        std::vector<uint32_t> tcnt, tgen, used;
        if constexpr (DIAG) {
            tkey.assign(TCAP, 0); tcnt.assign(TCAP, 0); tgen.assign(TCAP, 0);
            used.reserve(1u << 17);
        }
        uint32_t gen = 0;

        #pragma omp for schedule(dynamic, 32) nowait
        for (long long nn = 0; nn < (long long)ntraj; ++nn) {
            const uint64_t n = (uint64_t)nn + traj_offset;

            // ---- common random numbers: seeds depend on the walker index only --------
            const uint64_t dseed = splitmix64(c.seed ^ (0xA5A5A5A5ULL + n * 0x9E3779B97F4A7C15ULL));
            Xoshiro rng;
            rng.seed(splitmix64(c.seed ^ (0x5A5A5A5AULL + n * 0xC2B2AE3D27D4EB4FULL)));

            int64_t  xl = 0;              // longitudinal coordinate (lattice units)
            int      pa = 0, pb = 0;      // cubic: y,z in [0,w).  hex: axial p,q.
            uint64_t L  = 0;
            double   tau = tg(splitmix64(L + dseed));

            double   t = 0.0, comp = 0.0; // Kahan compensated sum: T = 1e17 vs tau ~ O(1)
            uint64_t N = 0;

            if constexpr (DIAG) { ++gen; used.clear(); }

            for (;;) {
                if constexpr (DIAG) {
                    size_t h = (size_t)(splitmix64(L * 0x9E3779B97F4A7C15ULL) & TMASK);
                    while (tgen[h] == gen && tkey[h] != L) h = (h + 1) & TMASK;
                    if (tgen[h] != gen) { tgen[h] = gen; tkey[h] = L; tcnt[h] = 1; used.push_back((uint32_t)h); }
                    else                { ++tcnt[h]; }
                }

                const double y = tau - comp;
                const double s = t + y;
                comp = (s - t) - y;
                t = s;
                if (t >= T) break;
                ++N;

                const uint64_t r = rng.next();
                bool moved = true;
                if (r < TH1)      { ++xl; L += CL; }
                else if (r < TH2) { --xl; L -= CL; }
                else {
                    const uint64_t r2 = rng.next();
                    if constexpr (!HEX) {
                        // square cross-section, periodic walls
                        switch ((unsigned)(r2 >> 62)) {
                        case 0: if (pa == w-1) { pa = 0;   L -= (uint64_t)(int64_t)(w-1) * CA; }
                                else           { ++pa;     L += CA; } break;
                        case 1: if (pa == 0)   { pa = w-1; L += (uint64_t)(int64_t)(w-1) * CA; }
                                else           { --pa;     L -= CA; } break;
                        case 2: if (pb == w-1) { pb = 0;   L -= (uint64_t)(int64_t)(w-1) * CB; }
                                else           { ++pb;     L += CB; } break;
                        default:if (pb == 0)   { pb = w-1; L += (uint64_t)(int64_t)(w-1) * CB; }
                                else           { --pb;     L -= CB; } break;
                        }
                    } else {
                        // hexagonal patch of the triangular layer, reflecting walls:
                        // a jump that would leave the cross-section is rejected and the
                        // walker stays put -- the step still counts and tau_r is paid again.
                        const unsigned d = (unsigned)(((__uint128_t)r2 * 6u) >> 64);
                        const int np = pa + HEX_DP[d];
                        const int nq = pb + HEX_DQ[d];
                        if (std::abs(np) <= w && std::abs(nq) <= w && std::abs(np + nq) <= w) {
                            L += (uint64_t)(int64_t)HEX_DP[d] * CA
                               + (uint64_t)(int64_t)HEX_DQ[d] * CB;
                            pa = np; pb = nq;
                        } else moved = false;
                    }
                }
                if (moved) tau = tg(splitmix64(L + dseed));
            }

            const double xd = (double)xl, Nd = (double)N;
            a.sx += xd; a.sx2 += xd * xd;
            a.sN += Nd; a.sN2 += Nd * Nd;
            a.steps += N;

            if constexpr (DIAG) {
                double Sa = 0.0;
                for (uint32_t h : used) Sa += std::pow((double)tcnt[h], alpha);
                if (N > 0) {
                    a.slam += Sa / Nd;
                    a.sesc += (double)used.size() / Nd;
                    ++a.nlam;
                }
            }
        }

        acc[tid].sx += a.sx; acc[tid].sx2 += a.sx2;
        acc[tid].sN += a.sN; acc[tid].sN2 += a.sN2;
        acc[tid].slam += a.slam; acc[tid].sesc += a.sesc; acc[tid].nlam += a.nlam;
        acc[tid].steps += a.steps;
    }
}

static PointResult simulate_point(const Cfg& c, const Channel& ch, double F, const TauGen& tg)
{
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int  nth = omp_get_max_threads();

    std::vector<Acc> main_acc(nth), diag_acc(nth);

    if (ch.hex) run_batch<true , false>(c, ch.w, F, tg, c.traj, 0, main_acc);
    else        run_batch<false, false>(c, ch.w, F, tg, c.traj, 0, main_acc);

    if (c.diag > 0) {
        // offset the indices so the diagnostic walkers are independent of the main ones
        if (ch.hex) run_batch<true , true>(c, ch.w, F, tg, c.diag, 1ULL << 40, diag_acc);
        else        run_batch<false, true>(c, ch.w, F, tg, c.diag, 1ULL << 40, diag_acc);
    }

    Acc M{}, D{};
    for (int i = 0; i < nth; ++i) {
        M.sx += main_acc[i].sx; M.sx2 += main_acc[i].sx2;
        M.sN += main_acc[i].sN; M.sN2 += main_acc[i].sN2;
        M.steps += main_acc[i].steps;
        D.slam += diag_acc[i].slam; D.sesc += diag_acc[i].sesc; D.nlam += diag_acc[i].nlam;
        D.steps += diag_acc[i].steps;
    }

    const double m = (double)c.traj;
    PointResult R;
    R.mean_x = M.sx / m;
    R.std_x  = std::sqrt(std::max(0.0, M.sx2 / m - R.mean_x * R.mean_x));
    R.sem_x  = R.std_x / std::sqrt(m);
    R.mean_N = M.sN / m;
    R.sem_N  = std::sqrt(std::max(0.0, M.sN2 / m - R.mean_N * R.mean_N)) / std::sqrt(m);
    R.lam_meas = D.nlam ? D.slam / (double)D.nlam : 0.0;
    R.esc_meas = D.nlam ? D.sesc / (double)D.nlam : 0.0;

    const int    qn  = coord_of(ch.hex);
    const double nrm = 1.0 / (2.0 * std::cosh(0.5 * F) + (double)(qn - 2));
    R.v_par = 2.0 * nrm * std::sinh(0.5 * F);
    R.steps = (double)(M.steps + D.steps);

    R.secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    return R;
}

// -------------------------------------------------------------------------------------
//  CLI
// -------------------------------------------------------------------------------------
static void parse_args(int argc, char** argv, Cfg& c) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if      (k == "--alpha") c.alpha = std::stod(val());
        else if (k == "--t")     c.T     = std::stod(val());
        else if (k == "--traj")  c.traj  = std::stoull(val());
        else if (k == "--diag")  c.diag  = std::stoull(val());
        else if (k == "--fmin")  c.fmin  = std::stod(val());
        else if (k == "--fmax")  c.fmax  = std::stod(val());
        else if (k == "--nf")    c.nf    = std::stoi(val());
        else if (k == "--seed")  c.seed  = std::stoull(val());
        else if (k == "--out")   c.out   = val();
        else if (k == "--help") {
            std::cout << "usage: qtm_force [--alpha 0.3] [--t 1e17] [--traj 200000] [--diag 128]\n"
                         "                 [--fmin 4e-5] [--fmax 4e-3] [--nf 13] [--seed N] [--out f.csv]\n";
            std::exit(0);
        }
    }
}

// -------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(false);
    Cfg c; parse_args(argc, argv, c);

    if (c.alpha <= 0.0 || c.alpha >= 1.0) { std::cerr << "alpha must lie in (0,1)\n"; return 1; }
    for (const auto& ch : CHANNELS)
        if (!ch.hex && ch.w < 3) { std::cerr << "cubic PBC needs w >= 3 (w = 2 makes the two "
                                                "transverse neighbours coincide)\n"; return 1; }

    TauGen tg; tg.init(c.alpha);

    std::vector<double> F(c.nf);
    for (int i = 0; i < c.nf; ++i) {
        const double e = std::log10(c.fmin)
                       + (c.nf > 1 ? i * (std::log10(c.fmax) - std::log10(c.fmin)) / (c.nf - 1) : 0.0);
        F[i] = std::pow(10.0, e);
    }

    const double A      = 1.0;                      // by construction of tau0
    const double G1a    = std::tgamma(1.0 + c.alpha);
    const double Ta     = std::pow(c.T, c.alpha);

    std::cout << "===================================================================\n"
              << " QTM restricted channel -- mean displacement vs force\n"
              << "===================================================================\n"
              << " alpha        = " << c.alpha << "\n"
              << " T            = " << std::scientific << c.T << std::defaultfloat << "\n"
              << " A            = 1  (tau0 = Gamma(1-alpha)^(-1/alpha) = "
              << std::fixed << std::setprecision(6) << tg.tau0 << std::defaultfloat << ")\n"
              << " walkers      = " << c.traj << "   (+ " << c.diag << " diagnostic)\n"
              << " threads      = " << omp_get_max_threads() << "\n"
              << " forces       = " << c.nf << " log-spaced in ["
              << std::scientific << c.fmin << ", " << c.fmax << "]" << std::defaultfloat << "\n\n";

    // ---- plan: show the validity window BEFORE spending the CPU ----------------------
    std::cout << " plan (eps = 1-Q0 = Omega*v_par must be << 1)\n"
              << " ------------------------------------------------------------------\n"
              << "  lattice   w  Omega    D0     eps(Fmax)   <N>(Fmin)    <x>(Fmin)\n";
    double total_steps = 0.0;
    for (const auto& ch : CHANNELS) {
        const int    Om = omega_of(ch.hex, ch.w);
        const int    qn = coord_of(ch.hex);
        const double D0 = 1.0 / (double)qn;
        const double epx = (double)Om * D0 * c.fmax;
        const double vmn = D0 * c.fmin;
        const double emn = (double)Om * vmn;
        const double Nmn = Ta / (A * G1a * G1a * std::pow(emn, 1.0 - c.alpha));
        for (double f : F) {
            const double e = (double)Om * D0 * f;
            total_steps += (double)(c.traj + c.diag)
                         * Ta / (A * G1a * G1a * std::pow(e, 1.0 - c.alpha));
        }
        std::cout << "  " << std::setw(7) << (ch.hex ? "hex" : "cubic")
                  << std::setw(4) << ch.w << std::setw(6) << Om
                  << std::setw(8) << std::fixed << std::setprecision(4) << D0
                  << std::setw(12) << std::setprecision(4) << epx
                  << std::setw(13) << std::scientific << std::setprecision(2) << Nmn
                  << std::setw(13) << vmn * Nmn << std::defaultfloat << "\n";
    }
    std::cout << " ------------------------------------------------------------------\n"
              << " estimated total steps = " << std::scientific << std::setprecision(2)
              << total_steps << std::defaultfloat
              << "   (~" << std::fixed << std::setprecision(1) << total_steps / 5e9 / 3600.0
              << " h at 5e9 steps/s)\n\n" << std::defaultfloat;

    std::ofstream csv(c.out);
    csv << "lattice,w,Omega,q,D0,alpha,T,A,F,v_par,eps_theory,M,"
           "mean_x,sem_x,std_x,mean_N,sem_N,M_diag,esc_meas,lambda_meas,steps,seconds\n";
    csv << std::setprecision(17);

    for (const auto& ch : CHANNELS) {
        const int    Om = omega_of(ch.hex, ch.w);
        const int    qn = coord_of(ch.hex);
        const double D0 = 1.0 / (double)qn;
        std::cout << " --- " << (ch.hex ? "hexagonal" : "cubic") << "  w = " << ch.w
                  << "  Omega = " << Om << " ---\n";

        for (double f : F) {
            const PointResult R = simulate_point(c, ch, f, tg);
            const double eps_th = (double)Om * R.v_par;
            const double lam_th = G1a * std::pow(eps_th, 1.0 - c.alpha);

            std::cout << std::scientific << std::setprecision(3)
                      << "   F = " << f
                      << " | eps = " << eps_th
                      << " (meas " << R.esc_meas << ")"
                      << " | Lam = " << lam_th
                      << " (meas " << R.lam_meas << ")"
                      << " | <x> = " << R.mean_x << " +/- " << R.sem_x
                      << " | <N> = " << R.mean_N
                      << std::fixed << std::setprecision(1) << " | " << R.secs << " s\n"
                      << std::defaultfloat;

            csv << (ch.hex ? "hex" : "cubic") << ',' << ch.w << ',' << Om << ',' << qn << ','
                << D0 << ',' << c.alpha << ',' << c.T << ',' << A << ','
                << f << ',' << R.v_par << ',' << eps_th << ',' << c.traj << ','
                << R.mean_x << ',' << R.sem_x << ',' << R.std_x << ','
                << R.mean_N << ',' << R.sem_N << ',' << c.diag << ','
                << R.esc_meas << ',' << R.lam_meas << ','
                << R.steps << ',' << R.secs << '\n';
            csv.flush();
        }
    }

    csv.close();
    std::cout << "\n done -> " << c.out << "\n";
    return 0;
}