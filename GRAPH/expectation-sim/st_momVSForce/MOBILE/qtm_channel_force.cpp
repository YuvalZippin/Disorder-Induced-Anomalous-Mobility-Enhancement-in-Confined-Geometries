// =====================================================================================
//  qtm_channel_force.cpp   (v2 -- portable: laptop and server from one source)
//
//  Quenched Trap Model on a restricted channel -- MEAN DISPLACEMENT vs APPLIED FORCE.
//      * simple cubic      (q = 6), square cross-section  Omega = w^2,       PBC walls
//      * simple hexagonal  (q = 8), hexagonal patch       Omega = 3w^2+3w+1, reflecting
//
//  Build (no OpenMP, no external dependency -- std::thread only):
//      Linux / macOS :  g++ -std=c++17 -O3 -march=native -pthread -o qtm_force qtm_channel_force.cpp
//      Apple clang   :  clang++ -std=c++17 -O3 -mcpu=native -pthread -o qtm_force qtm_channel_force.cpp
//      Xeon server   :  g++ -std=c++17 -O3 -march=broadwell -pthread -o qtm_force qtm_channel_force.cpp
//
//  Run:
//      ./qtm_force --preset laptop                 (~1-2 h on 8 cores)
//      ./qtm_force --preset server --threads 64    (production, t = 1e17)
//      ./qtm_force --bench                         (measure the step rate and stop)
//
// -------------------------------------------------------------------------------------
//  WHY THIS RUNS ON A LAPTOP AT ALL
// -------------------------------------------------------------------------------------
//  The statistical error of <x> at one point is dominated by the diffusive spread of the
//  walker about its drift,
//        delta = sqrt(2 D0 / <N>) / (v sqrt(M)),      <N> = t^a / (A G^2(1+a) eps^(1-a)),
//  so the number of steps needed to reach a prescribed delta is
//        M <N> = 2 D0 / (v^2 delta^2) = 2 / (D0 F^2 delta^2).
//  That budget is INDEPENDENT of t and of Omega. Two consequences drive the design:
//
//    * t is nearly free to lower. Reducing t from 1e17 to 1e14 leaves the accuracy-per-
//      step untouched; it only shortens each walker and raises the walker count. The one
//      quantity that does degrade is the number of distinct traps a walker samples,
//      D = (t eps)^alpha / [A Gamma^2(1+alpha)], which controls how sharply
//      S_alpha concentrates on Lambda N. At t = 1e14 the worst point still has D ~ 1.5e3,
//      far inside the regime where that collapse holds. The laptop preset therefore
//      tests exactly the same physics as the production run, at a smaller scale.
//
//    * The cost is set by F_min alone, as 1/F^2. Shortening the force range from two
//      decades to 1.5 is worth a factor of ten, which is why the laptop preset starts
//      at F = 1.2e-4 rather than 4e-5.
//
//  (2) Adaptive walker count. The old fixed N_TRAJ oversamples the large forces by
//      orders of magnitude -- at F = 4e-3 a few thousand walkers already give 1%, while
//      at F = 4e-5 even 1e5 leave 5%. Each point now runs in batches until its relative
//      standard error meets --target, between --min-traj and --max-traj. Uniform error
//      bars, and roughly a factor of five saved over a flat walker count.
//
//  (3) std::thread instead of OpenMP: Apple clang ships without -fopenmp, and the work
//      here is embarrassingly parallel. Walkers are handed out dynamically through an
//      atomic counter, so the heavy-tailed spread in N does not leave threads idle.
//
//  Everything about the model is unchanged from v1: genuinely quenched traps addressed
//  by a stateless hash, A = 1 through tau0 = Gamma(1-alpha)^(-1/alpha), common random
//  numbers across the force axis, and a diagnostic pass that measures 1 - Q_0 and Lambda.
// =====================================================================================

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int clz64(uint64_t x) { unsigned long i; _BitScanReverse64(&i, x); return 63 - (int)i; }
#else
  static inline int clz64(uint64_t x) { return __builtin_clzll(x); }
#endif

static inline unsigned pick6(uint64_t r) {
#if defined(__SIZEOF_INT128__)
    return (unsigned)(((__uint128_t)r * 6u) >> 64);
#else
    return (unsigned)(r % 6u);
#endif
}

// -------------------------------------------------------------------------------------
struct Cfg {
    double   alpha     = 0.3;
    double   T         = 1e17;
    double   fmin      = 4.0e-5;
    double   fmax      = 4.0e-3;
    int      nf        = 13;
    double   target    = 0.02;      // wanted relative standard error of <x>
    uint64_t min_traj  = 5000;
    uint64_t max_traj  = 4000000;
    uint64_t diag      = 128;
    int      threads   = 0;         // 0 -> hardware_concurrency
    uint64_t seed      = 0x5DEECE66DULL;
    std::string out    = "qtm_force.csv";
    bool     bench     = false;
};

struct Channel { bool hex; int w; };

// --- the four channels ---------------------------------------------------------------
//   lattice     w   Omega   D0     eps at F = 4e-3
//   cubic       3     9    1/6      0.0060
//   hexagonal   2    19    1/8      0.0095
//   cubic       7    49    1/6      0.0327
//   hexagonal   5    91    1/8      0.0455   <-- binding constraint on the window
//
//   The hexagonal channel of width 5 is narrower than the cubic channel of width 7 yet
//   holds more sites (91 against 49), so it must carry the walker less far: the
//   width/cross-section inversion of the paper is preserved.
static const std::vector<Channel> CHANNELS = {
    { true,  1 },   // hexagonal, Omega =  7
    { false, 4 },   // cubic,     Omega = 16
    { false, 6 },   // cubic,     Omega = 36
    { true,  4 }    // hexagonal, Omega = 61
};

static inline int omega_of(bool hex, int w) { return hex ? (3*w*w + 3*w + 1) : (w*w); }
static inline int coord_of(bool hex)        { return hex ? 8 : 6; }

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
        for (int i = 0; i < 8; ++i) next();
    }
};

// -------------------------------------------------------------------------------------
//  tau = tau0 * u^(-1/alpha) with A = 1, evaluated without pow(): the binary exponent of
//  the random word is exact and only the mantissa is tabulated (4096 nodes, linear
//  interpolation, relative error < 1e-6). The heavy tail is therefore never truncated.
// -------------------------------------------------------------------------------------
struct TauGen {
    static constexpr int LOGT = 12;
    static constexpr int TSZ  = 1 << LOGT;
    double mant[TSZ + 1];
    double pow2a[65];
    double tau0 = 1.0;

    void init(double alpha) {
        const double inv = -1.0 / alpha;
        for (int i = 0; i <= TSZ; ++i) mant[i] = std::pow(1.0 + (double)i / (double)TSZ, inv);
        for (int k = 0; k <= 64; ++k) {
            const double v = std::pow(2.0, (double)k / alpha);
            pow2a[k] = std::isfinite(v) ? v : 1e300;
        }
        tau0 = std::pow(1.0 / std::tgamma(1.0 - alpha), 1.0 / alpha);   // A = 1
    }
    inline double operator()(uint64_t h) const noexcept {
        if (h == 0) h = 1;
        const int      k    = clz64(h);
        const uint64_t m53  = (h << k) >> 11;
        const uint64_t frac = m53 & ((1ULL << 52) - 1);
        const uint32_t idx  = (uint32_t)(frac >> (52 - LOGT));
        const uint64_t lowm = (1ULL << (52 - LOGT)) - 1;
        const double   f    = (double)(frac & lowm) * (1.0 / (double)(1ULL << (52 - LOGT)));
        return tau0 * pow2a[k + 1] * (mant[idx] + (mant[idx + 1] - mant[idx]) * f);
    }
};

static constexpr uint64_t CL = 0x9E3779B97F4A7C15ULL;   // along the channel
static constexpr uint64_t CA = 0xC2B2AE3D27D4EB4FULL;   // transverse axis 1
static constexpr uint64_t CB = 0x165667B19E3779F9ULL;   // transverse axis 2
static const int HEX_DP[6] = { 1, -1,  0,  0,  1, -1 };
static const int HEX_DQ[6] = { 0,  0,  1, -1, -1,  1 };

struct alignas(64) Acc {
    double   sx = 0, sx2 = 0, sN = 0, sN2 = 0, slam = 0, sesc = 0;
    uint64_t nlam = 0, steps = 0;
    char pad[8];
};

// -------------------------------------------------------------------------------------
//  Walkers [lo, hi). Seeds depend on the walker index only, so a point can be extended
//  batch by batch and the same disorder is reused at every force (common random numbers).
// -------------------------------------------------------------------------------------
template <bool HEX, bool DIAG>
static void walk_range(const Cfg& c, int w, uint64_t TH1, uint64_t TH2, const TauGen& tg,
                       uint64_t lo, uint64_t hi, uint64_t index_offset,
                       std::atomic<uint64_t>& cursor, Acc& out)
{
    constexpr size_t TCAP = 1u << 18, TMASK = TCAP - 1;
    const double T = c.T, alpha = c.alpha;

    std::vector<uint64_t> tkey;
    std::vector<uint32_t> tcnt, tgen, used;
    if constexpr (DIAG) { tkey.assign(TCAP, 0); tcnt.assign(TCAP, 0); tgen.assign(TCAP, 0); used.reserve(1u << 17); }
    uint32_t gen = 0;

    Acc a{};
    for (;;) {
        const uint64_t base = cursor.fetch_add(16, std::memory_order_relaxed);
        if (base >= hi) break;
        const uint64_t end = (base + 16 < hi) ? base + 16 : hi;

        for (uint64_t nn = base; nn < end; ++nn) {
            const uint64_t n = nn + index_offset;
            const uint64_t dseed = splitmix64(c.seed ^ (0xA5A5A5A5ULL + n * 0x9E3779B97F4A7C15ULL));
            Xoshiro rng; rng.seed(splitmix64(c.seed ^ (0x5A5A5A5AULL + n * 0xC2B2AE3D27D4EB4FULL)));

            int64_t  xl = 0;
            int      pa = 0, pb = 0;
            uint64_t L  = 0;
            double   tau = tg(splitmix64(L + dseed));
            // Countdown rather than accumulation. One dependent flop per step instead of
            // the four of a Kahan sum, and better behaved: the ulp of the remainder
            // shrinks as it approaches zero, so the last increments are never swallowed
            // the way they are when adding tau ~ 1 to a running total of 1e17.
            double   rem = T;
            uint64_t N = 0;

            if constexpr (DIAG) { ++gen; used.clear(); }

            for (;;) {
                if constexpr (DIAG) {
                    size_t h = (size_t)(splitmix64(L * 0x9E3779B97F4A7C15ULL) & TMASK);
                    while (tgen[h] == gen && tkey[h] != L) h = (h + 1) & TMASK;
                    if (tgen[h] != gen) { tgen[h] = gen; tkey[h] = L; tcnt[h] = 1; used.push_back((uint32_t)h); }
                    else ++tcnt[h];
                }
                rem -= tau;
                if (rem <= 0.0) break;
                ++N;

                const uint64_t r = rng.next();
                bool moved = true;
                if      (r < TH1) { ++xl; L += CL; }
                else if (r < TH2) { --xl; L -= CL; }
                else {
                    const uint64_t r2 = rng.next();
                    if constexpr (!HEX) {
                        switch ((unsigned)(r2 >> 62)) {
                        case 0: if (pa == w-1) { pa = 0;   L -= (uint64_t)(int64_t)(w-1) * CA; } else { ++pa; L += CA; } break;
                        case 1: if (pa == 0)   { pa = w-1; L += (uint64_t)(int64_t)(w-1) * CA; } else { --pa; L -= CA; } break;
                        case 2: if (pb == w-1) { pb = 0;   L -= (uint64_t)(int64_t)(w-1) * CB; } else { ++pb; L += CB; } break;
                        default:if (pb == 0)   { pb = w-1; L += (uint64_t)(int64_t)(w-1) * CB; } else { --pb; L -= CB; } break;
                        }
                    } else {
                        const unsigned d = pick6(r2);
                        const int np = pa + HEX_DP[d], nq = pb + HEX_DQ[d];
                        if (std::abs(np) <= w && std::abs(nq) <= w && std::abs(np + nq) <= w) {
                            L += (uint64_t)(int64_t)HEX_DP[d] * CA + (uint64_t)(int64_t)HEX_DQ[d] * CB;
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
                if (N > 0) { a.slam += Sa / Nd; a.sesc += (double)used.size() / Nd; ++a.nlam; }
            }
        }
    }
    out = a;
}

template <bool HEX, bool DIAG>
static Acc run_batch(const Cfg& c, int w, uint64_t TH1, uint64_t TH2, const TauGen& tg,
                     uint64_t lo, uint64_t hi, uint64_t index_offset, int nthreads)
{
    std::atomic<uint64_t> cursor(lo);
    std::vector<Acc> part((size_t)nthreads);
    std::vector<std::thread> th;
    th.reserve((size_t)nthreads);
    for (int i = 0; i < nthreads; ++i)
        th.emplace_back([&, i] { walk_range<HEX, DIAG>(c, w, TH1, TH2, tg, lo, hi,
                                                       index_offset, cursor, part[(size_t)i]); });
    for (auto& t : th) t.join();

    Acc tot{};
    for (const auto& p : part) {
        tot.sx += p.sx; tot.sx2 += p.sx2; tot.sN += p.sN; tot.sN2 += p.sN2;
        tot.slam += p.slam; tot.sesc += p.sesc; tot.nlam += p.nlam; tot.steps += p.steps;
    }
    return tot;
}

struct PointResult {
    double mean_x = 0, sem_x = 0, std_x = 0, mean_N = 0, sem_N = 0;
    double lam_meas = 0, esc_meas = 0, v_par = 0, secs = 0, steps = 0;
    uint64_t M = 0;
};

// -------------------------------------------------------------------------------------
//  One (channel, force) point, with the walker count grown until --target is met.
// -------------------------------------------------------------------------------------
static PointResult simulate_point(const Cfg& c, const Channel& ch, double F,
                                  const TauGen& tg, int nthreads)
{
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int    qn  = coord_of(ch.hex);
    const double Fh  = 0.5 * F;
    const double nrm = 1.0 / (2.0 * std::cosh(Fh) + (double)(qn - 2));
    const double pp  = nrm * std::exp( Fh), pm = nrm * std::exp(-Fh);
    const double TWO64 = 18446744073709551616.0;
    const uint64_t TH1 = (uint64_t)(pp * TWO64), TH2 = (uint64_t)((pp + pm) * TWO64);

    Acc tot{};
    uint64_t done = 0;
    while (done < c.max_traj) {
        uint64_t want = (done == 0) ? c.min_traj : 0;
        if (done > 0) {
            const double m  = tot.sx / (double)done;
            const double sd = std::sqrt(std::max(0.0, tot.sx2 / (double)done - m * m));
            const double need = (std::abs(m) > 0) ? (sd / (c.target * std::abs(m))) * (sd / (c.target * std::abs(m)))
                                                  : (double)c.max_traj;
            if (need <= (double)done) break;
            want = (uint64_t)std::min((double)(2 * done), std::min(need - (double)done + 1.0,
                                                                  (double)(c.max_traj - done)));
            if (want == 0) break;
        }
        const uint64_t lo = done, hi = std::min(done + want, c.max_traj);
        Acc b = ch.hex ? run_batch<true , false>(c, ch.w, TH1, TH2, tg, lo, hi, 0, nthreads)
                       : run_batch<false, false>(c, ch.w, TH1, TH2, tg, lo, hi, 0, nthreads);
        tot.sx += b.sx; tot.sx2 += b.sx2; tot.sN += b.sN; tot.sN2 += b.sN2; tot.steps += b.steps;
        done = hi;
    }

    Acc dg{};
    if (c.diag > 0) {
        dg = ch.hex ? run_batch<true , true>(c, ch.w, TH1, TH2, tg, 0, c.diag, 1ULL << 40, nthreads)
                    : run_batch<false, true>(c, ch.w, TH1, TH2, tg, 0, c.diag, 1ULL << 40, nthreads);
    }

    const double m = (double)done;
    PointResult R;
    R.M      = done;
    R.mean_x = tot.sx / m;
    R.std_x  = std::sqrt(std::max(0.0, tot.sx2 / m - R.mean_x * R.mean_x));
    R.sem_x  = R.std_x / std::sqrt(m);
    R.mean_N = tot.sN / m;
    R.sem_N  = std::sqrt(std::max(0.0, tot.sN2 / m - R.mean_N * R.mean_N)) / std::sqrt(m);
    R.lam_meas = dg.nlam ? dg.slam / (double)dg.nlam : 0.0;
    R.esc_meas = dg.nlam ? dg.sesc / (double)dg.nlam : 0.0;
    R.v_par  = 2.0 * nrm * std::sinh(Fh);
    R.steps  = (double)(tot.steps + dg.steps);
    R.secs   = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    return R;
}

// -------------------------------------------------------------------------------------
static void apply_preset(Cfg& c, const std::string& p) {
    // NOTE ON t. The measurement time is large because the second validity condition
    // below, rho >> 1, is far more demanding than eps << 1. See the header.
    if (p == "server") {            // production: ~3 h on sixty-four cores
        c.T = 1e24; c.fmin = 1.71e-3; c.fmax = 7.87e-3; c.nf = 11;
        c.target = 0.02; c.min_traj = 2000; c.max_traj = 500000; c.diag = 128;
    } else if (p == "long") {       // longer force lever arm, ~21 h on sixty-four cores
        c.T = 1e26; c.fmin = 5.91e-4; c.fmax = 7.87e-3; c.nf = 13;
        c.target = 0.02; c.min_traj = 2000; c.max_traj = 500000; c.diag = 128;
    } else if (p == "laptop") {     // ~15 min on eight cores. PIPELINE TEST, NOT PRODUCTION:
        c.T = 1e21; c.fmin = 3.0e-3; c.fmax = 7.87e-3; c.nf = 5;   // rho is only ~50 here,
        c.target = 0.05; c.min_traj = 500; c.max_traj = 50000; c.diag = 48;   // so expect
    } else if (p == "quick") {      // eps_meas to sit ~10% above Omega*v. Check the plumbing.
        c.T = 1e18; c.fmin = 7.87e-3; c.fmax = 7.87e-3; c.nf = 1;
        c.target = 0.10; c.min_traj = 200; c.max_traj = 5000; c.diag = 24;
    } else { std::cerr << "unknown preset: " << p << "\n"; std::exit(1); }
}

static void parse_args(int argc, char** argv, Cfg& c) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if      (k == "--preset")   apply_preset(c, val());
        else if (k == "--alpha")    c.alpha    = std::stod(val());
        else if (k == "--t")        c.T        = std::stod(val());
        else if (k == "--fmin")     c.fmin     = std::stod(val());
        else if (k == "--fmax")     c.fmax     = std::stod(val());
        else if (k == "--nf")       c.nf       = std::stoi(val());
        else if (k == "--target")   c.target   = std::stod(val());
        else if (k == "--min-traj") c.min_traj = std::stoull(val());
        else if (k == "--max-traj") c.max_traj = std::stoull(val());
        else if (k == "--diag")     c.diag     = std::stoull(val());
        else if (k == "--threads")  c.threads  = std::stoi(val());
        else if (k == "--seed")     c.seed     = std::stoull(val());
        else if (k == "--out")      c.out      = val();
        else if (k == "--bench")    c.bench    = true;
        else if (k == "--help") {
            std::cout <<
              "usage: qtm_force [--preset laptop|server|quick] [--threads N] [--bench]\n"
              "                 [--alpha 0.3] [--t 1e17] [--fmin F] [--fmax F] [--nf N]\n"
              "                 [--target 0.02] [--min-traj N] [--max-traj N] [--diag N]\n"
              "                 [--seed N] [--out file.csv]\n"
              "  --target  wanted relative standard error of <x>; walkers are added\n"
              "            until it is met, between --min-traj and --max-traj.\n";
            std::exit(0);
        }
        else { std::cerr << "unknown option: " << k << "\n"; std::exit(1); }
    }
}

// -------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(false);
    Cfg c; parse_args(argc, argv, c);
    if (c.alpha <= 0.0 || c.alpha >= 1.0) { std::cerr << "alpha must lie in (0,1)\n"; return 1; }
    for (const auto& ch : CHANNELS)
        if (!ch.hex && ch.w < 3) { std::cerr << "cubic PBC needs w >= 3\n"; return 1; }

    int nth = c.threads > 0 ? c.threads : (int)std::thread::hardware_concurrency();
    if (nth < 1) nth = 1;

    TauGen tg; tg.init(c.alpha);
    const double A = 1.0, G1a = std::tgamma(1.0 + c.alpha), Ta = std::pow(c.T, c.alpha);

    // ---- benchmark: measure the real step rate of THIS machine ----------------------
    {
        Cfg b = c; b.T = 1e9; b.diag = 0;
        const double Fb = 1e-3, Fh = 0.5 * Fb, nrm = 1.0 / (2.0 * std::cosh(Fh) + 4.0);
        const double TWO64 = 18446744073709551616.0;
        const uint64_t T1 = (uint64_t)(nrm * std::exp(Fh) * TWO64);
        const uint64_t T2 = (uint64_t)((nrm * std::exp(Fh) + nrm * std::exp(-Fh)) * TWO64);
        const auto s = std::chrono::high_resolution_clock::now();
        Acc r = run_batch<false, false>(b, 3, T1, T2, tg, 0, (uint64_t)(200 * nth), 0, nth);
        const double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - s).count();
        const double rate = (double)r.steps / std::max(dt, 1e-9);
        std::cout << "[bench] " << nth << " thread(s): " << std::scientific << std::setprecision(3)
                  << rate << " steps/s  (" << rate / nth << " per thread)\n" << std::defaultfloat;
        if (c.bench) return 0;
    }

    std::vector<double> F(c.nf);
    for (int i = 0; i < c.nf; ++i)
        F[i] = std::pow(10.0, std::log10(c.fmin) + (c.nf > 1
               ? i * (std::log10(c.fmax) - std::log10(c.fmin)) / (c.nf - 1) : 0.0));

    std::cout << "===================================================================\n"
              << " QTM restricted channel -- mean displacement vs force\n"
              << "===================================================================\n"
              << " alpha    = " << c.alpha << "\n"
              << " T        = " << std::scientific << c.T << std::defaultfloat << "\n"
              << " A        = 1   (tau0 = " << std::fixed << std::setprecision(6) << tg.tau0
              << ")" << std::defaultfloat << "\n"
              << " target   = " << c.target * 100 << " % relative error, walkers in ["
              << c.min_traj << ", " << c.max_traj << "]\n"
              << " threads  = " << nth << "\n"
              << " forces   = " << c.nf << " in [" << std::scientific << c.fmin << ", "
              << c.fmax << "]" << std::defaultfloat << "\n\n"
              << " plan (eps = 1-Q0 = Omega v_par;  D = distinct traps per walker)\n"
              << " -----------------------------------------------------------------------\n"
              << "  lattice   w  Omega      D0   eps(Fmax)  rho(Fmin)   <N>(Fmin)   est.steps\n";

    double grand = 0.0;
    for (const auto& ch : CHANNELS) {
        const int Om = omega_of(ch.hex, ch.w), qn = coord_of(ch.hex);
        const double D0 = 1.0 / (double)qn;
        const double emn = (double)Om * D0 * c.fmin;
        const double Nmn = Ta / (A * G1a * G1a * std::pow(emn, 1.0 - c.alpha));
        double sub = 0.0;
        for (double f : F) {                       // adaptive budget: 2/(D0 F^2 target^2)
            const double v = D0 * f;
            const double e = (double)Om * v;
            const double Nf = Ta / (A * G1a * G1a * std::pow(e, 1.0 - c.alpha));
            const double need = 2.0 * D0 / (v * v * c.target * c.target);
            sub += std::min(std::max(need, (double)c.min_traj * Nf), (double)c.max_traj * Nf);
        }
        grand += sub;
        std::cout << "  " << std::setw(7) << (ch.hex ? "hex" : "cubic") << std::setw(4) << ch.w
                  << std::setw(6) << Om << std::setw(9) << std::fixed << std::setprecision(4) << D0
                  << std::setw(11) << std::setprecision(4) << (double)Om * D0 * c.fmax
                  << std::setw(11) << std::setprecision(1) << Nmn * (D0*c.fmin)*(D0*c.fmin) / D0
                  << std::setw(12) << std::scientific << std::setprecision(2) << Nmn
                  << std::setw(12) << sub << std::defaultfloat << "\n";
    }
    std::cout << " -----------------------------------------------------------------------\n"
              << " estimated total = " << std::scientific << std::setprecision(2) << grand
              << " steps\n\n" << std::defaultfloat;

    std::ofstream csv(c.out);
    csv << "lattice,w,Omega,q,D0,alpha,T,A,F,v_par,eps_theory,M,"
           "mean_x,sem_x,std_x,mean_N,sem_N,M_diag,esc_meas,lambda_meas,rho,steps,seconds\n";
    csv << std::setprecision(17);

    const auto wall0 = std::chrono::high_resolution_clock::now();
    double done_steps = 0.0;

    for (const auto& ch : CHANNELS) {
        const int Om = omega_of(ch.hex, ch.w), qn = coord_of(ch.hex);
        const double D0 = 1.0 / (double)qn;
        std::cout << " --- " << (ch.hex ? "hexagonal" : "cubic") << "  w = " << ch.w
                  << "  Omega = " << Om << " ---\n";
        for (double f : F) {
            const PointResult R = simulate_point(c, ch, f, tg, nth);
            done_steps += R.steps;
            const double eps_th = (double)Om * R.v_par;
            const double lam_th = G1a * std::pow(eps_th, 1.0 - c.alpha);
            const double rho    = R.mean_N * R.v_par * R.v_par / D0;
            const double el = std::chrono::duration<double>(
                                  std::chrono::high_resolution_clock::now() - wall0).count();

            std::cout << std::scientific << std::setprecision(3)
                      << "   F=" << f << " | eps=" << eps_th << " (m " << R.esc_meas << ")"
                      << " | rho=" << rho
                      << " | Lam=" << lam_th << " (m " << R.lam_meas << ")"
                      << " | <x>=" << R.mean_x << " +/- " << R.sem_x
                      << std::fixed << std::setprecision(1)
                      << " | M=" << R.M << " | " << R.secs << "s"
                      << " | eta " << (grand > done_steps ? (el / done_steps) * (grand - done_steps) / 60.0 : 0.0)
                      << " min\n" << std::defaultfloat << std::flush;

            csv << (ch.hex ? "hex" : "cubic") << ',' << ch.w << ',' << Om << ',' << qn << ','
                << D0 << ',' << c.alpha << ',' << c.T << ',' << A << ','
                << f << ',' << R.v_par << ',' << eps_th << ',' << R.M << ','
                << R.mean_x << ',' << R.sem_x << ',' << R.std_x << ','
                << R.mean_N << ',' << R.sem_N << ',' << c.diag << ','
                << R.esc_meas << ',' << R.lam_meas << ',' << rho << ','
                << R.steps << ',' << R.secs << '\n';
            csv.flush();
        }
    }
    csv.close();
    std::cout << "\n done -> " << c.out << "   ("
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall0).count() / 60.0
              << " min)\n";
    return 0;
}