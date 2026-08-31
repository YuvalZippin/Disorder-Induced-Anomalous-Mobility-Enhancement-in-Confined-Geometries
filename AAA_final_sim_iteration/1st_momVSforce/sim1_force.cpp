// =============================================================================
//  SIM 1 : First moment <x_par(t)> as a function of the applied force F_par
//  Quenched Trap Model in a restricted channel.
//    * Simple Cubic      , channel along a cubic axis   , PERIODIC   transverse BC
//    * Simple Hexagonal  , channel along the stacking axis, REFLECTING transverse BC
//  Output: sim1_results.csv , sim1_channels.csv
//  Build : see run instructions.
// =============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

// ============================ SIMULATION PARAMETERS ==========================
// -------- everything the physicist tunes lives in this block -----------------

constexpr long long N_WALKERS = 100000;      // N        = 1e5
constexpr double    T_MAX     = 1e14;        // t        = 1e14   (STRICT, never scaled)
constexpr double    ALPHA     = 0.3;         // alpha    = T/T_g
constexpr double    AMP_A     = 1.0;         // A        = disorder amplitude
constexpr int       POINTS    = 20;          // points scanned in FORCE
constexpr double    LAT_A     = 1.0;         // lattice constant a

enum RunMode { CUBIC_ONLY, HEX_ONLY, BOTH };
constexpr RunMode RUN_MODE = BOTH;           // RUN_MODE = BOTH

constexpr int  WIDTHS[]  = {5, 10, 15, 20};  // channel widths w (both lattices)
constexpr int  N_WIDTHS  = sizeof(WIDTHS)/sizeof(WIDTHS[0]);

// ---- VALIDITY WINDOW  epsilon = Omega * v_par / a  <<  1 --------------------
// F_MAX is solved from the EXACT v_par(F) for the MOST RESTRICTIVE channel
// (largest Omega/q), so that every channel and every scanned point obeys
// epsilon <= EPS_MAX.  F_MIN = F_MAX * 10^(-F_DECADES).
constexpr double EPS_MAX    = 0.10;          // hard ceiling on Omega*v_par/a
constexpr double F_DECADES  = 2.0;           // decades of force covered

// ---- disorder / RNG --------------------------------------------------------
constexpr bool     NEW_DISORDER_PER_WALKER = true;   // fresh quenched sample per walker
constexpr uint64_t SEED     = 20240501ULL;   // trajectory stream seed
constexpr uint64_t DIS_SEED = 918273645ULL;  // quenched-disorder stream seed
                                             // (disorder seed is INDEPENDENT of F:
                                             //  common random environments across the
                                             //  force scan -> smooth curves)

// ---- output ----------------------------------------------------------------
constexpr const char* CSV_MAIN = "sim1_results.csv";
constexpr const char* CSV_META = "sim1_channels.csv";

// ============================ site-id packing ================================
constexpr int      TR_BITS = 21;             // transverse index bits (Omega < 2^21)
constexpr int64_t  XBIAS   = (int64_t)1 << 40;

// ================================ PRNG =======================================
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

struct Xoshiro256pp {
    uint64_t s[4];
    inline void seed(uint64_t z) { for (int i = 0; i < 4; ++i) { z = splitmix64(z); s[i] = z; } }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    inline double uniform() { return (next() >> 11) * 0x1.0p-53; }   // [0,1)
};

// ====================== quenched tau cache (open addressing) =================
// The lattice is infinite along x_par: nothing is preallocated.  tau_r is
// generated lazily from a stateless hash of the site id and cached, so a revisit
// costs exactly the same tau_r (QUENCHED) at ~2 ns instead of a pow() call.
struct TauCache {
    std::vector<uint64_t> key;
    std::vector<double>   val;
    std::vector<uint32_t> stamp;
    uint64_t mask; uint32_t epoch; size_t count, limit;

    explicit TauCache(size_t cap0 = (1u << 13)) { alloc(cap0); }

    void alloc(size_t cap) {
        key.assign(cap, 0ULL); val.assign(cap, 0.0); stamp.assign(cap, 0u);
        mask = cap - 1; epoch = 0; count = 0; limit = cap / 2;
    }
    inline void reset() {
        ++epoch; count = 0;
        if (epoch == 0) { std::fill(stamp.begin(), stamp.end(), 0u); epoch = 1; }
    }
    static inline uint64_t slot_hash(uint64_t id) {
        uint64_t h = id * 0x9E3779B97F4A7C15ULL; h ^= h >> 29;
        h *= 0xBF58476D1CE4E5B9ULL; return h ^ (h >> 32);
    }
    inline void insert_raw(uint64_t id, double v) {
        uint64_t i = slot_hash(id) & mask;
        while (stamp[i] == epoch) i = (i + 1) & mask;
        stamp[i] = epoch; key[i] = id; val[i] = v; ++count;
    }
    void grow() {
        std::vector<uint64_t> ok; std::vector<double> ov;
        ok.reserve(count); ov.reserve(count);
        for (size_t i = 0; i <= mask; ++i)
            if (stamp[i] == epoch) { ok.push_back(key[i]); ov.push_back(val[i]); }
        alloc((mask + 1) * 2); epoch = 1;
        for (size_t j = 0; j < ok.size(); ++j) insert_raw(ok[j], ov[j]);
    }
    template <class GEN>
    inline double get(uint64_t id, GEN&& gen) {
        uint64_t i = slot_hash(id) & mask;
        while (stamp[i] == epoch) { if (key[i] == id) return val[i]; i = (i + 1) & mask; }
        const double v = gen(id);
        stamp[i] = epoch; key[i] = id; val[i] = v; ++count;
        if (count > limit) grow();
        return v;
    }
};

// ============================== channel geometry =============================
// Both lattices are reduced to the SAME kernel object: a transverse adjacency
// table nb[s*ntrans + d] (-1 == wall, reflecting) plus 2 free longitudinal
// neighbours.  Cubic wraps (PBC, no -1 entries); hexagonal has -1 on the rim.
struct Channel {
    std::string lat;             // "cubic" | "hex"
    int    w = 0;
    int    q = 0;                // coordination number
    int    ntrans = 0;           // q - 2
    int    Omega = 0;            // sites in the cross-section
    double D0 = 0.0;             // a^2/q
    bool   reflecting = false;
    int    origin = 0;
    std::vector<int32_t> nb;
};

static Channel make_cubic(int w) {
    Channel c; c.lat = "cubic"; c.w = w; c.q = 6; c.ntrans = 4;
    c.Omega = w * w; c.D0 = LAT_A * LAT_A / 6.0; c.reflecting = false;
    c.nb.assign((size_t)c.Omega * 4, -1);
    auto idx = [w](int y, int z) { return y * w + z; };
    for (int y = 0; y < w; ++y) for (int z = 0; z < w; ++z) {
        const int s = idx(y, z);
        c.nb[(size_t)s * 4 + 0] = idx((y + 1) % w, z);
        c.nb[(size_t)s * 4 + 1] = idx((y - 1 + w) % w, z);
        c.nb[(size_t)s * 4 + 2] = idx(y, (z + 1) % w);
        c.nb[(size_t)s * 4 + 3] = idx(y, (z - 1 + w) % w);
    }
    c.origin = idx(0, 0);
    return c;
}

// triangular layer, hexagonal patch of R = w-1 rings  ->  Omega = 3w^2-3w+1
static Channel make_hex(int w) {
    Channel c; c.lat = "hex"; c.w = w; c.q = 8; c.ntrans = 6;
    c.D0 = LAT_A * LAT_A / 8.0; c.reflecting = true;
    const int R = w - 1, S = 2 * R + 1;
    std::vector<int32_t> map((size_t)S * S, -1);
    std::vector<int> uu, vv;
    for (int u = -R; u <= R; ++u) for (int v = -R; v <= R; ++v) {
        if (std::abs(u + v) > R) continue;
        map[(size_t)(u + R) * S + (v + R)] = (int32_t)uu.size();
        uu.push_back(u); vv.push_back(v);
    }
    c.Omega = (int)uu.size();                    // = 3w^2 - 3w + 1
    c.nb.assign((size_t)c.Omega * 6, -1);
    const int du[6] = { 1, -1, 0,  0,  1, -1 };
    const int dv[6] = { 0,  0, 1, -1, -1,  1 };
    for (int s = 0; s < c.Omega; ++s) for (int d = 0; d < 6; ++d) {
        const int u = uu[s] + du[d], v = vv[s] + dv[d];
        if (std::abs(u) > R || std::abs(v) > R || std::abs(u + v) > R) continue;  // wall
        c.nb[(size_t)s * 6 + d] = map[(size_t)(u + R) * S + (v + R)];
    }
    c.origin = map[(size_t)R * S + R];
    return c;
}

// ============================== exact lattice constants ======================
static inline double v_par_exact(double F, int q) {          // Sec. 4.2
    return 2.0 * LAT_A * std::sinh(F * LAT_A * 0.5) /
           (2.0 * std::cosh(F * LAT_A * 0.5) + (double)(q - 2));
}
static inline double D_par_exact(double F, int q) {
    return LAT_A * LAT_A * std::cosh(F * LAT_A * 0.5) /
           (2.0 * std::cosh(F * LAT_A * 0.5) + (double)(q - 2));
}
// largest F with Omega*v_par(F)/a <= eps
static double F_for_eps(int q, int Omega, double eps) {
    double lo = 1e-14, hi = 10.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if ((double)Omega * v_par_exact(mid, q) / LAT_A > eps) hi = mid; else lo = mid;
    }
    return lo;
}

// ================================ kernel =====================================
struct Result { double mean, se, njumps, vpar, eps; };

template <bool REFLECT>
static Result run_point(const Channel& ch, double F, uint64_t chid, double tau0) {
    const int      nt  = ch.ntrans;
    const int32_t* nb  = ch.nb.data();
    const double   h   = 0.5 * F * LAT_A;
    const double   Z   = 2.0 * std::cosh(h) + (double)(ch.q - 2);
    const double   c1  = std::exp(h)  / Z;                 // p(+e_par)
    const double   c2  = c1 + std::exp(-h) / Z;            // + p(-e_par)
    const double   nia = -1.0 / ALPHA;
    const int      org = ch.origin;

    double sx = 0.0, sx2 = 0.0, sn = 0.0;

#pragma omp parallel reduction(+:sx,sx2,sn)
    {
        TauCache cache(1u << 13);
        Xoshiro256pp rng;
#pragma omp for schedule(dynamic, 64)
        for (long long iw = 0; iw < N_WALKERS; ++iw) {
            cache.reset();
            rng.seed(splitmix64(SEED ^ (chid * 0x9E3779B97F4A7C15ULL)
                                     ^ ((uint64_t)(iw + 1) * 0xD1B54A32D192ED03ULL)));
            const uint64_t dseed = NEW_DISORDER_PER_WALKER
                ? splitmix64(DIS_SEED ^ (chid * 0xC2B2AE3D27D4EB4FULL)
                                     ^ ((uint64_t)(iw + 1) * 0x9E3779B97F4A7C15ULL))
                : splitmix64(DIS_SEED ^ (chid * 0xC2B2AE3D27D4EB4FULL));

            int64_t x = 0; int s = org; double elapsed = 0.0, nj = 0.0;

            for (;;) {
                const uint64_t id = ((uint64_t)(x + XBIAS) << TR_BITS) | (uint64_t)s;
                // QUENCHED: charged on arrival, identical on every revisit
                const double tau = cache.get(id, [&](uint64_t k) {
                    const uint64_t hh = splitmix64(k ^ dseed);
                    const double   U  = (double)((hh >> 11) + 1ULL) * 0x1.0p-53;  // (0,1]
                    return tau0 * std::pow(U, nia);                                // Pareto
                });
                elapsed += tau;
                if (elapsed >= T_MAX) break;

                const double u = rng.uniform();
                if (u < c1)      ++x;                       // longitudinal, unbounded
                else if (u < c2) --x;
                else {
                    int d = (int)((u - c2) * Z);
                    if (d >= nt) d = nt - 1;
                    const int32_t s2 = nb[(size_t)s * nt + d];
                    if (REFLECT) { if (s2 >= 0) s = s2; }   // wall: attempt consumed, stay
                    else           s = (int)s2;             // PBC: always valid
                }
                nj += 1.0;
            }
            sx += (double)x; sx2 += (double)x * (double)x; sn += nj;
        }
    }

    const double Nd  = (double)N_WALKERS;
    const double m   = sx / Nd;
    const double var = std::max(0.0, (sx2 - Nd * m * m) / (Nd - 1.0));
    Result r;
    r.mean = m; r.se = std::sqrt(var / Nd); r.njumps = sn / Nd;
    r.vpar = v_par_exact(F, ch.q);
    r.eps  = (double)ch.Omega * r.vpar / LAT_A;
    return r;
}

// ================================== main =====================================
int main() {
    const double G1a  = std::tgamma(1.0 + ALPHA);
    const double A_al = AMP_A * G1a * G1a;                    // A_alpha
    const double tau0 = std::pow(std::tgamma(1.0 - ALPHA), -1.0 / ALPHA)
                      * std::pow(AMP_A, 1.0 / ALPHA);         // A = 1 -> Sec. 2 table
    const double tpow = std::pow(T_MAX, ALPHA);

    // ---- build channels ----
    std::vector<Channel> chans;
    if (RUN_MODE == CUBIC_ONLY || RUN_MODE == BOTH)
        for (int i = 0; i < N_WIDTHS; ++i) chans.push_back(make_cubic(WIDTHS[i]));
    if (RUN_MODE == HEX_ONLY || RUN_MODE == BOTH)
        for (int i = 0; i < N_WIDTHS; ++i) chans.push_back(make_hex(WIDTHS[i]));
    for (const Channel& c : chans)
        if (c.Omega >= (1 << TR_BITS)) { std::fprintf(stderr, "Omega too large\n"); return 1; }

    // ---- validity window: F_MAX from the most restrictive channel ----
    double F_MAX = 1e30;
    const Channel* bind = nullptr;
    for (const Channel& c : chans) {
        const double f = F_for_eps(c.q, c.Omega, EPS_MAX);
        if (f < F_MAX) { F_MAX = f; bind = &c; }
    }
    const double F_MIN = F_MAX * std::pow(10.0, -F_DECADES);

    std::vector<double> Fgrid(POINTS);
    for (int i = 0; i < POINTS; ++i)
        Fgrid[i] = F_MIN * std::pow(F_MAX / F_MIN, (double)i / (double)(POINTS - 1));

    std::printf("SIM 1 : <x_par(t)> vs F_par\n");
    std::printf("  N=%lld  t=%.3e  alpha=%.3f  A=%.3f  tau0=%.6f  POINTS=%d\n",
                N_WALKERS, T_MAX, ALPHA, AMP_A, tau0, POINTS);
    std::printf("  validity  eps = Omega*v_par/a <= %.3f  (binding: %s w=%d, Omega=%d)\n",
                EPS_MAX, bind->lat.c_str(), bind->w, bind->Omega);
    std::printf("  F range   [%.6e , %.6e]  (%.1f decades)\n", F_MIN, F_MAX, F_DECADES);
#ifdef _OPENMP
    std::printf("  threads   %d\n", omp_get_max_threads());
#endif
    std::fflush(stdout);

    // ---- channel metadata ----
    {
        FILE* fm = std::fopen(CSV_META, "w");
        std::fprintf(fm, "lattice,w,q,Omega,D0,boundary,ntrans\n");
        for (const Channel& c : chans)
            std::fprintf(fm, "%s,%d,%d,%d,%.12g,%s,%d\n", c.lat.c_str(), c.w, c.q, c.Omega,
                         c.D0, c.reflecting ? "reflecting" : "periodic", c.ntrans);
        std::fclose(fm);
    }

    // ---- scan ----
    std::vector<std::vector<Result>> R(chans.size(), std::vector<Result>(POINTS));
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t ic = 0; ic < chans.size(); ++ic) {
        const Channel& c = chans[ic];
        const uint64_t chid = splitmix64(0xABCDEF01ULL + 1000ULL * (uint64_t)c.w
                                         + (c.lat == "hex" ? 7ULL : 3ULL));
        for (int ip = 0; ip < POINTS; ++ip) {
            const double F = Fgrid[ip];
            R[ic][ip] = c.reflecting ? run_point<true >(c, F, chid, tau0)
                                     : run_point<false>(c, F, chid, tau0);
            const double el = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0).count();
            std::printf("  [%s w=%2d Om=%4d] F=%.4e  <x>=%.6e +- %.2e  eps=%.4f  "
                        "<N>=%.3e  (%.1f s)\n",
                        c.lat.c_str(), c.w, c.Omega, F, R[ic][ip].mean, R[ic][ip].se,
                        R[ic][ip].eps, R[ic][ip].njumps, el);
            std::fflush(stdout);
        }
    }

    // ---- wide CSV ----
    FILE* f = std::fopen(CSV_MAIN, "w");
    std::fprintf(f, "F");
    for (const Channel& c : chans) {
        const char* L = c.lat.c_str();
        std::fprintf(f, ",x_%s_w%d,se_%s_w%d,eps_%s_w%d,njumps_%s_w%d,xth_%s_w%d",
                     L, c.w, L, c.w, L, c.w, L, c.w, L, c.w);
    }
    std::fprintf(f, "\n");
    for (int ip = 0; ip < POINTS; ++ip) {
        std::fprintf(f, "%.10g", Fgrid[ip]);
        for (size_t ic = 0; ic < chans.size(); ++ic) {
            const Channel& c = chans[ic];
            // Eq. (35): <x> = (D0*F)^a / A_alpha * (a/Omega)^(1-a) * t^a
            const double xth = std::pow(c.D0 * Fgrid[ip], ALPHA) / A_al
                             * std::pow(LAT_A / (double)c.Omega, 1.0 - ALPHA) * tpow;
            std::fprintf(f, ",%.10g,%.10g,%.10g,%.10g,%.10g",
                         R[ic][ip].mean, R[ic][ip].se, R[ic][ip].eps,
                         R[ic][ip].njumps, xth);
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);

    std::printf("wrote %s and %s\n", CSV_MAIN, CSV_META);
    return 0;
}