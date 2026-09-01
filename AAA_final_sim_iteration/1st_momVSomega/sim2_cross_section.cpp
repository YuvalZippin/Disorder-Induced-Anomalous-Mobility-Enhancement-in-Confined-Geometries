// =====================================================================================
//  SIM 2 : First moment <x_par(t)> as a function of CROSS-SECTION AREA Omega
//  Quenched Trap Model in a restricted channel.
//    * Simple cubic  (q=6), square section, PERIODIC transverse BC,   Omega = w^2
//    * Simple hexagonal (q=8), hexagonal section, REFLECTING BC,      Omega = 3w^2-3w+1
//  Independent variable : Omega (scanned over POINTS widths). Force F is FIXED.
//  Output               : sim2_results.csv  (+ sim2_params.json)
// =====================================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// =====================================================================================
//  ================================  PARAMETERS  =====================================
// =====================================================================================

enum RunMode { CUBIC_ONLY, HEX_ONLY, BOTH };

constexpr RunMode   RUN_MODE   = BOTH;          // run both geometries
constexpr long long N_WALKERS  = 100000LL;      // N   = 1e5
constexpr double    T_MAX      = 1.0e14;        // t   = 1e14  (STRICT, never rescaled)
constexpr double    ALPHA      = 0.3;           // disorder index
constexpr double    A_AMP      = 1.0;           // amplitude A
constexpr double    F_VALS[]   = { 5.0e-4, 1.0e-3 };  // FIXED forces (outer loop)
constexpr int       N_F        = int(sizeof(F_VALS) / sizeof(F_VALS[0]));
constexpr int       POINTS     = 16;            // number of cross-section points
constexpr int       W_MIN      = 1;             // smallest transverse width
constexpr int       W_MAX      = 16;            // largest  transverse width
constexpr double    EPS_MAX    = 0.10;          // validity window: eps = Omega*v_par/a
constexpr uint64_t  SEED       = 20260901ULL;   // master seed
constexpr int       CACHE_BITS = 13;            // per-walker quenched-disorder cache
constexpr const char* OUT_CSV  = "sim2_results.csv";
constexpr const char* OUT_JSON = "sim2_params.json";

// Cross-section ranges produced by (W_MIN..W_MAX):
//   cubic : Omega = w^2        = 1 ... 256
//   hex   : Omega = 3w^2-3w+1  = 1 ... 721
// Validity  eps = Omega*v_par/a  at the two forces (binding case = hex, w = W_MAX):
//   F = 5.0e-4 : eps_max = 0.0451      F = 1.0e-3 : eps_max = 0.0901
// Both stay inside EPS_MAX. The largest force fixes W_MAX:
//   F_max(Omega) = EPS_MAX * a / (Omega * D_0).

// =====================================================================================

constexpr int CACHE_SIZE = 1 << CACHE_BITS;

struct CacheEntry { uint64_t tag; double tau; };

static inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Xoshiro {
    uint64_t s[4];
    explicit Xoshiro(uint64_t seed) {
        uint64_t z = seed;
        for (int i = 0; i < 4; ++i) { z += 0x9E3779B97F4A7C15ULL; s[i] = mix64(z); }
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;    s[3] = rotl(s[3], 45);
        return r;
    }
    inline double uni() { return double(next() >> 11) * 0x1.0p-53; }   // [0,1)
};

// ---- quenched waiting time of a site: stateless hash + per-walker direct-mapped cache
static inline double tau_at(uint64_t pk, uint64_t wkey, CacheEntry* cache,
                            double tau0, double nia)
{
    const uint64_t idx = (pk * 0x9E3779B97F4A7C15ULL) >> (64 - CACHE_BITS);
    CacheEntry& e = cache[idx];
    if (e.tag == pk) return e.tau;                       // quenched: identical on revisit
    uint64_t h = mix64(wkey ^ (pk * 0xD6E8FEB86659FD93ULL));
    h = mix64(h);
    const double u = double((h >> 11) + 1) * 0x1.0p-53;  // (0,1]
    const double t = tau0 * std::exp2(nia * std::log2(u));
    e.tag = pk; e.tau = t;
    return t;
}

// =====================================================================================
//  LATTICES
// =====================================================================================

struct CubicLattice {
    static constexpr int    Q    = 6;
    static constexpr double D0   = 1.0 / 6.0;
    static constexpr const char* NAME = "cubic";
    int w; long long Omega;
    explicit CubicLattice(int w_) : w(w_), Omega((long long)w_ * (long long)w_) {}

    struct State { int64_t x; int32_t y, z; };
    inline void init(State& s) const { s.x = 0; s.y = 0; s.z = 0; }

    inline void move(State& s, int dir) const {                 // PERIODIC transverse BC
        switch (dir) {
            case 0: ++s.x; break;
            case 1: --s.x; break;
            case 2: s.y = (s.y + 1 == w) ? 0     : s.y + 1; break;
            case 3: s.y = (s.y == 0)     ? w - 1 : s.y - 1; break;
            case 4: s.z = (s.z + 1 == w) ? 0     : s.z + 1; break;
            default:s.z = (s.z == 0)     ? w - 1 : s.z - 1; break;
        }
    }
    inline uint64_t pack(const State& s) const {
        return ((uint64_t)s.x << 20) ^ (uint64_t)(uint32_t)(s.y * w + s.z);
    }
};

struct HexLattice {
    static constexpr int    Q    = 8;
    static constexpr double D0   = 1.0 / 8.0;
    static constexpr const char* NAME = "hexagonal";
    static constexpr int DI[6] = { 1, -1, 0,  0,  1, -1 };
    static constexpr int DJ[6] = { 0,  0, 1, -1, -1,  1 };
    int w, R, side; long long Omega;
    explicit HexLattice(int w_)
        : w(w_), R(w_ - 1), side(2 * w_ - 1),
          Omega(3LL * w_ * w_ - 3LL * w_ + 1LL) {}      // actual site count of the patch

    struct State { int64_t x; int32_t i, j; };
    inline void init(State& s) const { s.x = 0; s.i = 0; s.j = 0; }

    inline void move(State& s, int dir) const {                 // REFLECTING transverse BC
        if (dir == 0) { ++s.x; return; }
        if (dir == 1) { --s.x; return; }
        const int k  = dir - 2;
        const int ni = s.i + DI[k], nj = s.j + DJ[k];
        const int d  = (std::abs(ni) + std::abs(nj) + std::abs(ni + nj)) >> 1;
        if (d <= R) { s.i = ni; s.j = nj; }   // else: attempt consumed, walker stays put
    }
    inline uint64_t pack(const State& s) const {
        return ((uint64_t)s.x << 20) ^ (uint64_t)(uint32_t)((s.i + R) * side + (s.j + R));
    }
};

// =====================================================================================
//  WALKER KERNEL
// =====================================================================================

template <class Lat>
static inline void run_walker(const Lat& lat, double c1, double c2, double inv_pt,
                              int Qm1, double tau0, double nia,
                              uint64_t wkey, Xoshiro& rng, CacheEntry* cache,
                              double& out_x, double& out_n)
{
    typename Lat::State s; lat.init(s);
    uint64_t key = lat.pack(s);
    double   tau = tau_at(key, wkey, cache, tau0, nia);
    double   elapsed = 0.0;
    uint64_t n = 0;

    for (;;) {
        elapsed += tau;                       // time charged on ARRIVAL, quenched value
        if (elapsed >= T_MAX) break;
        const double r = rng.uni();
        int dir = (r < c1) ? 0 : ((r < c2) ? 1 : 2 + int((r - c2) * inv_pt));
        if (dir > Qm1) dir = Qm1;
        lat.move(s, dir);
        ++n;
        const uint64_t nk = lat.pack(s);
        if (nk != key) { key = nk; tau = tau_at(key, wkey, cache, tau0, nia); }
    }
    out_x = double(s.x);
    out_n = double(n);
}

// =====================================================================================
//  POINT RUNNER
// =====================================================================================

struct PointResult {
    int w; long long Omega;
    double F, v_par, eps, mean_x, sem_x, mean_N, theory, seconds;
};

template <class Lat>
static PointResult run_point(int w, double F, uint64_t seed, uint64_t task_id)
{
    const Lat lat(w);
    constexpr int Q = Lat::Q;

    const double zn      = 2.0 * std::cosh(0.5 * F) + double(Q - 2);
    const double p_plus  = std::exp( 0.5 * F) / zn;
    const double p_minus = std::exp(-0.5 * F) / zn;
    const double p_tr    = 1.0 / zn;
    const double c1      = p_plus;
    const double c2      = p_plus + p_minus;
    const double inv_pt  = 1.0 / p_tr;
    const double v_par   = 2.0 * std::sinh(0.5 * F) / zn;   // exact drift per jump

    const double tau0 = std::pow(A_AMP / std::tgamma(1.0 - ALPHA), 1.0 / ALPHA);
    const double nia  = -1.0 / ALPHA;

    double sx = 0.0, sx2 = 0.0, sn = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

#pragma omp parallel reduction(+ : sx, sx2, sn)
    {
        std::vector<CacheEntry> cache(CACHE_SIZE);
#pragma omp for schedule(dynamic, 16)
        for (long long iw = 0; iw < N_WALKERS; ++iw) {
            std::memset(cache.data(), 0xFF, size_t(CACHE_SIZE) * sizeof(CacheEntry));
            const uint64_t wkey = mix64(seed
                                        + 0x100000001B3ULL * task_id
                                        + 0x9E3779B97F4A7C15ULL * (uint64_t)iw);
            Xoshiro rng(mix64(wkey ^ 0xA5A5A5A5DEADBEEFULL));
            double x = 0.0, n = 0.0;
            run_walker(lat, c1, c2, inv_pt, Q - 1, tau0, nia, wkey, rng, cache.data(), x, n);
            sx += x; sx2 += x * x; sn += n;
        }
    }

    const double Nd    = double(N_WALKERS);
    const double mean  = sx / Nd;
    const double var   = std::max(0.0, sx2 / Nd - mean * mean);
    const double A_alp = A_AMP * std::tgamma(1.0 + ALPHA) * std::tgamma(1.0 + ALPHA);

    PointResult r;
    r.w       = w;
    r.Omega   = lat.Omega;
    r.F       = F;
    r.v_par   = v_par;
    r.eps     = double(lat.Omega) * v_par;                     // a = 1
    r.mean_x  = mean;
    r.sem_x   = std::sqrt(var / (Nd - 1.0));
    r.mean_N  = sn / Nd;
    r.theory  = std::pow(v_par, ALPHA)
              * std::pow(1.0 / double(lat.Omega), 1.0 - ALPHA)
              * std::pow(T_MAX, ALPHA) / A_alp;                // Eq. (34)/(35)
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

// =====================================================================================
//  DRIVER
// =====================================================================================

static std::vector<int> build_widths()
{
    std::set<int> s;
    for (int k = 0; k < POINTS; ++k) {
        const double f = (POINTS == 1) ? 0.0 : double(k) / double(POINTS - 1);
        int w = int(std::llround(double(W_MIN) * std::pow(double(W_MAX) / double(W_MIN), f)));
        s.insert(std::min(std::max(w, W_MIN), W_MAX));
    }
    for (int w = W_MIN; w <= W_MAX && int(s.size()) < POINTS; ++w) s.insert(w);
    return std::vector<int>(s.begin(), s.end());
}

static double vpar_of(double F, int q) {
    return 2.0 * std::sinh(0.5 * F) / (2.0 * std::cosh(0.5 * F) + double(q - 2));
}

int main()
{
    const std::vector<int> W = build_widths();

    // ---------------- validity window enforced BEFORE any cycle is burnt ----------------
    bool ok = true;
    double worst_F = 0.0;
    for (int fi = 0; fi < N_F; ++fi) {
        const double F = F_VALS[fi];
        worst_F = std::max(worst_F, F);
        for (int w : W) {
            if (RUN_MODE != HEX_ONLY) {
                const double e = double(w) * double(w) * vpar_of(F, 6);
                if (e > EPS_MAX) { std::fprintf(stderr,
                    "[VALIDITY] F=%.6g cubic w=%d : eps=%.4g > EPS_MAX=%.3g\n",
                    F, w, e, EPS_MAX); ok = false; }
            }
            if (RUN_MODE != CUBIC_ONLY) {
                const double e = double(3 * w * w - 3 * w + 1) * vpar_of(F, 8);
                if (e > EPS_MAX) { std::fprintf(stderr,
                    "[VALIDITY] F=%.6g hex   w=%d : eps=%.4g > EPS_MAX=%.3g\n",
                    F, w, e, EPS_MAX); ok = false; }
            }
        }
    }
    if (!ok) {
        const long long OmMax = (RUN_MODE == CUBIC_ONLY)
                              ? (long long)W_MAX * W_MAX
                              : 3LL * W_MAX * W_MAX - 3LL * W_MAX + 1LL;
        const double D0min = (RUN_MODE == CUBIC_ONLY) ? 1.0 / 6.0 : 1.0 / 8.0;
        std::fprintf(stderr,
            "ABORT: largest force %.6g exceeds F_max=%.4g at W_MAX=%d. "
            "Lower max(F_VALS) or lower W_MAX.\n",
            worst_F, EPS_MAX / (double(OmMax) * D0min), W_MAX);
        return 1;
    }

    std::fprintf(stderr,
        "SIM2  alpha=%.3f  A=%.3f  t=%.3g  N=%lld  widths=%zu  forces=%d  threads=%d\n",
        ALPHA, A_AMP, T_MAX, N_WALKERS, W.size(), N_F,
#ifdef _OPENMP
        omp_get_max_threads()
#else
        1
#endif
    );

    // results[fi] = vector over widths
    std::vector<std::vector<PointResult>> RC(N_F), RH(N_F);

    for (int fi = 0; fi < N_F; ++fi) {                       // ---- OUTER LOOP: force ----
        const double F = F_VALS[fi];
        std::fprintf(stderr, "--- F_par = %.6g ---\n", F);
        for (size_t k = 0; k < W.size(); ++k) {              // ---- scan cross-section ----
            if (RUN_MODE != HEX_ONLY) {
                PointResult r = run_point<CubicLattice>(
                    W[k], F, SEED, 1000000ULL + 1000ULL * uint64_t(fi) + k);
                RC[fi].push_back(r);
                std::fprintf(stderr,
                    "  F=%.4g cubic w=%2d Om=%5lld eps=%.4g <x>=%.6g +-%.3g <N>=%.4g [%.1fs]\n",
                    F, r.w, r.Omega, r.eps, r.mean_x, r.sem_x, r.mean_N, r.seconds);
            }
            if (RUN_MODE != CUBIC_ONLY) {
                PointResult r = run_point<HexLattice>(
                    W[k], F, SEED, 2000000ULL + 1000ULL * uint64_t(fi) + k);
                RH[fi].push_back(r);
                std::fprintf(stderr,
                    "  F=%.4g hex   w=%2d Om=%5lld eps=%.4g <x>=%.6g +-%.3g <N>=%.4g [%.1fs]\n",
                    F, r.w, r.Omega, r.eps, r.mean_x, r.sem_x, r.mean_N, r.seconds);
            }
        }
    }

    // ---------------------------------- CSV ----------------------------------
    FILE* f = std::fopen(OUT_CSV, "w");
    if (!f) { std::perror("fopen"); return 1; }
    std::fprintf(f, "Force,w,"
                    "Omega_cubic,eps_cubic,vpar_cubic,x_cubic,x_cubic_sem,x_cubic_theory,Njumps_cubic,"
                    "Omega_hex,eps_hex,vpar_hex,x_hex,x_hex_sem,x_hex_theory,Njumps_hex\n");
    for (int fi = 0; fi < N_F; ++fi) {
        for (size_t k = 0; k < W.size(); ++k) {
            std::fprintf(f, "%.10g,%d,", F_VALS[fi], W[k]);
            if (RUN_MODE != HEX_ONLY) {
                const PointResult& r = RC[fi][k];
                std::fprintf(f, "%lld,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,",
                             r.Omega, r.eps, r.v_par, r.mean_x, r.sem_x, r.theory, r.mean_N);
            } else std::fprintf(f, "nan,nan,nan,nan,nan,nan,nan,");
            if (RUN_MODE != CUBIC_ONLY) {
                const PointResult& r = RH[fi][k];
                std::fprintf(f, "%lld,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                             r.Omega, r.eps, r.v_par, r.mean_x, r.sem_x, r.theory, r.mean_N);
            } else std::fprintf(f, "nan,nan,nan,nan,nan,nan,nan\n");
        }
    }
    std::fclose(f);

    // --------------------------------- params --------------------------------
    FILE* g = std::fopen(OUT_JSON, "w");
    if (g) {
        std::fprintf(g,
            "{\n  \"sim\": \"sim2_cross_section\",\n"
            "  \"alpha\": %.10g,\n  \"A\": %.10g,\n  \"A_alpha\": %.10g,\n"
            "  \"F_vals\": [", ALPHA, A_AMP,
            A_AMP * std::tgamma(1.0 + ALPHA) * std::tgamma(1.0 + ALPHA));
        for (int fi = 0; fi < N_F; ++fi)
            std::fprintf(g, "%s%.10g", (fi ? ", " : ""), F_VALS[fi]);
        std::fprintf(g,
            "],\n  \"t\": %.10g,\n  \"N_walkers\": %lld,\n"
            "  \"points\": %zu,\n  \"eps_max\": %.10g,\n  \"a\": 1,\n"
            "  \"D0_cubic\": %.10g,\n  \"D0_hex\": %.10g,\n"
            "  \"q_cubic\": 6,\n  \"q_hex\": 8,\n"
            "  \"bc_cubic\": \"periodic\",\n  \"bc_hex\": \"reflecting\",\n"
            "  \"Omega_cubic\": \"w^2\",\n  \"Omega_hex\": \"3w^2-3w+1\",\n"
            "  \"seed\": %llu\n}\n",
            T_MAX, N_WALKERS, W.size(), EPS_MAX,
            1.0 / 6.0, 1.0 / 8.0, (unsigned long long)SEED);
        std::fclose(g);
    }

    std::fprintf(stderr, "wrote %s and %s\n", OUT_CSV, OUT_JSON);
    return 0;
}