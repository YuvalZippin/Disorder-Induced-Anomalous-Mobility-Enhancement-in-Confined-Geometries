// =====================================================================
//  SIM 3 :  <x_par(t)>  as a function of CROSS-SECTION AREA
//           Anisotropic lattices:
//              2D rectangular   (a = longitudinal, b = transverse)
//              3D orthorhombic  (a = longitudinal, b,c = transverse)
//           Quenched Trap Model. Transverse PBC. Force along the a-axis.
//
//  ULTRA-OPTIMIZED BUILD
//   * per-thread memo cache lives on the thread stack  -> no false sharing
//   * 64B-aligned 4-slot groups                        -> 1 cache line / hop
//   * branchless direction + branchless PBC wrap       -> no mispredicts
//   * software prefetch of the next site               -> hides L2/L3 latency
//   * int64->double conversions only                   -> single cvtsi2sd
//   * Kahan-compensated 1e14 clock  (DO NOT BUILD WITH -ffast-math)
//
//  VALIDITY GUARD (both enforced, see banner printed at startup)
//   (V3) near recurrence : eps = Omega*v_par/a <= EPS_MAX
//   (V6) drift dominance : <N(t)> >= RATIO_MIN * N*,  N* = 4*D_par/v_par^2
//        Q_0 only saturates at 1 - Omega*v_par/a once the walk is drift
//        dominated; below that the walk is still in the recurrent crossover
//        and Eq.(35) is not yet reached.

// g++ -std=c++17 -O3 -march=native -mtune=native -funroll-loops -flto=auto \
    -fno-math-errno -fno-semantic-interposition -fomit-frame-pointer \
    -falign-functions=64 -fopenmp -DNDEBUG \
    -o sim3 sim3_anisotropic.cpp -lm

// run: ./sim3
// =====================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

// ========================== PARAMETERS ===============================
namespace par {
constexpr long long   N_WALKERS  = 100000LL;   // N   = 1e5
constexpr double      T_MAX      = 1.0e14;     // t   = 1e14  (STRICT, never rescaled)
constexpr double      ALPHA      = 0.30;       // alpha
constexpr double      A_AMP      = 1.0;        // A
constexpr double      FORCE      = 5.0e-2;     // F_par = f/T  (fixed for this scan)
constexpr double      LAT_A      = 1.0;        // a  (longitudinal / open axis)
constexpr double      LAT_B      = 2.0;        // b  (transverse 1)
constexpr double      LAT_C      = 1.5;        // c  (transverse 2, 3D only)
constexpr int         POINTS     = 20;         // requested points (clamped to window)
constexpr int         OMEGA_MIN  = 1;          // smallest cross-section (sites)
constexpr double      EPS_MAX    = 0.10;       // (V3)  eps = Omega*v_par/a <= EPS_MAX
constexpr double      RATIO_MIN  = 10.0;       // (V6)  <N(t)> / N*  >= RATIO_MIN
constexpr uint64_t    SEED       = 0x5EEDC0FFEE2024ULL;
constexpr const char* OUT_CSV    = "sim3_results.csv";
enum Mode { MODE_2D = 0, MODE_3D = 1, MODE_BOTH = 2 };
constexpr Mode        RUN_MODE   = MODE_BOTH;  // RUN_MODE = BOTH
constexpr int         GROUP_BITS = 16;         // memo table = 2^16 lines = 4 MB / thread
}

// ========================== DERIVED ==================================
static const double INV_ALPHA = 1.0 / par::ALPHA;
// A = Gamma(1-alpha)*tau0^alpha  ->  tau0 = ( A / Gamma(1-alpha) )^(1/alpha)
static const double TAU0    = std::pow(par::A_AMP / std::tgamma(1.0 - par::ALPHA), INV_ALPHA);
static const double G1PA    = std::tgamma(1.0 + par::ALPHA);
static const double A_ALPHA = par::A_AMP * G1PA * G1PA;          // A_alpha = A*Gamma^2(1+alpha)
static const double TPOWA   = std::pow(par::T_MAX, par::ALPHA);

#define HOT __attribute__((hot, flatten))
#define LIKELY(x) __builtin_expect(!!(x), 1)

// ========================== RNG ======================================
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

struct Xoshiro {                                  // xoshiro256+
    uint64_t s[4];
    explicit Xoshiro(uint64_t seed) {
        for (int i = 0; i < 4; ++i) { seed = splitmix64(seed); s[i] = seed; }
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = s[0] + s[3];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;    s[3] = rotl(s[3], 45);
        return r;
    }
    // cast via int64_t: single cvtsi2sd, not the unsigned fixup sequence
    inline double uni() { return (double)(int64_t)(next() >> 11) * 0x1.0p-53; }
};

// ============ QUENCHED WAITING TIME: authoritative, stateless ========
// tau_r = tau0 * U^(-1/alpha),  U = deterministic function of (site key, disorder seed).
// Cold path: reached only on a memo miss.
static double site_tau(uint64_t site_key, uint64_t dseed) {
    uint64_t h = splitmix64(site_key ^ (dseed * 0x9E3779B97F4A7C15ULL));
    h = splitmix64(h ^ 0xD6E8FEB86659FD93ULL);
    const double u = (double)(int64_t)((h >> 11) + 1ULL) * (1.0 / 9007199254740993.0); // (0,1)
    return TAU0 * std::pow(u, -INV_ALPHA);
}

// =================== PER-THREAD MEMO CACHE ===========================
// 64-bit stamp layout (injective -> a hit is always the right site):
//     bits 56..63 : walker generation (8 bit, table wiped on wrap)
//     bits 24..55 : x + 2^31          (32 bit, full int32 range)
//     bits 12..23 : y                 (12 bit, wy < 4096 enforced)
//     bits  0..11 : z                 (12 bit, wz < 4096 enforced)
// One group = one 64-byte cache line = 4 slots. Every lookup touches
// exactly one line; the 4-way probe never leaves it.
struct Slot { uint64_t stamp; double tau; };
struct alignas(64) SlotGroup { Slot s[4]; };

static constexpr uint64_t GEN_MASK = 0xFF00000000000000ULL;
static constexpr uint64_t KEY_MASK = 0x00FFFFFFFFFFFFFFULL;
static constexpr int      GSHIFT   = 64 - par::GROUP_BITS;

static inline uint64_t make_stamp(uint64_t genbits, int32_t x, int32_t y, int32_t z) {
    return genbits
         | ((uint64_t)(uint32_t)(x + (int32_t)0x80000000) << 24)
         | ((uint64_t)(uint32_t)y << 12)
         |  (uint64_t)(uint32_t)z;
}
static inline size_t grp_of(uint64_t stamp) {
    return (size_t)((stamp * 0x9E3779B97F4A7C15ULL) >> GSHIFT);   // multiply-shift, 2 ops
}

struct Memo {
    std::vector<SlotGroup> tab;      // heap buffer, first-touched by the owning thread
    uint64_t genbits = 0;
    uint32_t gen     = 0;
    void init() {
        tab.assign((size_t)1 << par::GROUP_BITS, SlotGroup{});
        gen = 0; genbits = 0;
    }
    inline void new_walker() {
        if (++gen == 256u) { std::memset(tab.data(), 0, tab.size() * sizeof(SlotGroup)); gen = 1; }
        genbits = (uint64_t)gen << 56;
    }
    inline double fetch(size_t g, uint64_t stamp, uint64_t dseed) {
        Slot* __restrict s = tab[g].s;
        if (LIKELY(s[0].stamp == stamp)) return s[0].tau;
        if ((s[0].stamp & GEN_MASK) != genbits) { const double t = site_tau(stamp & KEY_MASK, dseed); s[0].stamp = stamp; s[0].tau = t; return t; }
        if (s[1].stamp == stamp) return s[1].tau;
        if ((s[1].stamp & GEN_MASK) != genbits) { const double t = site_tau(stamp & KEY_MASK, dseed); s[1].stamp = stamp; s[1].tau = t; return t; }
        if (s[2].stamp == stamp) return s[2].tau;
        if ((s[2].stamp & GEN_MASK) != genbits) { const double t = site_tau(stamp & KEY_MASK, dseed); s[2].stamp = stamp; s[2].tau = t; return t; }
        if (s[3].stamp == stamp) return s[3].tau;
        if ((s[3].stamp & GEN_MASK) != genbits) { const double t = site_tau(stamp & KEY_MASK, dseed); s[3].stamp = stamp; s[3].tau = t; return t; }
        return site_tau(stamp & KEY_MASK, dseed);   // line full: recompute, still exact
    }
};

// ========================== CHANNEL ==================================
template<int DIM>
struct Channel {
    static constexpr int NB = 2 * DIM;
    int32_t wy = 1, wz = 1;
    double  cum[6];
    double  v_par = 0, D_par = 0, Dy = 0, Dz = 0;
    double  Omega = 0, area = 0, eps = 0;

    void build(int wy_, int wz_) {
        wy = (int32_t)wy_; wz = (int32_t)(DIM == 3 ? wz_ : 1);
        const double a = par::LAT_A, b = par::LAT_B, c = par::LAT_C;
        const double hf = 0.5 * par::FORCE * a;                    // F.e/2 along +a
        const double Z  = 2.0 * std::cosh(hf) + 2.0 * (DIM - 1);
        const double pp = std::exp(hf) / Z, pm = std::exp(-hf) / Z, pt = 1.0 / Z;

        // bin order: [+x][-x][+y][-y]([+z][-z])
        for (int i = 0; i < 6; ++i) cum[i] = 1.0;
        cum[0] = pp; cum[1] = pp + pm;
        for (int i = 2; i < NB; ++i) cum[i] = cum[i - 1] + pt;
        cum[NB - 1] = 1.0;                                          // exact upper bound

        v_par = a * (pp - pm);                                      // exact drift per jump
        D_par = a * a * std::cosh(hf) / Z;                          // exact
        Dy    = b * b / Z;
        Dz    = (DIM == 3 ? c * c / Z : 0.0);

        Omega = (double)wy * (double)wz;                            // cross-section SITES
        area  = (DIM == 3) ? (wy * b) * (wz * c) : (wy * b);        // physical cross-section
        eps   = Omega * v_par / a;                                  // 1 - Q_0
    }
    // Eq.(34)/(35) asymptote with exact v_par:
    //   <x> = v_par^alpha (a/Omega)^(1-alpha) t^alpha / A_alpha
    double theory()  const {
        return std::pow(v_par, par::ALPHA)
             * std::pow(par::LAT_A / Omega, 1.0 - par::ALPHA)
             * TPOWA / A_ALPHA;
    }
    double n_jumps() const { return TPOWA * std::pow(eps, par::ALPHA - 1.0) / A_ALPHA; }
    double n_star()  const { return 4.0 * D_par / (v_par * v_par); }   // drift-dominance scale
    double ratio()   const { return n_jumps() / n_star(); }

    // (V3) Omega <= EPS_MAX*a/v_par
    // (V6) Omega <= [ t^a v^(1+alpha) a^(1-alpha) / (4 A_alpha D_par R) ]^(1/(1-alpha))
    double omega_cap() const {
        const double c1 = par::EPS_MAX * par::LAT_A / v_par;
        const double num = TPOWA * std::pow(v_par, 1.0 + par::ALPHA)
                                 * std::pow(par::LAT_A, 1.0 - par::ALPHA);
        const double den = 4.0 * A_ALPHA * D_par * par::RATIO_MIN;
        const double c2 = std::pow(num / den, 1.0 / (1.0 - par::ALPHA));
        return std::fmin(c1, c2);
    }
};

struct Result { double mean = 0, sem = 0, njumps = 0; };

// ========================== KERNEL ===================================
template<int DIM>
HOT Result run_point(const Channel<DIM>& ch, uint64_t pseed) {
    // hoist every loop-invariant into a register-resident local
    const double  c0 = ch.cum[0], c1 = ch.cum[1], c2 = ch.cum[2];
    const double  c3 = ch.cum[3], c4 = ch.cum[4];
    const int32_t WY = ch.wy, WZ = ch.wz;

    double sx = 0.0, sx2 = 0.0;
    unsigned long long sn = 0ULL;

    #pragma omp parallel
    {
        Memo M;                       // THREAD-LOCAL OBJECT: no shared array, no false sharing
        M.init();                     // buffer first-touched by this thread (NUMA-local)

        #pragma omp for schedule(dynamic, 32) reduction(+:sx,sx2,sn)
        for (long long iw = 0; iw < par::N_WALKERS; ++iw) {
            // independent quenched realization + independent thermal history per walker
            const uint64_t dseed = splitmix64(pseed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(iw + 1)));
            Xoshiro rng(splitmix64(dseed ^ 0xA5A5A5A5DEADBEEFULL));
            M.new_walker();
            const uint64_t genbits = M.genbits;

            int32_t x = 0, y = 0, z = 0;
            uint64_t stamp = make_stamp(genbits, 0, 0, 0);
            size_t   g     = grp_of(stamp);
            double   elapsed = 0.0, comp = 0.0;      // Kahan-compensated clock
            unsigned long long nj = 0ULL;

            for (;;) {
                // QUENCHED: identical tau on every revisit. One cache line touched.
                const double tau = M.fetch(g, stamp, dseed);

                const double yk = tau - comp;        // compensated summation
                const double tk = elapsed + yk;
                comp = (tk - elapsed) - yk;
                elapsed = tk;
                if (elapsed >= par::T_MAX) break;    // only loop branch; perfectly predicted

                // ---- branchless direction draw (no table, no mispredict) ----
                const double  u  = rng.uni();
                const int32_t b0 = (int32_t)(u >= c0), b1 = (int32_t)(u >= c1);
                const int32_t b2 = (int32_t)(u >= c2), b3 = (int32_t)(u >= c3);
                const int32_t dx = 1 - 2 * b0 + b1;
                const int32_t dy = b1 - 2 * b2 + b3;
                x += dx;

                // ---- branchless PBC wrap (dy,dz in {-1,0,1}) ----
                int32_t ny = y + dy;
                ny += (ny >> 31) & WY;
                ny -= WY & -(int32_t)(ny >= WY);
                y = ny;
                if (DIM == 3) {
                    const int32_t b4 = (int32_t)(u >= c4);
                    const int32_t dz = b3 - 2 * b4;
                    int32_t nz = z + dz;
                    nz += (nz >> 31) & WZ;
                    nz -= WZ & -(int32_t)(nz >= WZ);
                    z = nz;
                }

                // ---- next key computed early + prefetched: hides L2/L3 latency ----
                stamp = make_stamp(genbits, x, y, z);
                g     = grp_of(stamp);
                __builtin_prefetch(&M.tab[g], 1, 3);
                ++nj;
            }
            const double xd = (double)x * par::LAT_A;
            sx += xd; sx2 += xd * xd; sn += nj;
        }
    }
    const double n = (double)par::N_WALKERS;
    Result r;
    r.mean   = sx / n;
    const double var = std::fmax(0.0, sx2 / n - r.mean * r.mean);
    r.sem    = std::sqrt(var / (n - 1.0));
    r.njumps = (double)sn / n;
    return r;
}

// =============== CROSS-SECTION GRID CONSTRUCTION =====================
// 3D: split Omega into (wy,wz) keeping the PHYSICAL section near-square: wy*b ~ wz*c
static void choose_wy_wz(double target, const std::vector<int>& used, int cap3d,
                         int& wy, int& wz) {
    double best = 1e300; wy = 1; wz = 1;
    const int zmax = (int)std::ceil(std::sqrt(target * par::LAT_B / par::LAT_C)) + 6;
    for (int zc = 1; zc <= zmax; ++zc) {
        const int y0 = std::max(1, (int)std::llround(target / (double)zc));
        for (int dy = -2; dy <= 2; ++dy) {
            const int yc = y0 + dy; if (yc < 1) continue;
            const int om = yc * zc; if (om > cap3d) continue;      // stay inside 3D window
            double cost = std::fabs(std::log((double)om / target))
                        + 0.25 * std::fabs(std::log((yc * par::LAT_B) / (zc * par::LAT_C)));
            if (std::find(used.begin(), used.end(), om) != used.end()) cost += 1.0;
            if (cost < best) { best = cost; wy = yc; wz = zc; }
        }
    }
}

int main() {
    const bool do2 = (par::RUN_MODE == par::MODE_2D || par::RUN_MODE == par::MODE_BOTH);
    const bool do3 = (par::RUN_MODE == par::MODE_3D || par::RUN_MODE == par::MODE_BOTH);

    Channel<2> ref2; ref2.build(1, 1);
    Channel<3> ref3; ref3.build(1, 1);
    double cap = 1e300;
    if (do2) cap = std::fmin(cap, ref2.omega_cap());
    if (do3) cap = std::fmin(cap, ref3.omega_cap());
    const int OM_MAX = std::max(par::OMEGA_MIN + 1, (int)std::floor(cap));

    std::fprintf(stderr,
        "SIM3  alpha=%.3g  F=%.3g  t=%.3g  N=%lld  a=%.3g b=%.3g c=%.3g\n"
        "tau0=%.6g  A_alpha=%.6g\n"
        "2D(q=4): v_par=%.6g D_par=%.6g N*=%.4g  Omega_cap=%.3g\n"
        "3D(q=6): v_par=%.6g D_par=%.6g N*=%.4g  Omega_cap=%.3g\n"
        "window: eps<=%.3g AND <N>/N*>=%.3g   ->   Omega in [%d, %d]\n"
        "threads=%d   memo=%zu MB/thread\n\n",
        par::ALPHA, par::FORCE, par::T_MAX, par::N_WALKERS,
        par::LAT_A, par::LAT_B, par::LAT_C, TAU0, A_ALPHA,
        ref2.v_par, ref2.D_par, ref2.n_star(), ref2.omega_cap(),
        ref3.v_par, ref3.D_par, ref3.n_star(), ref3.omega_cap(),
        par::EPS_MAX, par::RATIO_MIN, par::OMEGA_MIN, OM_MAX,
        omp_get_max_threads(), (((size_t)1 << par::GROUP_BITS) * sizeof(SlotGroup)) >> 20);

    // distinct integer Omega grid inside the window, clamped to POINTS
    std::vector<double> tgt;
    const int span = OM_MAX - par::OMEGA_MIN + 1;
    if (span <= par::POINTS) {
        for (int o = par::OMEGA_MIN; o <= OM_MAX; ++o) tgt.push_back((double)o);
        if (span < par::POINTS)
            std::fprintf(stderr, "NOTE: POINTS=%d requested, only %d distinct Omega fit the "
                                 "validity window -> using %d points.\n\n",
                         par::POINTS, span, span);
    } else {
        const double lo = std::log((double)par::OMEGA_MIN), hi = std::log((double)OM_MAX);
        int last = 0;
        for (int i = 0; i < par::POINTS; ++i) {
            int o = (int)std::llround(std::exp(lo + (hi - lo) * i / (par::POINTS - 1.0)));
            o = std::max(o, last + 1); if (o > OM_MAX) break;
            tgt.push_back((double)o); last = o;
        }
    }
    const int NP = (int)tgt.size();

    std::vector<int> w2(NP, 0), y3(NP, 0), z3(NP, 0);
    std::vector<Channel<2>> ch2(NP);
    std::vector<Channel<3>> ch3(NP);
    std::vector<Result> r2(NP), r3(NP);

    const int cap3d = (int)std::floor(ref3.omega_cap());
    std::vector<int> used3;
    for (int i = 0; i < NP; ++i) {
        w2[i] = (int)tgt[i]; ch2[i].build(w2[i], 1);
        int a3, b3; choose_wy_wz(tgt[i], used3, cap3d, a3, b3);
        y3[i] = a3; z3[i] = b3; used3.push_back(a3 * b3); ch3[i].build(a3, b3);
        if (w2[i] >= 4096 || a3 >= 4096 || b3 >= 4096) {
            std::fprintf(stderr, "FATAL: transverse width >= 4096 breaks the memo key packing\n");
            return 1;
        }
    }

    for (int i = 0; i < NP; ++i) {
        if (do2) {
            std::fprintf(stderr, "[2D %2d/%d] wy=%-4d Om=%-5.0f area=%-7.3g eps=%.4g N/N*=%.1f ... ",
                         i + 1, NP, w2[i], ch2[i].Omega, ch2[i].area, ch2[i].eps, ch2[i].ratio());
            const double t0 = omp_get_wtime();
            r2[i] = run_point<2>(ch2[i], splitmix64(par::SEED ^ (0x1000ULL + (uint64_t)i)));
            std::fprintf(stderr, "<x>=%.6g +- %.3g  (%.1fs)\n", r2[i].mean, r2[i].sem, omp_get_wtime() - t0);
        }
        if (do3) {
            std::fprintf(stderr, "[3D %2d/%d] wy=%-3d wz=%-3d Om=%-5.0f area=%-7.3g eps=%.4g N/N*=%.1f ... ",
                         i + 1, NP, y3[i], z3[i], ch3[i].Omega, ch3[i].area, ch3[i].eps, ch3[i].ratio());
            const double t0 = omp_get_wtime();
            r3[i] = run_point<3>(ch3[i], splitmix64(par::SEED ^ (0x2000ULL + (uint64_t)i)));
            std::fprintf(stderr, "<x>=%.6g +- %.3g  (%.1fs)\n", r3[i].mean, r3[i].sem, omp_get_wtime() - t0);
        }
    }

    FILE* f = std::fopen(par::OUT_CSV, "w");
    std::fprintf(f, "# SIM3 first moment vs cross-section area, anisotropic QTM channels (transverse PBC)\n");
    std::fprintf(f, "# alpha=%.10g,A=%.10g,F_par=%.10g,t=%.10g,N_walkers=%lld\n",
                 par::ALPHA, par::A_AMP, par::FORCE, par::T_MAX, par::N_WALKERS);
    std::fprintf(f, "# a=%.10g,b=%.10g,c=%.10g,tau0=%.10g,A_alpha=%.10g\n",
                 par::LAT_A, par::LAT_B, par::LAT_C, TAU0, A_ALPHA);
    std::fprintf(f, "# eps_max=%.10g,ratio_min=%.10g,Omega_min=%d,Omega_max=%d\n",
                 par::EPS_MAX, par::RATIO_MIN, par::OMEGA_MIN, OM_MAX);
    std::fprintf(f, "# geom2D=rectangular(q=4,D0=a^2/4), geom3D=orthorhombic(q=6,D0=a^2/6)\n");
    std::fprintf(f,
        "point,Omega_target,"
        "rect2D_wy,rect2D_Omega,rect2D_area,rect2D_vpar,rect2D_Dpar,rect2D_eps,rect2D_ratio,"
        "rect2D_x_mean,rect2D_x_sem,rect2D_x_theory,rect2D_njumps,"
        "orth3D_wy,orth3D_wz,orth3D_Omega,orth3D_area,orth3D_vpar,orth3D_Dpar,orth3D_eps,orth3D_ratio,"
        "orth3D_x_mean,orth3D_x_sem,orth3D_x_theory,orth3D_njumps\n");

    for (int i = 0; i < NP; ++i) {
        std::fprintf(f, "%d,%.10g,", i, tgt[i]);
        if (do2) std::fprintf(f, "%d,%.0f,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,",
                              w2[i], ch2[i].Omega, ch2[i].area, ch2[i].v_par, ch2[i].D_par,
                              ch2[i].eps, ch2[i].ratio(),
                              r2[i].mean, r2[i].sem, ch2[i].theory(), r2[i].njumps);
        else     std::fprintf(f, "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,");
        if (do3) std::fprintf(f, "%d,%d,%.0f,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                              y3[i], z3[i], ch3[i].Omega, ch3[i].area, ch3[i].v_par, ch3[i].D_par,
                              ch3[i].eps, ch3[i].ratio(),
                              r3[i].mean, r3[i].sem, ch3[i].theory(), r3[i].njumps);
        else     std::fprintf(f, "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan\n");
    }
    std::fclose(f);
    std::fprintf(stderr, "\nwrote %s  (%d points)\n", par::OUT_CSV, NP);
    return 0;
}