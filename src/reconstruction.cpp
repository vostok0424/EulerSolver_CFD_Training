// reconstruction.cpp
// Spatial reconstruction for the 2D finite-volume Euler solver.
// Reconstruct left/right conservative face states UL/UR from ghosted
// cell-centered conservative states U.
//
// Active cfg options:
//   - reconstruction.scheme         : firstOrder | muscl | weno5
//   - reconstruction.limiter        : none | minmod | vanleer
//   - reconstruction.positivityFix  : enable post-reconstruction repair
//   - reconstruction.enableFallback : fall back to first order when needed
//
// This implementation always uses conservative-characteristic reconstruction
// with a Roe face linearization.


#include "reconstruction.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <array>

// -----------------------------------------------------------------------------
// Shared reconstruction pattern:
//   1) Check array sizes and ghost-cell width.
//   2) Build a Roe face eigensystem for the local 2D face orientation.
//   3) Project the local stencil to characteristic space.
//   4) Reconstruct each scalar characteristic component.
//   5) Map the reconstructed face states back to conservative variables.
//   6) Apply fallback and positivity repair if needed.
// -----------------------------------------------------------------------------

namespace recon {

// -----------------------------------------------------------------------------
// 1) Limiter utilities
// -----------------------------------------------------------------------------
static inline std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static inline double sgn(double x) { return (x > 0) - (x < 0); }

static inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return sgn(a) * std::min(std::abs(a), std::abs(b));
}

static inline double vanleer(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (2.0 * a * b) / (a + b);
}


static inline double applyLimiter(Limiter lim, double dl, double dr) {
    switch (lim) {
        case Limiter::None:    return 0.5 * (dl + dr);
        case Limiter::Minmod:  return minmod(dl, dr);
        case Limiter::VanLeer: return vanleer(dl, dr);
    }
    return minmod(dl, dr);
}

// -----------------------------------------------------------------------------
// 2) WENO5 kernels
// -----------------------------------------------------------------------------

static inline double invPowWeno(double x) {
    return 1.0 / (x * x);
}

static inline void weno5_weights(double b0, double b1, double b2,
                                 double eps,
                                 double g0, double g1, double g2,
                                 double& w0, double& w1, double& w2) {
    const double a0 = g0 * invPowWeno(eps + b0);
    const double a1 = g1 * invPowWeno(eps + b1);
    const double a2 = g2 * invPowWeno(eps + b2);
    const double asum = a0 + a1 + a2;

    w0 = a0 / asum;
    w1 = a1 / asum;
    w2 = a2 / asum;
}

// Left state at interface (i+1/2) from cell i using stencil {i-2,i-1,i,i+1,i+2}
static inline double weno5_left(double qim2, double qim1, double qi, double qip1, double qip2,
                                double eps) {
    // Candidate polynomials
    const double p0 = (1.0/3.0)*qim2 - (7.0/6.0)*qim1 + (11.0/6.0)*qi;
    const double p1 = (-1.0/6.0)*qim1 + (5.0/6.0)*qi + (1.0/3.0)*qip1;
    const double p2 = (1.0/3.0)*qi + (5.0/6.0)*qip1 - (1.0/6.0)*qip2;

    // Smoothness indicators (beta)
    const double d0 = qim2 - 2.0*qim1 + qi;
    const double d1 = qim1 - 2.0*qi   + qip1;
    const double d2 = qi   - 2.0*qip1 + qip2;

    const double b0 = (13.0/12.0)*d0*d0 + 0.25*(qim2 - 4.0*qim1 + 3.0*qi)*(qim2 - 4.0*qim1 + 3.0*qi);
    const double b1 = (13.0/12.0)*d1*d1 + 0.25*(qim1 - qip1)*(qim1 - qip1);
    const double b2 = (13.0/12.0)*d2*d2 + 0.25*(3.0*qi - 4.0*qip1 + qip2)*(3.0*qi - 4.0*qip1 + qip2);

    // Linear weights
    const double g0 = 0.1;
    const double g1 = 0.6;
    const double g2 = 0.3;

    // Nonlinear weights
    double w0 = 0.0, w1 = 0.0, w2 = 0.0;
    weno5_weights(b0, b1, b2, eps, g0, g1, g2, w0, w1, w2);

    return w0*p0 + w1*p1 + w2*p2;
}

// Right-biased interface state from cell i.
// Implemented by symmetry: reverse the stencil and reuse the left kernel.
static inline double weno5_right(double qim2, double qim1, double qi, double qip1, double qip2,
                                 double eps) {
    return weno5_left(qip2, qip1, qi, qim1, qim2, eps);
}

// -----------------------------------------------------------------------------
// 3) Option parsing
// -----------------------------------------------------------------------------
static Scheme parseScheme(const std::string& s) {
    const std::string v = trim(s);
    if (v.empty() || v == "firstOrder") return Scheme::FirstOrder;
    if (v == "muscl") return Scheme::MUSCL;
    if (v == "weno5") return Scheme::WENO5;
    throw std::runtime_error("reconstruction.scheme unsupported: " + v);
}

static Limiter parseLimiter(const std::string& s) {
    const std::string v = trim(s);
    if (v.empty() || v == "vanleer") return Limiter::VanLeer;
    if (v == "none") return Limiter::None;
    if (v == "minmod") return Limiter::Minmod;
    throw std::runtime_error("reconstruction.limiter unsupported: " + v + " (supported: none|minmod|vanleer)");
}

// Read the active reconstruction options.
Options readOptions(const Cfg& cfg) {
    Options opt;
    opt.scheme   = parseScheme(cfg.getString("reconstruction.scheme", "firstOrder"));
    opt.limiter  = parseLimiter(cfg.getString("reconstruction.limiter", "vanleer"));

    opt.positivityFix = cfg.getBool("reconstruction.positivityFix", true);
    opt.enableFallback = cfg.getBool("reconstruction.enableFallback", true);
    opt.eps            = cfg.getDouble("reconstruction.eps", 1e-12);
    opt.rhoMin         = cfg.getDouble("reconstruction.rhoMin", 1e-12);
    opt.pMin           = cfg.getDouble("reconstruction.pMin", 1e-12);

    const std::string legacyVars = trim(cfg.getString("reconstruction.variables", ""));
    if (!legacyVars.empty() && legacyVars != "characteristic") {
        throw std::runtime_error("reconstruction.variables is deprecated; only characteristic reconstruction is supported");
    }

    if (opt.eps <= 0.0) {
        throw std::runtime_error("reconstruction.eps must be > 0");
    }
    if (opt.rhoMin <= 0.0) {
        throw std::runtime_error("reconstruction.rhoMin must be > 0");
    }
    if (opt.pMin <= 0.0) {
        throw std::runtime_error("reconstruction.pMin must be > 0");
    }

    return opt;
}


// -----------------------------------------------------------------------------
// 4) Characteristic helpers
// -----------------------------------------------------------------------------

// Roe eigensystem used by characteristic reconstruction.
struct Eigen4 { double R[4][4]; double L[4][4]; };



// Roe face state for 2D reconstruction.
static inline void faceState2D(const Vec4& ULc, const Vec4& URc, double gamma,
                               double& rho0, double& u0, double& v0, double& p0, double& H0, double& c0) {
    const double rhoL = ULc[0];
    const double rhoR = URc[0];

    const double invRhoL = 1.0 / rhoL;
    const double invRhoR = 1.0 / rhoR;

    const double rhouL = ULc[1];
    const double rhouR = URc[1];
    const double rhovL = ULc[2];
    const double rhovR = URc[2];

    const double uL = rhouL * invRhoL;
    const double uR = rhouR * invRhoR;
    const double vL = rhovL * invRhoL;
    const double vR = rhovR * invRhoR;

    const double kineticL = 0.5 * (rhouL * rhouL + rhovL * rhovL) * invRhoL;
    const double kineticR = 0.5 * (rhouR * rhouR + rhovR * rhovR) * invRhoR;

    const double pL = (gamma - 1.0) * (ULc[3] - kineticL);
    const double pR = (gamma - 1.0) * (URc[3] - kineticR);

    const double HL = (ULc[3] + pL) * invRhoL;
    const double HR = (URc[3] + pR) * invRhoR;

    const double sL = std::sqrt(std::max(rhoL, 0.0));
    const double sR = std::sqrt(std::max(rhoR, 0.0));
    const double s  = sL + sR;

    rho0 = sL * sR;
    u0   = (sL * uL + sR * uR) / s;
    v0   = (sL * vL + sR * vR) / s;
    H0   = (sL * HL + sR * HR) / s;

    const double q2 = 0.5 * (u0*u0 + v0*v0);
    const double c2 = (gamma - 1.0) * std::max(0.0, H0 - q2);
    c0 = std::sqrt(c2);

    p0 = 0.5 * (pL + pR);
}


// 2D conservative eigensystem for x-faces.
static inline void buildEigen2DConservativeX(double u, double v, double H, double c, double gamma, Eigen4& e) {
    const double beta  = (gamma - 1.0) / (c * c);
    const double q2    = 0.5 * (u*u + v*v);
    const double alpha = beta * q2;

    // R columns: [wave-, entropy, shear(v), wave+]
    e.R[0][0] = 1.0; e.R[1][0] = u - c; e.R[2][0] = v;     e.R[3][0] = H - u*c;
    e.R[0][1] = 1.0; e.R[1][1] = u;     e.R[2][1] = v;     e.R[3][1] = q2;
    e.R[0][2] = 0.0; e.R[1][2] = 0.0;   e.R[2][2] = 1.0;   e.R[3][2] = v;
    e.R[0][3] = 1.0; e.R[1][3] = u + c; e.R[2][3] = v;     e.R[3][3] = H + u*c;

    // L rows
    e.L[0][0] = 0.5 * (alpha + u / c);
    e.L[0][1] = -0.5 * (beta * u + 1.0 / c);
    e.L[0][2] = -0.5 * beta * v;
    e.L[0][3] = 0.5 * beta;

    e.L[1][0] = 1.0 - alpha;
    e.L[1][1] = beta * u;
    e.L[1][2] = beta * v;
    e.L[1][3] = -beta;

    e.L[2][0] = -v;
    e.L[2][1] = 0.0;
    e.L[2][2] = 1.0;
    e.L[2][3] = 0.0;

    e.L[3][0] = 0.5 * (alpha - u / c);
    e.L[3][1] = -0.5 * (beta * u - 1.0 / c);
    e.L[3][2] = -0.5 * beta * v;
    e.L[3][3] = 0.5 * beta;
}

// 2D conservative eigensystem for y-faces.
static inline void buildEigen2DConservativeY(double u, double v, double H, double c, double gamma, Eigen4& e) {
    const double beta  = (gamma - 1.0) / (c * c);
    const double q2    = 0.5 * (u*u + v*v);
    const double alpha = beta * q2;

    // R columns: [wave-, entropy, shear(u), waves+]
    e.R[0][0] = 1.0; e.R[1][0] = u;     e.R[2][0] = v - c; e.R[3][0] = H - v*c;
    e.R[0][1] = 1.0; e.R[1][1] = u;     e.R[2][1] = v;     e.R[3][1] = q2;
    e.R[0][2] = 0.0; e.R[1][2] = 1.0;   e.R[2][2] = 0.0;   e.R[3][2] = u;
    e.R[0][3] = 1.0; e.R[1][3] = u;     e.R[2][3] = v + c; e.R[3][3] = H + v*c;

    // L rows
    e.L[0][0] = 0.5 * (alpha + v / c);
    e.L[0][1] = -0.5 * beta * u;
    e.L[0][2] = -0.5 * (beta * v + 1.0 / c);
    e.L[0][3] = 0.5 * beta;

    e.L[1][0] = 1.0 - alpha;
    e.L[1][1] = beta * u;
    e.L[1][2] = beta * v;
    e.L[1][3] = -beta;

    e.L[2][0] = -u;
    e.L[2][1] = 1.0;
    e.L[2][2] = 0.0;
    e.L[2][3] = 0.0;

    e.L[3][0] = 0.5 * (alpha - v / c);
    e.L[3][1] = -0.5 * beta * u;
    e.L[3][2] = -0.5 * (beta * v - 1.0 / c);
    e.L[3][3] = 0.5 * beta;
}



static inline void reconstructConservativeFromChar2D(const Eigen4& e, const double* w, Vec4& q) {
    for (int m = 0; m < 4; ++m) {
        q[m] = e.R[m][0] * w[0] + e.R[m][1] * w[1] + e.R[m][2] * w[2] + e.R[m][3] * w[3];
    }
}



static inline void projectChar4(const Eigen4& e, const Vec4& q, double w[4]) {
    for (int s = 0; s < 4; ++s) {
        w[s] = e.L[s][0] * q[0] + e.L[s][1] * q[1] + e.L[s][2] * q[2] + e.L[s][3] * q[3];
    }
}

// File-local admissibility helpers.
template <typename VecT>
struct CachedStateCheck {
    bool quickDone = false;
    bool quickOk   = false;
    bool fullDone  = false;
    bool fullOk    = false;
};

template <typename VecT>
static inline bool quickAdmissibleStateCached(const VecT& Uc,
                                              double gamma,
                                              const StateLimits& limits,
                                              CachedStateCheck<VecT>& cache) {
    if (!cache.quickDone) {
        cache.quickOk = quickCheckConservative(Uc, gamma, limits).ok;
        cache.quickDone = true;
    }
    return cache.quickOk;
}

template <typename VecT>
static inline bool admissibleStateCached(const VecT& Uc,
                                         double gamma,
                                         const StateLimits& limits,
                                         CachedStateCheck<VecT>& cache) {
    if (quickAdmissibleStateCached(Uc, gamma, limits, cache)) {
        cache.fullDone = true;
        cache.fullOk   = true;
        return true;
    }
    if (!cache.fullDone) {
        cache.fullOk = checkConservative(Uc, gamma, limits).ok;
        cache.fullDone = true;
    }
    return cache.fullOk;
}

template <typename VecT>
static inline bool isAdmissibleState(const VecT& Uc, double gamma, const StateLimits& limits) {
    CachedStateCheck<VecT> cache;
    return admissibleStateCached(Uc, gamma, limits, cache);
}


template <typename VecT>
static inline bool isQuickAdmissibleState(const VecT& Uc, double gamma, const StateLimits& limits) {
    CachedStateCheck<VecT> cache;
    return quickAdmissibleStateCached(Uc, gamma, limits, cache);
}

// First-order face loading shared by the 2D reconstruction drivers. The caller
// provides a small loader that fills ULf/URf from neighboring cell-centered states.
// Optional centralized repair is then applied to each face state.
template <typename VecT, typename LoadFaceFn>
static inline void loadFirstOrderFaceStates(LoadFaceFn&& loadFace,
                                            bool positivityFix,
                                            double gamma,
                                            const StateLimits& limits,
                                            VecT& ULf,
                                            VecT& URf) {
    loadFace(ULf, URf);
    if (!positivityFix) {
        return;
    }
    if (!isAdmissibleState(ULf, gamma, limits)) {
        repairConservative(ULf, gamma, limits);
    }
    if (!isAdmissibleState(URf, gamma, limits)) {
        repairConservative(URf, gamma, limits);
    }
}


template <typename VecT, typename EigenT, int NVAR>
static inline void reconstructFaceMUSCLFromCharCache(const EigenT& eig,
                                                     Limiter limiter,
                                                     const double (&wc)[4][NVAR],
                                                     void (*mulR)(const EigenT&, const double*, VecT&),
                                                     VecT& ULf,
                                                     VecT& URf) {
    double wLf[NVAR]{};
    double wRf[NVAR]{};

    for (int s = 0; s < NVAR; ++s) {
        const double wmL = wc[0][s];
        const double w0L = wc[1][s];
        const double wpL = wc[2][s];
        const double sL  = applyLimiter(limiter, w0L - wmL, wpL - w0L);
        wLf[s] = w0L + 0.5 * sL;

        const double wmR = wc[1][s];
        const double w0R = wc[2][s];
        const double wpR = wc[3][s];
        const double sR  = applyLimiter(limiter, w0R - wmR, wpR - w0R);
        wRf[s] = w0R - 0.5 * sR;
    }

    mulR(eig, wLf, ULf);
    mulR(eig, wRf, URf);
}


template <typename VecT, typename EigenT, int NVAR>
static inline void reconstructFaceWENO5FromCharCache(const EigenT& eig,
                                                     double eps,
                                                     const double (&wc)[6][NVAR],
                                                     void (*mulR)(const EigenT&, const double*, VecT&),
                                                     VecT& ULf,
                                                     VecT& URf) {
    double wLf[NVAR]{};
    double wRf[NVAR]{};

    for (int s = 0; s < NVAR; ++s) {
        wLf[s] = weno5_left(wc[0][s], wc[1][s], wc[2][s], wc[3][s], wc[4][s], eps);
        wRf[s] = weno5_right(wc[1][s], wc[2][s], wc[3][s], wc[4][s], wc[5][s], eps);
    }

    mulR(eig, wLf, ULf);
    mulR(eig, wRf, URf);
}

// Shared high-order characteristic reconstruction dispatcher. The caller only
// provides cache-filling lambdas for the MUSCL and WENO5 stencil layouts.
template <typename VecT, typename EigenT, int NVAR,
          typename FillMusclCacheFn, typename FillWenoCacheFn>
static inline void reconstructHighOrderFaceFromChar(Scheme scheme,
                                                    const EigenT& eig,
                                                    Limiter limiter,
                                                    double eps,
                                                    FillMusclCacheFn&& fillMusclCache,
                                                    FillWenoCacheFn&& fillWenoCache,
                                                    void (*mulR)(const EigenT&, const double*, VecT&),
                                                    VecT& ULf,
                                                    VecT& URf) {
    if (scheme == Scheme::MUSCL) {
        double wc[4][NVAR];
        fillMusclCache(wc);
        reconstructFaceMUSCLFromCharCache<VecT, EigenT, NVAR>(eig,
                                                               limiter,
                                                               wc,
                                                               mulR,
                                                               ULf,
                                                               URf);
        return;
    }

    if (scheme == Scheme::WENO5) {
        double wc[6][NVAR];
        fillWenoCache(wc);
        reconstructFaceWENO5FromCharCache<VecT, EigenT, NVAR>(eig,
                                                               eps,
                                                               wc,
                                                               mulR,
                                                               ULf,
                                                               URf);
        return;
    }

    throw std::runtime_error("Characteristic reconstruction: unsupported scheme");
}

// Final face-state selection pipeline shared by the 2D reconstruction paths:
//   1) reconstruct with the requested high-order scheme;
//   2) check admissibility;
//   3) if enabled, fall back to first order when the high-order state fails;
//   4) if enabled, attempt centralized positivity repair as a last step.
template <typename VecT, typename ReconstructFaceFn, typename LoadFirstOrderFn>
static inline void finalizeFaceWithFallback(Scheme requestedScheme,
                                            bool enableFallback,
                                            bool positivityFix,
                                            double gamma,
                                            const StateLimits& limits,
                                            ReconstructFaceFn&& reconstructFaceWithScheme,
                                            LoadFirstOrderFn&& loadFirstOrderFace,
                                            VecT& ULf,
                                            VecT& URf)
{
    CachedStateCheck<VecT> leftCheck;
    CachedStateCheck<VecT> rightCheck;

    auto resetChecks = [&]() {
        leftCheck = CachedStateCheck<VecT>{};
        rightCheck = CachedStateCheck<VecT>{};
    };

    auto reconstructAndReset = [&](Scheme scheme) {
        reconstructFaceWithScheme(scheme, ULf, URf);
        resetChecks();
    };

    auto loadFirstOrderAndReset = [&]() {
        loadFirstOrderFace(ULf, URf);
        resetChecks();
    };

    auto faceAdmissible = [&]() {
        return admissibleStateCached(ULf, gamma, limits, leftCheck)
            && admissibleStateCached(URf, gamma, limits, rightCheck);
    };

    // First try the requested reconstruction scheme.
    reconstructAndReset(requestedScheme);

    bool ok = faceAdmissible();

    // If the requested high-order state is not admissible, retry with a
    // first-order face state when fallback is enabled.
    if (!ok && enableFallback && requestedScheme != Scheme::FirstOrder) {
        loadFirstOrderAndReset();
        ok = quickAdmissibleStateCached(ULf, gamma, limits, leftCheck)
          && quickAdmissibleStateCached(URf, gamma, limits, rightCheck);
        if (!ok) {
            ok = faceAdmissible();
        }
    }

    // As a final safeguard, attempt centralized conservative-state repair on
    // each face state independently.
    if (!ok && positivityFix) {
        bool okL = admissibleStateCached(ULf, gamma, limits, leftCheck);
        bool okR = admissibleStateCached(URf, gamma, limits, rightCheck);
        if (!okL) {
            okL = repairConservative(ULf, gamma, limits);
            leftCheck = CachedStateCheck<VecT>{};
            if (okL) {
                okL = admissibleStateCached(ULf, gamma, limits, leftCheck);
            }
        }
        if (!okR) {
            okR = repairConservative(URf, gamma, limits);
            rightCheck = CachedStateCheck<VecT>{};
            if (okR) {
                okR = admissibleStateCached(URf, gamma, limits, rightCheck);
            }
        }
        ok = okL && okR;
    }
}


// -------------------------
// Reconstruction2D
// -------------------------
recon::Reconstruction2D::Reconstruction2D(const Cfg& cfg) : opt_(readOptions(cfg)) {}

// Flatten a ghosted row-major 2D index.
static inline int idx2(int i, int j, int nxTot) { return i + nxTot * j; }

// Reconstruct conservative states on all x-faces.
void recon::Reconstruction2D::reconstructFacesX(const std::vector<Vec4>& U,
                                         int nx, int ny, int ng,
                                         double gamma,
                                         std::vector<Vec4>& ULx,
                                         std::vector<Vec4>& URx) const {
    const int nxTot = nx + 2 * ng;
    const int nyTot = ny + 2 * ng;
    if ((int)U.size() != nxTot * nyTot) {
        throw std::runtime_error("Reconstruction2D::reconstructFacesX: U size mismatch");
    }
    if (ng < requiredGhostCells(opt_.scheme)) {
        throw std::runtime_error("Reconstruction2D::reconstructFacesX: ng too small for selected reconstruction.scheme");
    }

    const size_t nFaceX = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
    if (ULx.size() != nFaceX) ULx.resize(nFaceX);
    if (URx.size() != nFaceX) URx.resize(nFaceX);

    const StateLimits limits = opt_.stateLimits();

    if (opt_.scheme == Scheme::FirstOrder) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx + 1; ++i) {
                const int I_R = ng + i;
                const int J   = ng + j;
                const int I_L = I_R - 1;
                const int f   = i + (nx + 1) * j;
                auto loadFace = [&](Vec4& ULf, Vec4& URf) {
                    ULf = U[idx2(I_L, J, nxTot)];
                    URf = U[idx2(I_R, J, nxTot)];
                };
                loadFirstOrderFaceStates(loadFace, opt_.positivityFix, gamma, limits, ULx[f], URx[f]);
            }
        }
        return;
    }
    // High-order x-face path: build the face-local Roe eigensystem and perform
    // characteristic reconstruction for the x-normal 2D Euler system.
    // Reconstruct one x-face with the requested scheme.
    auto reconstructFace = [&](int iFace, int jFace, Scheme scheme, Vec4& ULf, Vec4& URf) {
        const int I_R = ng + iFace;
        const int I_L = I_R - 1;
        const int J   = ng + jFace;

        const int kL = idx2(I_L, J, nxTot);
        const int kR = idx2(I_R, J, nxTot);

        double rho0 = 0.0, u0 = 0.0, v0 = 0.0, p0 = 0.0, H0 = 0.0, c0 = 0.0;
        faceState2D(U[kL], U[kR], gamma, rho0, u0, v0, p0, H0, c0);

        Eigen4 eig{};
        buildEigen2DConservativeX(u0, v0, H0, std::max(c0, opt_.eps), gamma, eig);

        auto fillMusclCache = [&](double (&wc)[4][4]) {
            const Vec4* qBase = &U[idx2(I_L - 1, J, nxTot)];
            projectChar4(eig, qBase[0], wc[0]);
            projectChar4(eig, qBase[1], wc[1]);
            projectChar4(eig, qBase[2], wc[2]);
            projectChar4(eig, qBase[3], wc[3]);
        };

        auto fillWenoCache = [&](double (&wc)[6][4]) {
            const Vec4* qBase = &U[idx2(I_L - 2, J, nxTot)];
            projectChar4(eig, qBase[0], wc[0]);
            projectChar4(eig, qBase[1], wc[1]);
            projectChar4(eig, qBase[2], wc[2]);
            projectChar4(eig, qBase[3], wc[3]);
            projectChar4(eig, qBase[4], wc[4]);
            projectChar4(eig, qBase[5], wc[5]);
        };

        reconstructHighOrderFaceFromChar<Vec4, Eigen4, 4>(scheme,
                                                           eig,
                                                           opt_.limiter,
                                                           opt_.eps,
                                                           fillMusclCache,
                                                           fillWenoCache,
                                                           reconstructConservativeFromChar2D,
                                                           ULf,
                                                           URf);
    };

    // Finalize each x-face with admissibility checks, fallback, and repair.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx + 1; ++i) {
            const int f = i + (nx + 1) * j;

            auto reconstructWithScheme = [&](Scheme scheme, Vec4& ULf, Vec4& URf) {
                reconstructFace(i, j, scheme, ULf, URf);
            };

            auto loadFirstOrder = [&](Vec4& ULf, Vec4& URf) {
                const int I_R = ng + i;
                const int I_L = I_R - 1;
                const int J   = ng + j;
                ULf = U[idx2(I_L, J, nxTot)];
                URf = U[idx2(I_R, J, nxTot)];
            };

            finalizeFaceWithFallback(opt_.scheme,
                                       opt_.enableFallback,
                                       opt_.positivityFix,
                                       gamma,
                                       limits,
                                       reconstructWithScheme,
                                       loadFirstOrder,
                                       ULx[f],
                                       URx[f]);
        }
    }
}

// Reconstruct conservative states on all y-faces.
void recon::Reconstruction2D::reconstructFacesY(const std::vector<Vec4>& U,
                                         int nx, int ny, int ng,
                                         double gamma,
                                         std::vector<Vec4>& ULy,
                                         std::vector<Vec4>& URy) const {
    const int nxTot = nx + 2 * ng;
    const int nyTot = ny + 2 * ng;
    if ((int)U.size() != nxTot * nyTot) {
        throw std::runtime_error("Reconstruction2D::reconstructFacesY: U size mismatch");
    }
    if (ng < requiredGhostCells(opt_.scheme)) {
        throw std::runtime_error("Reconstruction2D::reconstructFacesY: ng too small for selected reconstruction.scheme");
    }

    const size_t nFaceY = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);
    if (ULy.size() != nFaceY) ULy.resize(nFaceY);
    if (URy.size() != nFaceY) URy.resize(nFaceY);

    const StateLimits limits = opt_.stateLimits();

    if (opt_.scheme == Scheme::FirstOrder) {
        for (int j = 0; j < ny + 1; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int I = ng + i;
                const int J_T = ng + j;
                const int J_B = J_T - 1;
                const int f = i + nx * j;
                auto loadFace = [&](Vec4& ULf, Vec4& URf) {
                    ULf = U[idx2(I, J_B, nxTot)];
                    URf = U[idx2(I, J_T, nxTot)];
                };
                loadFirstOrderFaceStates(loadFace, opt_.positivityFix, gamma, limits, ULy[f], URy[f]);
            }
        }
        return;
    }
    // High-order y-face path: same reconstruction pattern as the x-face case,
    // but using the y-normal Roe eigensystem and a y-directed stencil.
    // Reconstruct one y-face with the requested scheme.
    auto reconstructFace = [&](int iFace, int jFace, Scheme scheme, Vec4& ULf, Vec4& URf) {
        const int I   = ng + iFace;
        const int J_T = ng + jFace;
        const int J_B = J_T - 1;

        const int kB = idx2(I, J_B, nxTot);
        const int kT = idx2(I, J_T, nxTot);

        double rho0 = 0.0, u0 = 0.0, v0 = 0.0, p0 = 0.0, H0 = 0.0, c0 = 0.0;
        faceState2D(U[kB], U[kT], gamma, rho0, u0, v0, p0, H0, c0);

        Eigen4 eig{};
        buildEigen2DConservativeY(u0, v0, H0, std::max(c0, opt_.eps), gamma, eig);

        auto fillMusclCache = [&](double (&wc)[4][4]) {
            const Vec4* qCache[4] = {
                &U[idx2(I, J_B - 1, nxTot)],
                &U[idx2(I, J_B,     nxTot)],
                &U[idx2(I, J_T,     nxTot)],
                &U[idx2(I, J_T + 1, nxTot)]
            };
            projectChar4(eig, *qCache[0], wc[0]);
            projectChar4(eig, *qCache[1], wc[1]);
            projectChar4(eig, *qCache[2], wc[2]);
            projectChar4(eig, *qCache[3], wc[3]);
        };

        auto fillWenoCache = [&](double (&wc)[6][4]) {
            const Vec4* qCache[6] = {
                &U[idx2(I, J_B - 2, nxTot)],
                &U[idx2(I, J_B - 1, nxTot)],
                &U[idx2(I, J_B,     nxTot)],
                &U[idx2(I, J_T,     nxTot)],
                &U[idx2(I, J_T + 1, nxTot)],
                &U[idx2(I, J_T + 2, nxTot)]
            };
            projectChar4(eig, *qCache[0], wc[0]);
            projectChar4(eig, *qCache[1], wc[1]);
            projectChar4(eig, *qCache[2], wc[2]);
            projectChar4(eig, *qCache[3], wc[3]);
            projectChar4(eig, *qCache[4], wc[4]);
            projectChar4(eig, *qCache[5], wc[5]);
        };

        reconstructHighOrderFaceFromChar<Vec4, Eigen4, 4>(scheme,
                                                           eig,
                                                           opt_.limiter,
                                                           opt_.eps,
                                                           fillMusclCache,
                                                           fillWenoCache,
                                                           reconstructConservativeFromChar2D,
                                                           ULf,
                                                           URf);
    };

    // Finalize each y-face with admissibility checks, fallback, and repair.
    for (int j = 0; j < ny + 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int f = i + nx * j;

            auto reconstructWithScheme = [&](Scheme scheme, Vec4& ULf, Vec4& URf) {
                reconstructFace(i, j, scheme, ULf, URf);
            };

            auto loadFirstOrder = [&](Vec4& ULf, Vec4& URf) {
                const int I   = ng + i;
                const int J_T = ng + j;
                const int J_B = J_T - 1;
                ULf = U[idx2(I, J_B, nxTot)];
                URf = U[idx2(I, J_T, nxTot)];
            };

            finalizeFaceWithFallback(opt_.scheme,
                                       opt_.enableFallback,
                                       opt_.positivityFix,
                                       gamma,
                                       limits,
                                       reconstructWithScheme,
                                       loadFirstOrder,
                                       ULy[f],
                                       URy[f]);
        }
    }
}
} // namespace recon


