// reconstruction.cpp
// ------------------
// Spatial reconstruction for finite-volume Euler solvers.
//
// Given cell-centered conservative states U (including ghost cells), this file
// reconstructs left/right conservative states at every face:
//   - UL[f] : state seen from the left/bottom side of the face
//   - UR[f] : state seen from the right/top side of the face
// These face states are then passed to the numerical flux / Riemann solver.
//
// Supported cfg options:
//   - reconstruction.scheme         : firstOrder | muscl | weno5
//   - reconstruction.limiter        : none | minmod | vanleer | mc | superbee
//                                     (used by MUSCL only)
//   - reconstruction.positivityFix  : call the centralized state-layer repair
//                                     routine after reconstruction if needed
//   - reconstruction.enableFallback : retry a lower-order reconstruction if a
//                                     reconstructed face state is not admissible
//   - reconstruction.weno.p         : Jiang-Shu WENO exponent p
//
// Characteristic reconstruction is fixed in this implementation:
//   - reconstruction is performed in conservative variables
//   - the face eigensystem is built from a Roe-averaged state
//
// Notes:
//   - WENO5 is the Jiang-Shu formulation on a uniform grid.
//   - The output of every reconstruction path is conservative UL/UR face data.
//   - Optional fallback and state-layer repair are applied face-by-face.


#include "reconstruction.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <array>

// -----------------------------------------------------------------------------
// The same overall pattern is used in both 1D and 2D:
//
//   Inputs:
//     - U : ghosted cell-centered conservative states
//     - mesh extents (nx, ny if applicable) and ng
//     - gamma : ratio of specific heats
//   Outputs:
//     - UL, UR : left/right conservative face states
//
//   1) Check array sizes and the minimum ghost-cell requirement for the chosen
//      stencil width.
//
//   2) For each face, build a Roe-averaged linearization state and the
//      corresponding conservative left/right eigenvector matrices L and R.
//
//   3) Project the local conservative stencil into characteristic space,
//      reconstruct each scalar characteristic component with MUSCL or WENO5,
//      then map the reconstructed characteristic values back with R.
//
//   4) If a reconstructed face state is not admissible, optionally retry with a
//      lower-order method and finally apply positivity repair if enabled.
//
// -----------------------------------------------------------------------------
// Below is a compact mental model for extending this file.
//
//   // 0) Read reconstruction options once in readOptions(cfg)
//   Options opt = readOptions(cfg);
//
//   // 1) Size the face-state output arrays
//   UL.assign(nFaces, Vec{});
//   UR.assign(nFaces, Vec{});
//
//   // 2) For each face, gather the conservative stencil values directly from U
//   //    and project them into characteristic space with the Roe eigensystem.
//
//   // 3) Reconstruct each characteristic component with the chosen scalar
//   //    scheme (first-order / MUSCL / WENO5).
//
//   // 4) Transform the reconstructed characteristic values back to
//   //    conservative variables and store them directly in UL/UR.
//
//   // 5) If needed, apply fallback and positivity repair before accepting the
//   //    face state.

namespace recon {

// -----------------------------------------------------------------------------
// 1) Scalar limiter utilities
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

static inline double mc(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    const double s = sgn(a);
    const double aa = std::abs(a);
    const double bb = std::abs(b);
    return s * std::min({2.0 * aa, 2.0 * bb, 0.5 * (aa + bb)});
}

static inline double superbee(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    const double s = sgn(a);
    const double aa = std::abs(a);
    const double bb = std::abs(b);
    return s * std::max(std::min(2.0 * aa, bb), std::min(aa, 2.0 * bb));
}

static inline double applyLimiter(Limiter lim, double dl, double dr) {
    switch (lim) {
        case Limiter::None:     return 0.5 * (dl + dr);
        case Limiter::Minmod:   return minmod(dl, dr);
        case Limiter::VanLeer:  return vanleer(dl, dr);
        case Limiter::MC:       return mc(dl, dr);
        case Limiter::Superbee: return superbee(dl, dr);
    }
    return minmod(dl, dr);
}



// -----------------------------------------------------------------------------
// 2) Scalar WENO5 kernels on a uniform grid
// -----------------------------------------------------------------------------
// Left state at interface (i+1/2) from cell i using stencil {i-2,i-1,i,i+1,i+2}
static inline double weno5_left(double qim2, double qim1, double qi, double qip1, double qip2,
                                double eps, int p) {
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
    const double a0 = g0 / std::pow(eps + b0, p);
    const double a1 = g1 / std::pow(eps + b1, p);
    const double a2 = g2 / std::pow(eps + b2, p);
    const double asum = a0 + a1 + a2;

    const double w0 = a0 / asum;
    const double w1 = a1 / asum;
    const double w2 = a2 / asum;

    return w0*p0 + w1*p1 + w2*p2;
}

// Right-biased interface state from cell i.
// Implemented by symmetry: reverse the stencil and reuse the left kernel.
static inline double weno5_right(double qim2, double qim1, double qi, double qip1, double qip2,
                                 double eps, int p) {
    return weno5_left(qip2, qip1, qi, qim1, qim2, eps, p);
}

// -----------------------------------------------------------------------------
// 3) Parse reconstruction options from cfg
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
    if (v.empty() || v == "mc") return Limiter::MC;
    if (v == "none") return Limiter::None;
    if (v == "minmod") return Limiter::Minmod;
    if (v == "vanleer") return Limiter::VanLeer;
    if (v == "superbee") return Limiter::Superbee;
    throw std::runtime_error("reconstruction.limiter unsupported: " + v);
}



// readOptions
// -----------
// Parse only the cfg keys that are still active in the fixed
// conservative-characteristic / Roe-linearized implementation.
Options readOptions(const Cfg& cfg) {
    Options opt;
    opt.scheme   = parseScheme(cfg.getString("reconstruction.scheme", "firstOrder"));
    opt.limiter  = parseLimiter(cfg.getString("reconstruction.limiter", "mc"));

    opt.positivityFix = cfg.getBool("reconstruction.positivityFix", true);
    opt.enableFallback = cfg.getBool("reconstruction.enableFallback", true);
    opt.eps            = cfg.getDouble("reconstruction.eps", 1e-12);
    opt.rhoMin         = cfg.getDouble("reconstruction.rhoMin", 1e-12);
    opt.pMin           = cfg.getDouble("reconstruction.pMin", 1e-12);

    opt.wenoP    = cfg.getInt ("reconstruction.weno.p", 2);

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
// 4) Characteristic-projection helpers
// -----------------------------------------------------------------------------

// Eigen3 / Eigen4 store right-eigenvector columns R and left-eigenvector rows L
// for the Roe-linearized Euler system used in characteristic reconstruction.
struct Eigen3 { double R[3][3]; double L[3][3]; };
struct Eigen4 { double R[4][4]; double L[4][4]; };

static inline double enthalpyFromPrim1(const Prim1& W, double gamma) {
    const double u = W.u[0];
    return (gamma/(gamma - 1.0)) * (W.p / W.rho) + 0.5 * u * u;
}

static inline double enthalpyFromPrim2(const Prim2& W, double gamma) {
    const double u = W.u[0];
    const double v = W.u[1];
    return (gamma/(gamma - 1.0)) * (W.p / W.rho) + 0.5 * (u*u + v*v);
}

// Build the Roe-averaged linearization state used at a 1D face.
// The returned (u0, H0, c0) define the conservative eigensystem.
static inline void faceState1D(const Vec3& ULc, const Vec3& URc, double gamma,
                               double& rho0, double& u0, double& p0, double& H0, double& c0) {
    const Prim1 WL = EosIdealGas<1>::consToPrim(ULc, gamma);
    const Prim1 WR = EosIdealGas<1>::consToPrim(URc, gamma);

    // Roe average
    const double sL = std::sqrt(std::max(WL.rho, 0.0));
    const double sR = std::sqrt(std::max(WR.rho, 0.0));
    const double s  = sL + sR;

    const double HL = enthalpyFromPrim1(WL, gamma);
    const double HR = enthalpyFromPrim1(WR, gamma);

    rho0 = sL * sR;
    u0   = (sL * WL.u[0] + sR * WR.u[0]) / s;
    H0   = (sL * HL + sR * HR) / s;

    const double q2 = 0.5 * u0 * u0;
    const double c2 = (gamma - 1.0) * std::max(0.0, H0 - q2);
    c0 = std::sqrt(c2);

    p0 = 0.5 * (WL.p + WR.p);
}

// Build the Roe-averaged linearization state used at a 2D face.
// X- and Y-face reconstructions share this state builder.
static inline void faceState2D(const Vec4& ULc, const Vec4& URc, double gamma,
                               double& rho0, double& u0, double& v0, double& p0, double& H0, double& c0) {
    const Prim2 WL = EosIdealGas<2>::consToPrim(ULc, gamma);
    const Prim2 WR = EosIdealGas<2>::consToPrim(URc, gamma);

    // Roe average
    const double sL = std::sqrt(std::max(WL.rho, 0.0));
    const double sR = std::sqrt(std::max(WR.rho, 0.0));
    const double s  = sL + sR;

    const double HL = enthalpyFromPrim2(WL, gamma);
    const double HR = enthalpyFromPrim2(WR, gamma);

    rho0 = sL * sR;
    u0   = (sL * WL.u[0] + sR * WR.u[0]) / s;
    v0   = (sL * WL.u[1] + sR * WR.u[1]) / s;
    H0   = (sL * HL + sR * HR) / s;

    const double q2 = 0.5 * (u0*u0 + v0*v0);
    const double c2 = (gamma - 1.0) * std::max(0.0, H0 - q2);
    c0 = std::sqrt(c2);

    p0 = 0.5 * (WL.p + WR.p);
}

// Conservative-variable eigensystem for the 1D Euler equations.
static inline void buildEigen1DConservative(double u, double H, double c, double gamma, Eigen3& e) {
    const double beta  = (gamma - 1.0) / (c * c);
    const double alpha = 0.5 * beta * u * u;

    // R columns
    e.R[0][0] = 1.0; e.R[1][0] = u - c; e.R[2][0] = H - u * c;
    e.R[0][1] = 1.0; e.R[1][1] = u;     e.R[2][1] = 0.5 * u * u;
    e.R[0][2] = 1.0; e.R[1][2] = u + c; e.R[2][2] = H + u * c;

    // L rows
    e.L[0][0] = 0.5 * (alpha + u / c);
    e.L[0][1] = -0.5 * (beta * u + 1.0 / c);
    e.L[0][2] = 0.5 * beta;

    e.L[1][0] = 1.0 - alpha;
    e.L[1][1] = beta * u;
    e.L[1][2] = -beta;

    e.L[2][0] = 0.5 * (alpha - u / c);
    e.L[2][1] = -0.5 * (beta * u - 1.0 / c);
    e.L[2][2] = 0.5 * beta;
}

// Conservative-variable eigensystem for x-normal faces in 2D Euler.
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

// Conservative-variable eigensystem for y-normal faces in 2D Euler.
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


static inline Vec3 mulR1D(const Eigen3& e, const double w[3]) {
    Vec3 q{};
    for (int m = 0; m < 3; ++m) {
        q[m] = e.R[m][0] * w[0] + e.R[m][1] * w[1] + e.R[m][2] * w[2];
    }
    return q;
}

static inline Vec4 mulR2D(const Eigen4& e, const double w[4]) {
    Vec4 q{};
    for (int m = 0; m < 4; ++m) {
        q[m] = e.R[m][0] * w[0] + e.R[m][1] * w[1] + e.R[m][2] * w[2] + e.R[m][3] * w[3];
    }
    return q;
}


// The numerical flux module consumes the reconstructed conservative states
// (UL, UR) at each face.
// -----------------------------------------------------------------------------
// -------------------------
// Reconstruction1D
// -------------------------
// Reconstruct all 1D face states. Output arrays are sized to nx+1 faces and
// contain conservative left/right states ready for the flux routine.
Reconstruction1D::Reconstruction1D(const Cfg& cfg) : opt_(readOptions(cfg)) {}

void Reconstruction1D::reconstructFaces(const std::vector<Vec3>& U,
                                        int nx, int ng,
                                        double gamma,
                                        std::vector<Vec3>& UL,
                                        std::vector<Vec3>& UR) const {
    const int nxTot = nx + 2 * ng;
    if ((int)U.size() != nxTot) {
        throw std::runtime_error("Reconstruction1D: U size mismatch");
    }
    if (ng < requiredGhostCells(opt_.scheme)) {
        throw std::runtime_error("Reconstruction1D: ng too small for selected reconstruction.scheme");
    }

    UL.assign(nx + 1, Vec3{});
    UR.assign(nx + 1, Vec3{});

    const StateLimits limits = opt_.stateLimits();

    if (opt_.scheme == Scheme::FirstOrder) {
        for (int i = 0; i < nx + 1; ++i) {
            UL[i] = U[ng + i - 1];
            UR[i] = U[ng + i];
            if (opt_.positivityFix) {
                repairConservative(UL[i], gamma, limits);
                repairConservative(UR[i], gamma, limits);
            }
        }
        return;
    }

    // Reconstruct one face with the requested scheme, always in
    // conservative-characteristic variables with Roe linearization.
    auto reconstructFace = [&](int iFace, Scheme scheme, Vec3& ULf, Vec3& URf) {
        const int kR = ng + iFace;
        const int kL = kR - 1;

        double rho0 = 0.0, u0 = 0.0, p0 = 0.0, H0 = 0.0, c0 = 0.0;
        faceState1D(U[kL], U[kR], gamma, rho0, u0, p0, H0, c0);

        Eigen3 eig{};
        buildEigen1DConservative(u0, H0, std::max(c0, opt_.eps), gamma, eig);

        double wLf[3]{};
        double wRf[3]{};

        for (int s = 0; s < 3; ++s) {
            auto dotL = [&](const Vec3& q) -> double {
                return eig.L[s][0] * q[0] + eig.L[s][1] * q[1] + eig.L[s][2] * q[2];
            };
            auto qAt = [&](int k) -> Vec3 {
                return U[k];
            };

            if (scheme == Scheme::MUSCL) {
                const double wmL = dotL(qAt(kL - 1));
                const double w0L = dotL(qAt(kL));
                const double wpL = dotL(qAt(kL + 1));
                const double sL = applyLimiter(opt_.limiter, w0L - wmL, wpL - w0L);
                wLf[s] = w0L + 0.5 * sL;

                const double wmR = dotL(qAt(kR - 1));
                const double w0R = dotL(qAt(kR));
                const double wpR = dotL(qAt(kR + 1));
                const double sR = applyLimiter(opt_.limiter, w0R - wmR, wpR - w0R);
                wRf[s] = w0R - 0.5 * sR;
            } else if (scheme == Scheme::WENO5) {
                const double a0 = dotL(qAt(kL - 2));
                const double a1 = dotL(qAt(kL - 1));
                const double a2 = dotL(qAt(kL));
                const double a3 = dotL(qAt(kL + 1));
                const double a4 = dotL(qAt(kL + 2));
                wLf[s] = weno5_left(a0, a1, a2, a3, a4, opt_.eps, (int)opt_.wenoP);

                const double b0 = dotL(qAt(kR - 2));
                const double b1 = dotL(qAt(kR - 1));
                const double b2 = dotL(qAt(kR));
                const double b3 = dotL(qAt(kR + 1));
                const double b4 = dotL(qAt(kR + 2));
                wRf[s] = weno5_right(b0, b1, b2, b3, b4, opt_.eps, (int)opt_.wenoP);
            } else {
                throw std::runtime_error("Characteristic reconstruction: unsupported scheme");
            }
        }

        ULf = mulR1D(eig, wLf);
        URf = mulR1D(eig, wRf);
    };

    // Accept the requested reconstruction when admissible; otherwise degrade to
    // a more robust path and finally apply positivity repair if enabled.
    for (int i = 0; i < nx + 1; ++i) {
        reconstructFace(i, opt_.scheme, UL[i], UR[i]);

        bool ok = checkConservative(UL[i], gamma, limits).ok
               && checkConservative(UR[i], gamma, limits).ok;

        if (!ok && opt_.enableFallback && opt_.scheme == Scheme::WENO5) {
            reconstructFace(i, Scheme::MUSCL, UL[i], UR[i]);
            ok = checkConservative(UL[i], gamma, limits).ok
              && checkConservative(UR[i], gamma, limits).ok;
        }
        if (!ok && opt_.enableFallback && opt_.scheme != Scheme::FirstOrder) {
            UL[i] = U[ng + i - 1];
            UR[i] = U[ng + i];
            ok = checkConservative(UL[i], gamma, limits).ok
              && checkConservative(UR[i], gamma, limits).ok;
        }
        if (!ok && opt_.positivityFix) {
            const bool okL = repairConservative(UL[i], gamma, limits);
            const bool okR = repairConservative(UR[i], gamma, limits);
            ok = okL && okR;
        }
    }
}

// -------------------------
// Reconstruction2D
// -------------------------
recon::Reconstruction2D::Reconstruction2D(const Cfg& cfg) : opt_(readOptions(cfg)) {}

// Flatten a ghosted row-major 2D index (I,J) into a single storage index.
static inline int idx2(int i, int j, int nxTot) { return i + nxTot * j; }

// Reconstruct conservative states on all x-normal faces.
// Face storage is packed as f = i + (nx+1)*j over physical face indices.
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

    ULx.assign((nx + 1) * ny, Vec4{});
    URx.assign((nx + 1) * ny, Vec4{});

    const StateLimits limits = opt_.stateLimits();

    if (opt_.scheme == Scheme::FirstOrder) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx + 1; ++i) {
                const int I_R = ng + i;
                const int J   = ng + j;
                const int I_L = I_R - 1;
                const int f   = i + (nx + 1) * j;
                ULx[f] = U[idx2(I_L, J, nxTot)];
                URx[f] = U[idx2(I_R, J, nxTot)];
                if (opt_.positivityFix) {
                    repairConservative(ULx[f], gamma, limits);
                    repairConservative(URx[f], gamma, limits);
                }
            }
        }
        return;
    }

    // Reconstruct one x-face using the requested scalar scheme inside the
    // fixed conservative-characteristic / Roe framework.
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

        double wLf[4]{};
        double wRf[4]{};

        for (int s = 0; s < 4; ++s) {
            auto dotL = [&](const Vec4& q) -> double {
                return eig.L[s][0] * q[0] + eig.L[s][1] * q[1] + eig.L[s][2] * q[2] + eig.L[s][3] * q[3];
            };
            auto qAt = [&](int I, int Jrow) -> Vec4 {
                return U[idx2(I, Jrow, nxTot)];
            };

            if (scheme == Scheme::MUSCL) {
                const double wmL = dotL(qAt(I_L - 1, J));
                const double w0L = dotL(qAt(I_L,     J));
                const double wpL = dotL(qAt(I_L + 1, J));
                const double sL  = applyLimiter(opt_.limiter, w0L - wmL, wpL - w0L);
                wLf[s] = w0L + 0.5 * sL;

                const double wmR = dotL(qAt(I_R - 1, J));
                const double w0R = dotL(qAt(I_R,     J));
                const double wpR = dotL(qAt(I_R + 1, J));
                const double sR  = applyLimiter(opt_.limiter, w0R - wmR, wpR - w0R);
                wRf[s] = w0R - 0.5 * sR;
            } else if (scheme == Scheme::WENO5) {
                const double a0 = dotL(qAt(I_L - 2, J));
                const double a1 = dotL(qAt(I_L - 1, J));
                const double a2 = dotL(qAt(I_L,     J));
                const double a3 = dotL(qAt(I_L + 1, J));
                const double a4 = dotL(qAt(I_L + 2, J));
                wLf[s] = weno5_left(a0, a1, a2, a3, a4, opt_.eps, (int)opt_.wenoP);

                const double b0 = dotL(qAt(I_R - 2, J));
                const double b1 = dotL(qAt(I_R - 1, J));
                const double b2 = dotL(qAt(I_R,     J));
                const double b3 = dotL(qAt(I_R + 1, J));
                const double b4 = dotL(qAt(I_R + 2, J));
                wRf[s] = weno5_right(b0, b1, b2, b3, b4, opt_.eps, (int)opt_.wenoP);
            } else {
                throw std::runtime_error("Characteristic reconstruction: unsupported scheme");
            }
        }

        ULf = mulR2D(eig, wLf);
        URf = mulR2D(eig, wRf);
    };

    // Apply admissibility checks and fallback logic face-by-face.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx + 1; ++i) {
            const int f = i + (nx + 1) * j;
            reconstructFace(i, j, opt_.scheme, ULx[f], URx[f]);

            bool ok = checkConservative(ULx[f], gamma, limits).ok
                   && checkConservative(URx[f], gamma, limits).ok;

            if (!ok && opt_.enableFallback && opt_.scheme == Scheme::WENO5) {
                reconstructFace(i, j, Scheme::MUSCL, ULx[f], URx[f]);
                ok = checkConservative(ULx[f], gamma, limits).ok
                  && checkConservative(URx[f], gamma, limits).ok;
            }
            if (!ok && opt_.enableFallback && opt_.scheme != Scheme::FirstOrder) {
                const int I_R = ng + i;
                const int I_L = I_R - 1;
                const int J   = ng + j;
                ULx[f] = U[idx2(I_L, J, nxTot)];
                URx[f] = U[idx2(I_R, J, nxTot)];
                ok = checkConservative(ULx[f], gamma, limits).ok
                  && checkConservative(URx[f], gamma, limits).ok;
            }
            if (!ok && opt_.positivityFix) {
                const bool okL = repairConservative(ULx[f], gamma, limits);
                const bool okR = repairConservative(URx[f], gamma, limits);
                ok = okL && okR;
            }
        }
    }
}

// Reconstruct conservative states on all y-normal faces.
// Face storage is packed as f = i + nx*j over physical face indices.
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

    ULy.assign(nx * (ny + 1), Vec4{});
    URy.assign(nx * (ny + 1), Vec4{});

    const StateLimits limits = opt_.stateLimits();

    if (opt_.scheme == Scheme::FirstOrder) {
        for (int j = 0; j < ny + 1; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int I = ng + i;
                const int J_T = ng + j;
                const int J_B = J_T - 1;
                const int f = i + nx * j;
                ULy[f] = U[idx2(I, J_B, nxTot)];
                URy[f] = U[idx2(I, J_T, nxTot)];
                if (opt_.positivityFix) {
                    repairConservative(ULy[f], gamma, limits);
                    repairConservative(URy[f], gamma, limits);
                }
            }
        }
        return;
    }

    // Reconstruct one y-face using the requested scalar scheme inside the
    // fixed conservative-characteristic / Roe framework.
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

        double wLf[4]{};
        double wRf[4]{};

        for (int s = 0; s < 4; ++s) {
            auto dotL = [&](const Vec4& q) -> double {
                return eig.L[s][0] * q[0] + eig.L[s][1] * q[1] + eig.L[s][2] * q[2] + eig.L[s][3] * q[3];
            };
            auto qAt = [&](int Icol, int Jrow) -> Vec4 {
                return U[idx2(Icol, Jrow, nxTot)];
            };

            if (scheme == Scheme::MUSCL) {
                const double wmL = dotL(qAt(I, J_B - 1));
                const double w0L = dotL(qAt(I, J_B));
                const double wpL = dotL(qAt(I, J_B + 1));
                const double sL  = applyLimiter(opt_.limiter, w0L - wmL, wpL - w0L);
                wLf[s] = w0L + 0.5 * sL;

                const double wmR = dotL(qAt(I, J_T - 1));
                const double w0R = dotL(qAt(I, J_T));
                const double wpR = dotL(qAt(I, J_T + 1));
                const double sR  = applyLimiter(opt_.limiter, w0R - wmR, wpR - w0R);
                wRf[s] = w0R - 0.5 * sR;
            } else if (scheme == Scheme::WENO5) {
                const double a0 = dotL(qAt(I, J_B - 2));
                const double a1 = dotL(qAt(I, J_B - 1));
                const double a2 = dotL(qAt(I, J_B));
                const double a3 = dotL(qAt(I, J_B + 1));
                const double a4 = dotL(qAt(I, J_B + 2));
                wLf[s] = weno5_left(a0, a1, a2, a3, a4, opt_.eps, (int)opt_.wenoP);

                const double b0 = dotL(qAt(I, J_T - 2));
                const double b1 = dotL(qAt(I, J_T - 1));
                const double b2 = dotL(qAt(I, J_T));
                const double b3 = dotL(qAt(I, J_T + 1));
                const double b4 = dotL(qAt(I, J_T + 2));
                wRf[s] = weno5_right(b0, b1, b2, b3, b4, opt_.eps, (int)opt_.wenoP);
            } else {
                throw std::runtime_error("Characteristic reconstruction: unsupported scheme");
            }
        }

        ULf = mulR2D(eig, wLf);
        URf = mulR2D(eig, wRf);
    };

    // Apply admissibility checks and fallback logic face-by-face.
    for (int j = 0; j < ny + 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int f = i + nx * j;
            reconstructFace(i, j, opt_.scheme, ULy[f], URy[f]);

            bool ok = checkConservative(ULy[f], gamma, limits).ok
                   && checkConservative(URy[f], gamma, limits).ok;

            if (!ok && opt_.enableFallback && opt_.scheme == Scheme::WENO5) {
                reconstructFace(i, j, Scheme::MUSCL, ULy[f], URy[f]);
                ok = checkConservative(ULy[f], gamma, limits).ok
                  && checkConservative(URy[f], gamma, limits).ok;
            }
            if (!ok && opt_.enableFallback && opt_.scheme != Scheme::FirstOrder) {
                const int I   = ng + i;
                const int J_T = ng + j;
                const int J_B = J_T - 1;
                ULy[f] = U[idx2(I, J_B, nxTot)];
                URy[f] = U[idx2(I, J_T, nxTot)];
                ok = checkConservative(ULy[f], gamma, limits).ok
                  && checkConservative(URy[f], gamma, limits).ok;
            }
            if (!ok && opt_.positivityFix) {
                const bool okL = repairConservative(ULy[f], gamma, limits);
                const bool okR = repairConservative(URy[f], gamma, limits);
                ok = okL && okR;
            }
        }
    }
}
} // namespace recon


