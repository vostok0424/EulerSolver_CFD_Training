#include "flux.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// flux.cpp
// --------
// Implementations of numerical flux functions used by the solver.
//
// Conventions used in this file:
//   - UL / UR : left and right conservative states at a face
//   - dir     : face-normal direction (0 = x, 1 = y)
//   - gamma   : ratio of specific heats
//
// Design notes:
//   - Use evalFlowVars(...) to cache primitive / thermodynamic quantities
//     needed repeatedly in hot paths.
//   - Prefer physFluxFromFlowVars(...) or physFluxFromPrim(...) when the
//     corresponding state form is already available, so we avoid unnecessary
//     conservative <-> primitive reconversion.
//
// Fluxes currently implemented here:
//   1. Rusanov / LLF
//   2. HLLC
//   3. AUSM
//   4. Exact Godunov (directional 1D exact Riemann reduction in 2D)
//
// Extension pattern for a new flux:
//   1. Derive a new class from FluxD<Dim> in flux.hpp.
//   2. Implement numericalFlux(...) in this file.
//   3. Register the scheme in makeFluxD(...) at the bottom of this file.

// ============================================================================
// Rusanov / Local Lax-Friedrichs
// ============================================================================
//
// Template:
//   F = 0.5 * (F_L + F_R) - 0.5 * s_max * (U_R - U_L)
// where s_max is estimated from the largest left/right signal speed
// |u_n| + a in the requested face-normal direction.
template<int Dim>
std::string FluxRusanov<Dim>::name() const { return "rusanov"; }

template<int Dim>
typename FluxRusanov<Dim>::Cons
FluxRusanov<Dim>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    const double unL = (dir == 0) ? WL.u : WL.v;
    const double unR = (dir == 0) ? WR.u : WR.v;
    const double smax = std::max(std::abs(unL) + WL.a, std::abs(unR) + WR.a);

    const Cons FL = physFluxFromFlowVars(UL, WL, dir);
    const Cons FR = physFluxFromFlowVars(UR, WR, dir);

    Cons F{};
    for (int k = 0; k < Dim + 2; ++k) {
        F[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * smax * (UR[k] - UL[k]);
    }
    return F;
}

// Explicit instantiation for Dim = 2.
template class FluxRusanov<2>;

// ============================================================================
// HLLC
// ============================================================================
//
// Three-wave approximate Riemann solver:
//   S_L | S_M | S_R
//
// Programming pattern used here:
//   - estimate outer wave speeds S_L and S_R
//   - compute the contact speed S_M
//   - construct left/right star states
//   - select the correct flux branch by wave-speed sign
//
// In 2D, `dir` selects the normal direction used in the reduction.
// Tangential momentum is carried from the corresponding side state.
ConsD<2> FluxHLLC<2>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    const double aL = WL.a;
    const double aR = WR.a;

    const double unL = (dir == 0) ? WL.u : WL.v;
    const double unR = (dir == 0) ? WR.u : WR.v;

    const double SL = std::min(unL - aL, unR - aR);
    const double SR = std::max(unL + aL, unR + aR);

    const Cons FL = physFluxFromFlowVars(UL, WL, dir);
    const Cons FR = physFluxFromFlowVars(UR, WR, dir);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double rhoL = WL.rho, rhoR = WR.rho;
    const double pL   = WL.p,   pR   = WR.p;

    const double num = pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR);
    const double den = rhoL * (SL - unL) - rhoR * (SR - unR);
    const double SM  = (std::abs(den) < 1e-14) ? 0.0 : num / den;

    auto star = [&](const Cons& U, const FlowVars2& W, double S, double Sstar) -> Cons {
        const double rho = W.rho;
        const double un  = (dir == 0) ? W.u : W.v;
        const double ut  = (dir == 0) ? W.v : W.u;
        const double p   = W.p;
        const double E   = U[3];

        const double rho_star = rho * (S - un) / (S - Sstar);
        const double p_star   = p + rho * (S - un) * (Sstar - un);
        const double E_star   = ((S - un) * E - p * un + p_star * Sstar) / (S - Sstar);

        Cons Us{};
        Us[0] = rho_star;
        if (dir == 0) {
            Us[1] = rho_star * Sstar;
            Us[2] = rho_star * ut;
        } else {
            Us[1] = rho_star * ut;
            Us[2] = rho_star * Sstar;
        }
        Us[3] = E_star;
        return Us;
    };

    const Cons ULs = star(UL, WL, SL, SM);
    const Cons URs = star(UR, WR, SR, SM);

    Cons F{};
    if (SM >= 0.0) {
        for (int k = 0; k < 4; ++k) F[k] = FL[k] + SL * (ULs[k] - UL[k]);
    } else {
        for (int k = 0; k < 4; ++k) F[k] = FR[k] + SR * (URs[k] - UR[k]);
    }
    return F;
}

// ============================================================================
// AUSM
// ============================================================================
//
// AUSM splits the interface flux into:
//   - a convective part based on split Mach-number polynomials
//   - a pressure part based on split pressure polynomials
//
// The helper functions below implement the standard polynomial split pieces
// M^+, M^-, P^+, and P^-.
namespace ausm_detail {

inline double Mplus(double M) {
    const double aM = std::abs(M);
    if (aM >= 1.0) return 0.5 * (M + aM);
    return 0.25 * (M + 1.0) * (M + 1.0);
}

inline double Mminus(double M) {
    const double aM = std::abs(M);
    if (aM >= 1.0) return 0.5 * (M - aM);
    return -0.25 * (M - 1.0) * (M - 1.0);
}

inline double Pplus(double M) {
    const double aM = std::abs(M);
    if (aM >= 1.0) return 0.5 * (1.0 + (M >= 0.0 ? 1.0 : -1.0));
    return 0.25 * (M + 1.0) * (M + 1.0) * (2.0 - M);
}

inline double Pminus(double M) {
    const double aM = std::abs(M);
    if (aM >= 1.0) return 0.5 * (1.0 - (M >= 0.0 ? 1.0 : -1.0));
    return 0.25 * (M - 1.0) * (M - 1.0) * (2.0 + M);
}

} // namespace ausm_detail

ConsD<2> FluxAUSM<2>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    if (dir != 0 && dir != 1) {
        throw std::runtime_error("AUSM flux: dir must be 0 or 1");
    }

    // Use the normal velocity component for Mach splitting.
    // Pressure contributes only to the normal momentum flux component.
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    const double aL = WL.a;
    const double aR = WR.a;
    const double a  = std::max(1e-14, 0.5 * (aL + aR));

    const double unL = (dir == 0) ? WL.u : WL.v;
    const double unR = (dir == 0) ? WR.u : WR.v;
    const double ML  = unL / a;
    const double MR  = unR / a;

    const double mDot = a * (ausm_detail::Mplus(ML) * WL.rho + ausm_detail::Mminus(MR) * WR.rho);
    const double pInt = ausm_detail::Pplus(ML) * WL.p + ausm_detail::Pminus(MR) * WR.p;

    const bool upL = (mDot >= 0.0);
    const double uUp0 = upL ? WL.u : WR.u;
    const double uUp1 = upL ? WL.v : WR.v;
    const double HUp  = upL ? WL.H : WR.H;

    Cons F{};
    F[0] = mDot;

    if (dir == 0) {
        F[1] = mDot * uUp0 + pInt;
        F[2] = mDot * uUp1;
    } else {
        F[1] = mDot * uUp0;
        F[2] = mDot * uUp1 + pInt;
    }

    F[3] = mDot * HUp;
    return F;
}

// ============================================================================
// Exact Godunov
// ============================================================================
//
// This implementation performs a directional exact 1D Riemann solve inside the
// 2D flux function:
//   1. extract (rho, u_n, p) on the chosen face-normal direction
//   2. solve for the star state (p*, u*)
//   3. sample the self-similar solution at S = x / t = 0
//   4. rebuild a 2D primitive state and evaluate the physical flux
//
// The tangential velocity component is upwinded from the side that provides the
// sampled state.
namespace {

// Pressure function used in the Newton iteration for p*.
void godunovPrefunImpl(double p, const Prim1& W, double gamma, double& f, double& df) {
    const double rho = W.rho;
    const double pK  = W.p;
    const double aK  = std::sqrt(std::max(0.0, gamma * pK / rho));

    if (p > pK) {
        const double A = 2.0 / ((gamma + 1.0) * rho);
        const double B = (gamma - 1.0) / (gamma + 1.0) * pK;
        const double sqrtTerm = std::sqrt(A / (p + B));
        f  = (p - pK) * sqrtTerm;
        df = sqrtTerm * (1.0 - 0.5 * (p - pK) / (p + B));
    } else {
        const double pratio = p / pK;
        const double expo = (gamma - 1.0) / (2.0 * gamma);
        f  = (2.0 * aK / (gamma - 1.0)) * (std::pow(pratio, expo) - 1.0);
        df = (1.0 / (rho * aK)) * std::pow(pratio, -(gamma + 1.0) / (2.0 * gamma));
    }
}

// PVRS initial guess for the star-region pressure.
double guessPressurePVRSImpl(const Prim1& WL, const Prim1& WR, double gamma) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));
    const double pPV = 0.5 * (WL.p + WR.p)
                     - 0.125 * (WR.u[0] - WL.u[0]) * (WL.rho + WR.rho) * (aL + aR);
    return std::max(1e-12, pPV);
}

} // namespace

// Thin wrappers kept as FluxGodunov private helpers declared in flux.hpp.
void FluxGodunov<2>::prefun(double p, const Prim1& W, double gamma, double& f, double& df) {
    godunovPrefunImpl(p, W, gamma, f, df);
}

double FluxGodunov<2>::guessPressurePVRS(const Prim1& WL, const Prim1& WR, double gamma) {
    return guessPressurePVRSImpl(WL, WR, gamma);
}

// Solve the exact 1D Riemann problem for the star pressure and star velocity.
void FluxGodunov<2>::starPU(const Prim1& WL, const Prim1& WR, double gamma,
                            double& pStar, double& uStar) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));

    if ((WR.u[0] - WL.u[0]) > (2.0 / (gamma - 1.0)) * (aL + aR)) {
        throw std::runtime_error("Exact Riemann: vacuum not handled.");
    }

    double p = guessPressurePVRSImpl(WL, WR, gamma);

    // Newton iteration for p*.
    for (int it = 0; it < 40; ++it) {
        double fL, dfL, fR, dfR;
        godunovPrefunImpl(p, WL, gamma, fL, dfL);
        godunovPrefunImpl(p, WR, gamma, fR, dfR);

        const double g  = fL + fR + (WR.u[0] - WL.u[0]);
        const double dg = dfL + dfR;

        double pNew = p - g / std::max(1e-14, dg);
        pNew = std::max(1e-12, pNew);

        const double rel = std::abs(pNew - p) / (0.5 * (pNew + p) + 1e-14);
        p = pNew;
        if (rel < 1e-10) {
            break;
        }
    }

    double fL, dfL, fR, dfR;
    godunovPrefunImpl(p, WL, gamma, fL, dfL);
    godunovPrefunImpl(p, WR, gamma, fR, dfR);

    // Recover the star-region pressure and velocity after convergence.
    pStar = p;
    uStar = 0.5 * (WL.u[0] + WR.u[0] + fR - fL);
}

// Sample the exact self-similar solution at S = 0.
ExactSample1D FluxGodunov<2>::sampleAtS0(const Prim1& WL, const Prim1& WR, double gamma,
                                         double pStar, double uStar) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));

    ExactSample1D out{};

    if (0.0 <= uStar) {
        out.sideTag = -1;
        if (pStar > WL.p) {
            const double qL = std::sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pStar / WL.p - 1.0));
            const double SL = WL.u[0] - aL * qL;
            if (0.0 <= SL) {
                out = {WL.rho, WL.u[0], WL.p, -1};
            } else {
                const double pr = pStar / WL.p;
                const double rhoStar = WL.rho * (pr + (gamma - 1.0) / (gamma + 1.0)) /
                                      ((gamma - 1.0) / (gamma + 1.0) * pr + 1.0);
                out = {rhoStar, uStar, pStar, -1};
            }
        } else {
            const double SHL = WL.u[0] - aL;
            const double aStarL = aL * std::pow(pStar / WL.p, (gamma - 1.0) / (2.0 * gamma));
            const double STL = uStar - aStarL;
            if (0.0 <= SHL) {
                out = {WL.rho, WL.u[0], WL.p, -1};
            } else if (0.0 >= STL) {
                const double rhoStar = WL.rho * std::pow(pStar / WL.p, 1.0 / gamma);
                out = {rhoStar, uStar, pStar, -1};
            } else {
                const double S = 0.0;
                const double u = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * WL.u[0] + S);
                const double a = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * (WL.u[0] - S));
                const double rho = WL.rho * std::pow(a / aL, 2.0 / (gamma - 1.0));
                const double p   = WL.p   * std::pow(a / aL, 2.0 * gamma / (gamma - 1.0));
                out = {rho, u, p, -1};
            }
        }
    } else {
        out.sideTag = +1;
        if (pStar > WR.p) {
            const double qR = std::sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pStar / WR.p - 1.0));
            const double SR = WR.u[0] + aR * qR;
            if (0.0 >= SR) {
                out = {WR.rho, WR.u[0], WR.p, +1};
            } else {
                const double pr = pStar / WR.p;
                const double rhoStar = WR.rho * (pr + (gamma - 1.0) / (gamma + 1.0)) /
                                      ((gamma - 1.0) / (gamma + 1.0) * pr + 1.0);
                out = {rhoStar, uStar, pStar, +1};
            }
        } else {
            const double SHR = WR.u[0] + aR;
            const double aStarR = aR * std::pow(pStar / WR.p, (gamma - 1.0) / (2.0 * gamma));
            const double STR = uStar + aStarR;
            if (0.0 >= SHR) {
                out = {WR.rho, WR.u[0], WR.p, +1};
            } else if (0.0 <= STR) {
                const double rhoStar = WR.rho * std::pow(pStar / WR.p, 1.0 / gamma);
                out = {rhoStar, uStar, pStar, +1};
            } else {
                const double S = 0.0;
                const double u = (2.0 / (gamma + 1.0)) * (-aR + 0.5 * (gamma - 1.0) * WR.u[0] + S);
                const double a = (2.0 / (gamma + 1.0)) * ( aR - 0.5 * (gamma - 1.0) * (WR.u[0] - S));
                const double rho = WR.rho * std::pow(a / aR, 2.0 / (gamma - 1.0));
                const double p   = WR.p   * std::pow(a / aR, 2.0 * gamma / (gamma - 1.0));
                out = {rho, u, p, +1};
            }
        }
    }

    return out;
}

// Evaluate the directional exact Godunov flux in 2D.
ConsD<2> FluxGodunov<2>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    Prim1 WL1{}, WR1{};
    WL1.rho = WL.rho;
    WL1.u[0] = (dir == 0) ? WL.u : WL.v;
    WL1.p = WL.p;

    WR1.rho = WR.rho;
    WR1.u[0] = (dir == 0) ? WR.u : WR.v;
    WR1.p = WR.p;

    double pStar = 0.0;
    double uStar = 0.0;
    starPU(WL1, WR1, gamma, pStar, uStar);
    const auto S0 = sampleAtS0(WL1, WR1, gamma, pStar, uStar);

    Prim2 W0{};
    W0.rho = S0.rho;
    W0.p = S0.p;
    W0.u[dir] = S0.u;

    const int tan = 1 - dir;
    W0.u[tan] = (S0.sideTag < 0)
        ? ((tan == 0) ? WL.u : WL.v)
        : ((tan == 0) ? WR.u : WR.v);

    return physFluxFromPrim(W0, dir, gamma);
}

// ============================================================================
// Factory
// ============================================================================
// Map a string name from configuration to the corresponding flux object.
template<int Dim>
std::unique_ptr<FluxD<Dim>> makeFluxD(const std::string& name) {
    if (name == "rusanov") return std::make_unique<FluxRusanov<Dim>>();
    if (name == "hllc")    return std::make_unique<FluxHLLC<Dim>>();
    if (name == "ausm")    return std::make_unique<FluxAUSM<Dim>>();
    if (name == "godunov") return std::make_unique<FluxGodunov<Dim>>();
    throw std::runtime_error("Unknown flux: " + name);
}

template std::unique_ptr<FluxD<2>> makeFluxD<2>(const std::string& name);
