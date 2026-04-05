#pragma once

// riemann_exact.hpp
// Exact 1D Riemann solver utilities (ideal-gas Euler) and a Godunov exact flux.
//
// This implementation follows the standard "exact Riemann solver" algorithm described in
// many CFD texts (e.g., Toro):
//   1) Solve for the star-region pressure p* using a Newton iteration on f(p).
//   2) Compute the star-region velocity u*.
//   3) Sample the exact self-similar solution at S = x/t = 0 to obtain (rho, u, p).
//   4) Convert to conservative variables and evaluate the physical flux.
//
// Intended use in this codebase:
// - As a reference/baseline flux for 1D tests (e.g., Sod shock tube).
// - Not intended as the fastest production flux; it is more expensive than HLLC/Rusanov.
//
// Limitations:
// - Vacuum states are NOT handled (the code throws in that case).
// - A small pressure floor is applied for robustness in iterations.

#include "state.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ExactSample1D
// ------------
// Primitive state sampled from the exact 1D Riemann solution at a given similarity
// coordinate S = x/t (here we only sample at S=0).
//
// sideTag is a small diagnostic indicating which initial side influenced the sample:
//   -1 : left side
//   +1 : right side
struct ExactSample1D {
    double rho{};
    double u{};
    double p{};
    int sideTag{0}; // -1 left, +1 right
};

// The helper functions live in a namespace to avoid polluting the global scope.
// All functions are header-only (inline) for convenience.
namespace exact_riemann {

// prefun(p, W, gamma, f, df)
// -------------------------
// Evaluate the wave function f(p) and its derivative df/dp for one side state W.
//
// - If p > pK: the wave is a shock (uses Rankine–Hugoniot relations).
// - If p <= pK: the wave is a rarefaction (uses isentropic relations).
//
// This is the standard building block in the Newton iteration for p*.
inline void prefun(double p, const Prim1& W, double gamma, double& f, double& df) {
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

// guessPressurePVRS
// -----------------
// Provide an initial guess for the star pressure p*.
// Uses the PVRS (Primitive Variable Riemann Solver) estimate.
//
// A good initial guess improves Newton convergence.
inline double guessPressurePVRS(const Prim1& WL, const Prim1& WR, double gamma) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));
    const double pPV = 0.5 * (WL.p + WR.p)
                     - 0.125 * (WR.u[0] - WL.u[0]) * (WL.rho + WR.rho) * (aL + aR);
    return std::max(1e-12, pPV);
}

// starPU
// ------
// Solve the exact Riemann problem for the star-region pressure p* and velocity u*.
//
// The method:
// - Start from PVRS guess p.
// - Newton-iterate on g(p) = f_L(p) + f_R(p) + (u_R - u_L) = 0.
// - Clamp pressure to a small positive floor for robustness.
//
// Throws if a vacuum state is detected (not implemented).
inline void starPU(const Prim1& WL, const Prim1& WR, double gamma, double& pStar, double& uStar) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));
    if ((WR.u[0] - WL.u[0]) > (2.0/(gamma-1.0))*(aL + aR)) {
        throw std::runtime_error("Exact Riemann: vacuum not handled.");
    }

    double p = guessPressurePVRS(WL, WR, gamma);

    for (int it=0; it<40; ++it) { // Newton iterations (fixed max)
        double fL, dfL, fR, dfR;
        prefun(p, WL, gamma, fL, dfL);
        prefun(p, WR, gamma, fR, dfR);

        const double g  = fL + fR + (WR.u[0] - WL.u[0]);
        const double dg = dfL + dfR;

        double pNew = p - g / std::max(1e-14, dg);
        pNew = std::max(1e-12, pNew);

        const double rel = std::abs(pNew - p) / (0.5*(pNew + p) + 1e-14);
        p = pNew;
        if (rel < 1e-10) break; // relative convergence
    }

    double fL, dfL, fR, dfR;
    prefun(p, WL, gamma, fL, dfL);
    prefun(p, WR, gamma, fR, dfR);

    pStar = p;
    uStar = 0.5 * (WL.u[0] + WR.u[0] + fR - fL);
}

// sampleAtS0
// ----------
// Sample the exact self-similar solution at S = x/t = 0.
//
// Given (p*, u*) we determine whether the sample point lies in:
// - the initial left/right data state,
// - the left/right star state,
// - or inside a rarefaction fan.
//
// The returned (rho, u, p) is primitive.
inline ExactSample1D sampleAtS0(const Prim1& WL, const Prim1& WR, double gamma, double pStar, double uStar) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));

    ExactSample1D out{};

    if (0.0 <= uStar) {
        out.sideTag = -1;
        if (pStar > WL.p) {
            const double qL = std::sqrt(1.0 + (gamma+1.0)/(2.0*gamma) * (pStar/WL.p - 1.0));
            const double SL = WL.u[0] - aL * qL;
            if (0.0 <= SL) {
                out = {WL.rho, WL.u[0], WL.p, -1};
            } else {
                const double pr = pStar / WL.p;
                const double rhoStar = WL.rho * (pr + (gamma-1.0)/(gamma+1.0)) /
                                      ((gamma-1.0)/(gamma+1.0)*pr + 1.0);
                out = {rhoStar, uStar, pStar, -1};
            }
        } else {
            const double SHL = WL.u[0] - aL;
            const double aStarL = aL * std::pow(pStar/WL.p, (gamma-1.0)/(2.0*gamma));
            const double STL = uStar - aStarL;
            if (0.0 <= SHL) {
                out = {WL.rho, WL.u[0], WL.p, -1};
            } else if (0.0 >= STL) {
                const double rhoStar = WL.rho * std::pow(pStar/WL.p, 1.0/gamma);
                out = {rhoStar, uStar, pStar, -1};
            } else {
                const double S = 0.0;
                const double u = (2.0/(gamma+1.0)) * (aL + 0.5*(gamma-1.0)*WL.u[0] + S);
                const double a = (2.0/(gamma+1.0)) * (aL + 0.5*(gamma-1.0)*(WL.u[0] - S));
                const double rho = WL.rho * std::pow(a/aL, 2.0/(gamma-1.0));
                const double p   = WL.p   * std::pow(a/aL, 2.0*gamma/(gamma-1.0));
                out = {rho, u, p, -1};
            }
        }
    } else {
        out.sideTag = +1;
        if (pStar > WR.p) {
            const double qR = std::sqrt(1.0 + (gamma+1.0)/(2.0*gamma) * (pStar/WR.p - 1.0));
            const double SR = WR.u[0] + aR * qR;
            if (0.0 >= SR) {
                out = {WR.rho, WR.u[0], WR.p, +1};
            } else {
                const double pr = pStar / WR.p;
                const double rhoStar = WR.rho * (pr + (gamma-1.0)/(gamma+1.0)) /
                                      ((gamma-1.0)/(gamma+1.0)*pr + 1.0);
                out = {rhoStar, uStar, pStar, +1};
            }
        } else {
            const double SHR = WR.u[0] + aR;
            const double aStarR = aR * std::pow(pStar/WR.p, (gamma-1.0)/(2.0*gamma));
            const double STR = uStar + aStarR;
            if (0.0 >= SHR) {
                out = {WR.rho, WR.u[0], WR.p, +1};
            } else if (0.0 <= STR) {
                const double rhoStar = WR.rho * std::pow(pStar/WR.p, 1.0/gamma);
                out = {rhoStar, uStar, pStar, +1};
            } else {
                const double S = 0.0;
                const double u = (2.0/(gamma+1.0)) * (-aR + 0.5*(gamma-1.0)*WR.u[0] + S);
                const double a = (2.0/(gamma+1.0)) * ( aR - 0.5*(gamma-1.0)*(WR.u[0] - S));
                const double rho = WR.rho * std::pow(a/aR, 2.0/(gamma-1.0));
                const double p   = WR.p   * std::pow(a/aR, 2.0*gamma/(gamma-1.0));
                out = {rho, u, p, +1};
            }
        }
    }
    return out;
}

// godunovExactFlux1D
// ------------------
// Compute the Godunov flux using the *exact* Riemann solution.
//
// Steps:
//  1) Convert UL/UR from conservative to primitive.
//  2) Solve for (p*, u*).
//  3) Sample the solution at S=0 to get W(0,t>0).
//  4) Convert back to conservative U0 and evaluate the physical flux.
//
// Returns the x-direction Euler flux (Dim=1).
inline Vec3 godunovExactFlux1D(const Vec3& UL, const Vec3& UR, double gamma) {
    const auto WL = EosIdealGas<1>::consToPrim(UL, gamma);
    const auto WR = EosIdealGas<1>::consToPrim(UR, gamma);

    double pStar, uStar;
    starPU(WL, WR, gamma, pStar, uStar);
    const auto S0 = sampleAtS0(WL, WR, gamma, pStar, uStar);

    Prim1 W0{};
    W0.rho = S0.rho;
    W0.u[0]= S0.u;
    W0.p   = S0.p;

    const Vec3 U0 = EosIdealGas<1>::primToCons(W0, gamma);
    return EosIdealGas<1>::physFlux(U0, 0, gamma);
}

} // namespace exact_riemann
