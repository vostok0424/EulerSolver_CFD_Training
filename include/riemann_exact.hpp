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

namespace exact_riemann {

// Evaluate the one-sided wave function f(p) and its derivative df/dp for a
// candidate star pressure p. This is the core building block used by the
// Newton iteration for the exact Riemann solve.
void prefun(double p, const Prim1& W, double gamma, double& f, double& df);

// Return a PVRS-based initial guess for the star pressure p*. A reasonable
// initial guess improves robustness and convergence speed of the Newton solve.
double guessPressurePVRS(const Prim1& WL, const Prim1& WR, double gamma);

// Solve the exact 1D Riemann problem for the star-region pressure p* and
// velocity u*. The left and right inputs are primitive states.
void starPU(const Prim1& WL, const Prim1& WR, double gamma,
            double& pStar, double& uStar);

// Sample the exact self-similar solution at S = x/t = 0 using the previously
// computed star-region state. The returned value is a primitive sample.
ExactSample1D sampleAtS0(const Prim1& WL, const Prim1& WR, double gamma,
                         double pStar, double uStar);

// Compute the 1D Godunov exact flux. The inputs are conservative states on the
// left and right of an interface; the returned value is the physical flux at
// the sampled exact interface state.
Vec3 godunovExactFlux1D(const Vec3& UL, const Vec3& UR, double gamma);

} // namespace exact_riemann
