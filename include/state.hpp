#pragma once

// state.hpp
// ---------
// Basic state types and ideal-gas EOS helpers for the current 2D Euler solver.
//
// The solver stores the solution in **conservative variables** (U):
//   Dim = 2: U = (rho, rho*u, rho*v, rho*E)
//
// For some operations (e.g., limiters, characteristic reconstruction, or output),
// it is more convenient to work with **primitive variables** (W):
//   Dim = 2: W = (rho, u, v, p)
//
// EosIdealGas provides:
//   - consToPrim / primToCons conversions
//   - soundSpeed (a)
//   - physFlux: physical Euler flux F(U) in a given direction
//
// Notes:
// - This EOS assumes an ideal gas with constant gamma.
// - This header provides state definitions, EOS helpers, and single-state
//   validation utilities.
// - Global scan/summary diagnostics and repair logic now live outside this
//   module.

#include <array>
#include <cstddef>

// PrimD<Dim>
// ---------
// Primitive variables for the Euler equations in Dim dimensions.
//
// Members:
// - rho : density
// - u   : velocity vector (size Dim)
// - p   : pressure
template<int Dim>
struct PrimD {
    double rho{};
    std::array<double,Dim> u{};
    double p{};
};

// ConsD<Dim>
// ---------
// Conservative variables stored in the solver.
// Layout (size = Dim+2):
//   [0]        rho
//   [1..Dim]   rho*u_d
//   [Dim+1]    rho*E   (total energy per volume)
template<int Dim>
using ConsD = std::array<double, Dim+2>;

// Cached primitive and thermodynamic quantities derived from conservative states.
// These are intended to reduce repeated conservative -> primitive conversion and
// repeated recomputation of pressure / sound speed / total enthalpy in hot paths.
struct FlowVars2 {
    double rho{};
    double u{};
    double v{};
    double p{};
    double a{};
    double H{};
};

// StateStatus
// -----------
// Classification of state validity checks.
enum class StateStatus {
    Ok = 0,
    NonFinite,
    NegativeDensity,
    DensityTooSmall,
    NegativePressure,
    PressureTooSmall,
    NegativeInternalEnergy
};

// StateLimits
// -----------
// Global thresholds used when checking or repairing states.
struct StateLimits {
    double eps    = 1e-12;
    double rhoMin = 1e-12;
    double pMin   = 1e-12;
};

// StateCheckResult
// ----------------
// Detailed result returned by primitive/conservative state checks.
// The same result type is reused by both the conservative quick check and the
// fuller conservative admissibility check so callers can escalate from the fast
// path to the detailed path without changing data structures.
struct StateCheckResult {
    bool ok = true;
    StateStatus status = StateStatus::Ok;
    double rho = 0.0;
    double p = 0.0;
    double eInt = 0.0;
};

// EosIdealGas<Dim>
// ---------------
// Ideal-gas equation of state utilities for Euler.
//
// Conversions use:
//   p = (gamma-1) * rho * e_int
// and total energy per volume:
//   rho*E = rho*e_int + 0.5*rho*|u|^2
template<int Dim>
struct EosIdealGas {
    using Prim = PrimD<Dim>;
    using Cons = ConsD<Dim>;

    // Convert conservative U -> primitive W.
    static Prim consToPrim(const Cons& U, double gamma);

    // Convert primitive W -> conservative U.
    static Cons primToCons(const Prim& W, double gamma);

    // Ideal-gas speed of sound: a = sqrt(gamma * p / rho).
    static double soundSpeed(const Prim& W, double gamma);

    // Build cached primitive / thermodynamic quantities from a conservative state.
    // These helpers are intended for hot paths such as numerical fluxes and CFL scans.
    static FlowVars2 evalFlowVars(const ConsD<2>& U, double gamma);

    // Physical Euler flux in the requested direction.
    // dir = 0..Dim-1 selects x/y/... direction.
    // Returns F(U) with the same layout as U.
    static Cons physFlux(const Cons& U, int dir, double gamma);
};

using Prim1 = PrimD<1>;
using Prim2 = PrimD<2>;
using Vec4  = ConsD<2>;

// Cached primitive / thermodynamic evaluation helpers.
FlowVars2 evalFlowVars(const Vec4& U, double gamma);

Vec4 physFluxFromFlowVars(const Vec4& U, const FlowVars2& W, int dir);

Vec4 physFluxFromPrim(const Prim2& W, int dir, double gamma);

// Fast finite checks on conservative states.
bool isFiniteState(const Vec4& U);

// Primitive-state checks.
StateCheckResult checkPrimitive(const Prim2& W, const StateLimits& limits);

// Conservative-state checks.
// quickCheckConservative performs a minimal, hot-path-friendly screen for
// non-finite values, density floors, and pressure/internal-energy positivity.
// checkConservative performs the fuller admissibility check and returns the same
// detailed diagnostic structure when the fast screen fails or deeper reporting
// is needed.
StateCheckResult quickCheckConservative(const Vec4& U, double gamma, const StateLimits& limits);
StateCheckResult checkConservative(const Vec4& U, double gamma, const StateLimits& limits);
