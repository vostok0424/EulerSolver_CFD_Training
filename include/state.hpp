#pragma once

// state.hpp
// ---------
// Basic state types and ideal-gas EOS helpers for the Euler equations.
//
// The solver stores the solution in **conservative variables** (U):
//   Dim = 1: U = (rho, rho*u, rho*E)
//   Dim = 2: U = (rho, rho*u, rho*v, rho*E)
//
// For some operations (e.g., limiters, characteristic reconstruction, or output),
// it is more convenient to work with **primitive variables** (W):
//   Dim = 1: W = (rho, u, p)
//   Dim = 2: W = (rho, u, v, p)
//
// EosIdealGas provides:
//   - consToPrim / primToCons conversions
//   - soundSpeed (a)
//   - physFlux: physical Euler flux F(U) in a given direction
//
// Notes:
// - This EOS assumes an ideal gas with constant gamma.
// - No explicit positivity enforcement is performed here; callers should ensure
//   rho>0 and p>0 (reconstruction can optionally apply a positivity fix).

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

// FlowVars1 / FlowVars2
// --------------------
// Cached primitive and thermodynamic quantities derived from conservative states.
// These are intended to reduce repeated conservative -> primitive conversion and
// repeated recomputation of pressure / sound speed / total enthalpy in hot paths.
struct FlowVars1 {
    double rho{};
    double u{};
    double p{};
    double a{};
    double H{};
};

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
struct StateCheckResult {
    bool ok = true;
    StateStatus status = StateStatus::Ok;
    double rho = 0.0;
    double p = 0.0;
    double eInt = 0.0;
};

// StateScanReport
// ---------------
// Summary diagnostics for scanning a collection of states.
struct StateScanReport {
    std::size_t total = 0;
    std::size_t nonFiniteCount = 0;
    std::size_t badDensityCount = 0;
    std::size_t badPressureCount = 0;
    std::size_t badInternalEnergyCount = 0;
    double minRho = 0.0;
    double minP = 0.0;
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
    static FlowVars1 evalFlowVars(const ConsD<1>& U, double gamma);
    static FlowVars2 evalFlowVars(const ConsD<2>& U, double gamma);

    // Physical Euler flux in the requested direction.
    // dir = 0..Dim-1 selects x/y/... direction.
    // Returns F(U) with the same layout as U.
    static Cons physFlux(const Cons& U, int dir, double gamma);
};

using Prim1 = PrimD<1>;
using Prim2 = PrimD<2>;
using Vec3  = ConsD<1>;
using Vec4  = ConsD<2>;

// Cached primitive / thermodynamic evaluation helpers.
FlowVars1 evalFlowVars(const Vec3& U, double gamma);
FlowVars2 evalFlowVars(const Vec4& U, double gamma);

Vec3 physFluxFromFlowVars(const Vec3& U, const FlowVars1& W, int dir);
Vec4 physFluxFromFlowVars(const Vec4& U, const FlowVars2& W, int dir);

// Fast finite checks on conservative states.
bool isFiniteState(const Vec3& U);
bool isFiniteState(const Vec4& U);

// Primitive-state checks.
StateCheckResult checkPrimitive(const Prim1& W, const StateLimits& limits);
StateCheckResult checkPrimitive(const Prim2& W, const StateLimits& limits);

// Conservative-state checks.
StateCheckResult checkConservative(const Vec3& U, double gamma, const StateLimits& limits);
StateCheckResult checkConservative(const Vec4& U, double gamma, const StateLimits& limits);

// Primitive-state positivity clamps.
void clampPrimitive(Prim1& W, const StateLimits& limits);
void clampPrimitive(Prim2& W, const StateLimits& limits);

// Conservative-state repair via primitive conversion, clamping, and back conversion.
bool repairConservative(Vec3& U, double gamma, const StateLimits& limits);
bool repairConservative(Vec4& U, double gamma, const StateLimits& limits);
