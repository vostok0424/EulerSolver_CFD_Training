//
//  cell_repair.cpp
//  EulerSolver_CFD_Training
//
//  Created by SHUANG QIU on 2026/4/18.
//

// Cell-repair implementation for conservative cell-centered states.
//
// This module is responsible for:
// - parsing runtime repair options
// - attempting admissibility-preserving repair of one cell state
// - applying in-place repair to a state vector entry
// - aggregating repair statistics over a collection of cell states
//
// Design boundary:
// - this module repairs states
// - this module does not perform global diagnostics, MPI reduction, or logging
// - this module may use state-related ideas, but keeps its own local helpers so
//   that repair behavior remains explicit and self-contained

#include "cell_repair.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "cfg.hpp"

// Internal helpers used only by this translation unit.
namespace {

// Conservative-state indexing convention used throughout the solver.
constexpr int kRho  = 0;
constexpr int kRhoU = 1;
constexpr int kRhoV = 2;
constexpr int kE    = 3;

// Small local max helper used to avoid bringing algorithmic intent into every
// repair expression.
inline double safeMax(const double a, const double b) {
    return (a > b) ? a : b;
}

// Basic finite-value checks for scalar and conservative-state inputs.
inline bool isFiniteScalar(const double x) {
    return std::isfinite(x);
}

inline bool isFiniteConservativeState(const Vec4& U) {
    return isFiniteScalar(U[kRho]) &&
           isFiniteScalar(U[kRhoU]) &&
           isFiniteScalar(U[kRhoV]) &&
           isFiniteScalar(U[kE]);
}

// Compute |u|^2 from a conservative state.
//
// If density is invalid, return +inf so downstream checks naturally fail.
inline double velocitySquared(const Vec4& U) {
    const double rho = U[kRho];
    if (!std::isfinite(rho) || rho <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double u = U[kRhoU] / rho;
    const double v = U[kRhoV] / rho;
    return u * u + v * v;
}

// Recover pressure from a conservative state using the ideal-gas Euler form.
//
// Invalid density or velocity information maps to -inf pressure so that repair
// logic can treat the state as unacceptable without throwing exceptions.
inline double pressureFromConservative(const Vec4& U, const double gamma) {
    const double rho = U[kRho];
    if (!std::isfinite(rho) || rho <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double vel2 = velocitySquared(U);
    if (!std::isfinite(vel2)) {
        return -std::numeric_limits<double>::infinity();
    }

    const double kinetic = 0.5 * rho * vel2;
    return (gamma - 1.0) * (U[kE] - kinetic);
}

// Recover specific internal energy from a conservative state.
inline double specificInternalEnergyFromConservative(const Vec4& U) {
    const double rho = U[kRho];
    if (!std::isfinite(rho) || rho <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double vel2 = velocitySquared(U);
    if (!std::isfinite(vel2)) {
        return -std::numeric_limits<double>::infinity();
    }

    return U[kE] / rho - 0.5 * vel2;
}

// Check whether a conservative state satisfies the currently requested repair
// floors.
//
// This is the acceptance test used after each repair stage.
inline bool passesRequestedFloors(const Vec4& U,
                                  const double gamma,
                                  const cell_repair::CellRepairOptions& opts) {
    if (!isFiniteConservativeState(U)) {
        return false;
    }

    const double rho = U[kRho];
    const double p = pressureFromConservative(U, gamma);
    const double eint = specificInternalEnergyFromConservative(U);

    if (opts.enforceDensityFloor && (!std::isfinite(rho) || rho < opts.rhoFloor)) {
        return false;
    }
    if (opts.enforcePressureFloor && (!std::isfinite(p) || p < opts.pFloor)) {
        return false;
    }
    if (opts.enforceInternalEnergyFloor && (!std::isfinite(eint) || eint < opts.eintFloor)) {
        return false;
    }

    return true;
}

// Construct a guaranteed finite fallback state that satisfies the requested
// floor constraints as conservatively as possible.
//
// This is used only as a last resort when staged repair cannot recover the
// original state.
inline Vec4 makeFiniteFallbackState(const cell_repair::CellRepairOptions& opts,
                                    const double gamma) {
    const double rho = opts.enforceDensityFloor ? safeMax(opts.rhoFloor, 1.0e-12) : 1.0;
    const double u = 0.0;
    const double v = 0.0;

    double eint = 1.0;
    if (opts.enforcePressureFloor) {
        eint = safeMax(eint, opts.pFloor / ((gamma - 1.0) * rho));
    }
    if (opts.enforceInternalEnergyFloor) {
        eint = safeMax(eint, opts.eintFloor);
    }

    const double E = rho * (eint + 0.5 * (u * u + v * v));
    return Vec4{rho, rho * u, rho * v, E};
}

// First repair stage: enforce density and pressure floors while preserving the
// local velocity implied by the incoming conservative state.
inline bool applyDensityPressureFloor(Vec4& U,
                                      const double gamma,
                                      const cell_repair::CellRepairOptions& opts) {
    if (!isFiniteConservativeState(U)) {
        return false;
    }

    double rho = U[kRho];
    if (opts.enforceDensityFloor) {
        rho = safeMax(rho, opts.rhoFloor);
    }
    if (!std::isfinite(rho) || rho <= 0.0) {
        return false;
    }

    const double ux = U[kRhoU] / U[kRho];
    const double uy = U[kRhoV] / U[kRho];
    const double pOld = pressureFromConservative(U, gamma);
    const double pNew = opts.enforcePressureFloor ? safeMax(pOld, opts.pFloor) : pOld;

    if (!std::isfinite(ux) || !std::isfinite(uy) || !std::isfinite(pNew)) {
        return false;
    }

    U[kRho] = rho;
    U[kRhoU] = rho * ux;
    U[kRhoV] = rho * uy;
    U[kE] = pNew / (gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy);
    return isFiniteConservativeState(U);
}

// Second repair stage: raise specific internal energy when that option is
// enabled.
inline bool applyInternalEnergyFloor(Vec4& U,
                                     const double gamma,
                                     const cell_repair::CellRepairOptions& opts) {
    if (!isFiniteConservativeState(U)) {
        return false;
    }

    const double rho = U[kRho];
    if (!std::isfinite(rho) || rho <= 0.0) {
        return false;
    }

    const double ux = U[kRhoU] / rho;
    const double uy = U[kRhoV] / rho;
    const double eintOld = specificInternalEnergyFromConservative(U);
    const double eintNew = safeMax(eintOld, opts.eintFloor);
    const double pFloorAsEint = opts.enforcePressureFloor ? opts.pFloor / ((gamma - 1.0) * rho) : 0.0;
    const double eintTarget = safeMax(eintNew, pFloorAsEint);

    if (!std::isfinite(ux) || !std::isfinite(uy) || !std::isfinite(eintTarget)) {
        return false;
    }

    U[kE] = rho * (eintTarget + 0.5 * (ux * ux + uy * uy));
    return isFiniteConservativeState(U);
}

// Final staged repair before fallback construction.
//
// This path rebuilds a conservative state from sanitized density, momentum, and
// internal-energy information. If the incoming state is completely non-finite,
// it immediately falls back to a known-safe state.
inline bool applyConservativeRescale(Vec4& U,
                                     const double gamma,
                                     const cell_repair::CellRepairOptions& opts) {
    if (!isFiniteConservativeState(U)) {
        U = makeFiniteFallbackState(opts, gamma);
        return true;
    }

    double rho = U[kRho];
    if (!std::isfinite(rho) || rho <= 0.0) {
        rho = opts.enforceDensityFloor ? safeMax(opts.rhoFloor, 1.0e-12) : 1.0;
    }

    const double rhou = isFiniteScalar(U[kRhoU]) ? U[kRhoU] : 0.0;
    const double rhov = isFiniteScalar(U[kRhoV]) ? U[kRhoV] : 0.0;

    const double ux = rhou / rho;
    const double uy = rhov / rho;

    double eint = specificInternalEnergyFromConservative(U);
    if (!std::isfinite(eint)) {
        eint = 0.0;
    }
    if (opts.enforceInternalEnergyFloor) {
        eint = safeMax(eint, opts.eintFloor);
    }
    if (opts.enforcePressureFloor) {
        eint = safeMax(eint, opts.pFloor / ((gamma - 1.0) * rho));
    }

    U[kRho] = rho;
    U[kRhoU] = rho * ux;
    U[kRhoV] = rho * uy;
    U[kE] = rho * (eint + 0.5 * (ux * ux + uy * uy));
    return isFiniteConservativeState(U);
}

} // namespace

// Public cell-repair interfaces.
namespace cell_repair {

// Parse all cell-repair-related runtime controls from the case configuration.
CellRepairOptions parseCellRepairOptions(const Cfg& cfg) {
    CellRepairOptions opts;

    opts.enable = cfg.getBool("cellRepair.enable", true);
    opts.enforceDensityFloor = cfg.getBool("cellRepair.enforceDensityFloor", true);
    opts.enforcePressureFloor = cfg.getBool("cellRepair.enforcePressureFloor", true);
    opts.enforceInternalEnergyFloor = cfg.getBool("cellRepair.enforceInternalEnergyFloor", false);

    opts.rhoFloor = cfg.getDouble("cellRepair.rhoFloor", 1.0e-12);
    opts.pFloor = cfg.getDouble("cellRepair.pFloor", 1.0e-12);
    opts.eintFloor = cfg.getDouble("cellRepair.eintFloor", 0.0);

    return opts;
}

// Attempt to repair one conservative cell state.
//
// Repair order:
// 1. accept the state immediately if it already satisfies all requested floors
// 2. try density/pressure-floor repair
// 3. try internal-energy-floor repair when enabled
// 4. try conservative rescaling
// 5. fall back to a guaranteed finite constructed state
CellRepairResult repairCellState(const Vec4& U,
                                 const double gamma,
                                 const CellRepairOptions& opts) {
    CellRepairResult result;
    result.U = U;

    if (!opts.enable) {
        result.attempted = false;
        result.success = passesRequestedFloors(result.U, gamma, opts);
        result.changed = false;
        result.method = RepairMethod::None;
        return result;
    }

    if (passesRequestedFloors(result.U, gamma, opts)) {
        result.attempted = false;
        result.success = true;
        result.changed = false;
        result.method = RepairMethod::None;
        return result;
    }

    // From this point onward, at least one repair stage will be attempted.
    result.attempted = true;

    Vec4 candidate = result.U;
    if (applyDensityPressureFloor(candidate, gamma, opts) &&
        passesRequestedFloors(candidate, gamma, opts)) {
        result.U = candidate;
        result.success = true;
        result.changed = (result.U != U);
        result.method = RepairMethod::DensityPressureFloor;
        return result;
    }

    candidate = result.U;
    if (opts.enforceInternalEnergyFloor &&
        applyInternalEnergyFloor(candidate, gamma, opts) &&
        passesRequestedFloors(candidate, gamma, opts)) {
        result.U = candidate;
        result.success = true;
        result.changed = (result.U != U);
        result.method = RepairMethod::InternalEnergyFloor;
        return result;
    }

    candidate = result.U;
    if (applyConservativeRescale(candidate, gamma, opts) &&
        passesRequestedFloors(candidate, gamma, opts)) {
        result.U = candidate;
        result.success = true;
        result.changed = (result.U != U);
        result.method = RepairMethod::ConservativeRescale;
        return result;
    }

    // Last resort: replace the state by a constructed finite fallback state.
    result.U = makeFiniteFallbackState(opts, gamma);
    result.success = passesRequestedFloors(result.U, gamma, opts);
    result.changed = true;
    result.method = result.success ? RepairMethod::ConservativeRescale : RepairMethod::Failed;
    return result;
}

// In-place wrapper around the single-state repair routine.
bool repairCellStateInPlace(Vec4& U,
                            const double gamma,
                            const CellRepairOptions& opts,
                            CellRepairResult* result) {
    const CellRepairResult localResult = repairCellState(U, gamma, opts);
    const bool changed = (localResult.U != U);
    U = localResult.U;

    if (result != nullptr) {
        *result = localResult;
    }

    return changed;
}

// Apply cell repair to every entry in a state array and accumulate batch-level
// statistics.
CellRepairReport repairCellArray(std::vector<Vec4>& U,
                                 const double gamma,
                                 const CellRepairOptions& opts) {
    CellRepairReport report;

    for (Vec4& cell : U) {
        CellRepairResult result;
        repairCellStateInPlace(cell, gamma, opts, &result);

        if (result.attempted) {
            ++report.attemptedCount;
        }
        if (result.success) {
            ++report.successCount;
        } else {
            ++report.failureCount;
        }
        if (result.changed) {
            ++report.changedCount;
        }
    }

    return report;
}

} // namespace cell_repair
