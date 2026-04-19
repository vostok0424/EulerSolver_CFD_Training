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
// - this module reuses shared conservative-state reading helpers from the
//   state layer, while keeping repair-policy and fallback construction local

#include "cell_repair.hpp"
#include "cfg.hpp"

#include <cmath>
#include <limits>


// Internal helpers used only by this translation unit.
namespace {

// Conservative-state indexing convention used by this repair module.
constexpr int kRho  = 0;
constexpr int kRhoU = 1;
constexpr int kRhoV = 2;
constexpr int kE    = 3;

// Small local max helper used to avoid bringing algorithmic intent into every
// repair expression.
inline double safeMax(const double a, const double b) {
    return (a > b) ? a : b;
}

// Basic finite-value check used by local repair-stage reconstruction logic.
inline bool isFiniteScalar(const double x) {
    return std::isfinite(x);
}

// Check whether the current state already satisfies the enabled repair floors.
// State readability itself is delegated to shared state-layer helpers; this
// module remains responsible only for repair policy and stage ordering.
inline bool passesRequestedFloors(const Vec4& U,
                                  const double gamma,
                                  const cell_repair::CellRepairOptions& opts) {
    if (!isFiniteState(U)) {
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

// Final fallback: construct a guaranteed finite state using only configured
// floors and zero velocity. This is the last-resort path when staged repairs
// cannot recover the original state.
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

// Repair stage 1: enforce density/pressure floors while preserving velocity.
// Momentum and total energy are rebuilt from the floored density and pressure.
inline bool applyDensityPressureFloor(Vec4& U,
                                      const double gamma,
                                      const cell_repair::CellRepairOptions& opts) {
    if (isFiniteState(U) && passesRequestedFloors(U, gamma, opts)) {
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
    return isFiniteState(U);
}

// Repair stage 2: raise specific internal energy to the requested floor and
// rebuild total energy consistently with the current density and momentum.
inline bool applyInternalEnergyFloor(Vec4& U,
                                     const double gamma,
                                     const cell_repair::CellRepairOptions& opts) {
    if (isFiniteState(U) && passesRequestedFloors(U, gamma, opts)) {
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
    return isFiniteState(U);
}

// Repair stage 3: sanitize the conservative components directly, then rebuild
// a finite admissible state from floored density, velocity, and internal-
// energy information before the final fallback path is used.
inline bool applyConservativeRescale(Vec4& U,
                                     const double gamma,
                                     const cell_repair::CellRepairOptions& opts) {
    if (isFiniteState(U) && passesRequestedFloors(U, gamma, opts)) {
        return false;
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
    return isFiniteState(U);
}

} // namespace

// Public cell-repair interfaces.
namespace cell_repair {

// Parse all runtime controls and admissibility floors for cell repair.
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

// Apply staged admissibility-preserving repair to one conservative cell state.
//
// Repair order:
// 1. accept the state immediately if it already satisfies all enabled floors
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

    // From this point onward, staged repair has been entered.
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
    // This path is recorded separately from ConservativeRescale in the result.
    result.U = makeFiniteFallbackState(opts, gamma);
    result.success = passesRequestedFloors(result.U, gamma, opts);
    result.changed = true;
    result.method = result.success ? RepairMethod::FallbackState : RepairMethod::Failed;
    return result;
}

// In-place wrapper around the single-state staged repair routine.
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
// repair counters.
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
