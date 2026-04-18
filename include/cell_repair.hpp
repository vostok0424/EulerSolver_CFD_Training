#pragma once

// Cell-repair module for conservative cell states.
//
// This header defines:
// - repair status and reporting structures
// - runtime options controlling cell-repair behavior
// - interfaces for repairing a single conservative cell state
// - interfaces for repairing a collection of cell states
//
// Design intent:
// - cell_repair modifies invalid or marginal cell-centered states
// - cell_repair does not perform global diagnostics or logging
// - cell_repair may depend on state utilities, but state must not depend
//   on cell_repair

#include <string>
#include <vector>

#include "state.hpp"

class Cfg;

namespace cell_repair {

// Strategy used when a conservative cell state violates admissibility
// constraints.
enum class RepairMethod {
    None,
    DensityPressureFloor,
    InternalEnergyFloor,
    ConservativeRescale,
    Failed
};

// Runtime options controlling whether cell repair is enabled and which floors
// are enforced.
struct CellRepairOptions {
    // Master switch for cell-state repair.
    bool enable = true;

    // If true, enforce a minimum density floor during repair.
    bool enforceDensityFloor = true;

    // If true, enforce a minimum pressure floor during repair.
    bool enforcePressureFloor = true;

    // If true, enforce a minimum specific internal-energy floor during repair.
    bool enforceInternalEnergyFloor = false;

    // Minimum admissible density used during repair.
    double rhoFloor = 1.0e-12;

    // Minimum admissible pressure used during repair.
    double pFloor = 1.0e-12;

    // Minimum admissible specific internal energy used during repair.
    double eintFloor = 0.0;
};

// Result of attempting to repair a single conservative cell state.
struct CellRepairResult {
    // Whether the input state required any repair action.
    bool attempted = false;

    // Whether the returned state satisfies the requested repair constraints.
    bool success = false;

    // Whether the conservative state was modified.
    bool changed = false;

    // Repair method actually applied.
    RepairMethod method = RepairMethod::None;

    // Repaired conservative state.
    Vec4 U = Vec4{0.0, 0.0, 0.0, 0.0};
};

// Aggregate report for repairing a batch of cell states.
struct CellRepairReport {
    int attemptedCount = 0;
    int successCount = 0;
    int failureCount = 0;
    int changedCount = 0;
};

// Parse cell-repair options from the global configuration.
CellRepairOptions parseCellRepairOptions(const Cfg& cfg);

// Apply admissibility-preserving repair to one conservative cell state.
//
// Inputs:
// - U: input conservative state
// - gamma: ratio of specific heats
// - opts: repair options controlling floors and enabled repair stages
CellRepairResult repairCellState(const Vec4& U,
                                 double gamma,
                                 const CellRepairOptions& opts);

// In-place version of single-cell repair.
//
// Returns true if the state was modified.
bool repairCellStateInPlace(Vec4& U,
                            double gamma,
                            const CellRepairOptions& opts,
                            CellRepairResult* result = nullptr);

// Repair a collection of conservative cell states and return a batch report.
CellRepairReport repairCellArray(std::vector<Vec4>& U,
                                 double gamma,
                                 const CellRepairOptions& opts);

} // namespace cell_repair
