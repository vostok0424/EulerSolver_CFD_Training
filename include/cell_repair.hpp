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
// - cell_repair reuses shared conservative-state reading helpers from state
// - state must not depend on cell_repair

#include <string>
#include <vector>

#include "state.hpp"

class Cfg;

namespace cell_repair {

// Repair path used to recover a conservative cell state that violates
// admissibility constraints.
enum class RepairMethod {
    None,
    DensityPressureFloor,
    InternalEnergyFloor,
    ConservativeRescale,
    FallBackState,
    Failed
};

// Runtime options controlling whether repair is enabled and which admissibility
// floors are enforced during staged recovery.
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

// Result of attempting staged repair for one conservative cell state.
struct CellRepairResult {
    // Whether at least one repair stage or fallback path was attempted.
    bool attempted = false;

    // Whether the returned state satisfies the enabled admissibility floors.
    bool success = false;

    // Whether the returned conservative state differs from the input state.
    bool changed = false;

    // Final repair path that produced the returned state.
    RepairMethod method = RepairMethod::None;

    // Conservative state returned after staged repair/fallback processing.
    Vec4 U = Vec4{0.0, 0.0, 0.0, 0.0};
};

// Aggregate counters for repairing a batch of conservative cell states.
struct CellRepairReport {
    int attemptedCount = 0;
    int successCount = 0;
    int failureCount = 0;
    int changedCount = 0;
};

// Parse cell-repair controls and admissibility floors from the case config.
CellRepairOptions parseCellRepairOptions(const Cfg& cfg);

// Apply staged admissibility-preserving repair to one conservative cell state.
// The returned result records whether recovery was attempted, whether it
// succeeded, and which final repair path produced the returned state.
CellRepairResult repairCellState(const Vec4& U,
                                 double gamma,
                                 const CellRepairOptions& opts);

// In-place version of single-cell repair.
// Returns true when the conservative state was changed by staged repair.
bool repairCellStateInPlace(Vec4& U,
                            double gamma,
                            const CellRepairOptions& opts,
                            CellRepairResult* result = nullptr);

// Repair a collection of conservative cell states and return batch counters.
CellRepairReport repairCellArray(std::vector<Vec4>& U,
                                 double gamma,
                                 const CellRepairOptions& opts);

} // namespace cell_repair
