#pragma once

// Diagnostics module for solution-state inspection and reporting.
//
// Design intent:
// - diagnostics observes solution quality but never repairs states;
// - diagnostics may depend on state utilities, but state must not depend on diagnostics;
// - runtime controls are collected in StateDiagnosticsOptions so Solver can keep
//   diagnostics configuration in one place.

#include <string>
#include <vector>

#include "state.hpp"

// Forward declarations to avoid unnecessary header coupling.
namespace mpi_parallel {
class MpiParallel;
}
class Cfg;

// All diagnostics-related types and interfaces live in this namespace.
namespace diagnostics {

// Compact report for local or MPI-reduced solution diagnostics.
//
// It combines two groups of information:
// - state-health statistics from scanning interior conservative states;
// - optional positivity-preserving flux-limiter statistics supplied by Solver.
//
// Diagnostics are observational only: this report records problems but does not
// modify the numerical solution.
struct StateScanReport {
    // Failure flags. These are equivalent to the corresponding count > 0,
    // but are kept for quick readability in summaries.
    bool hasNonFinite = false;
    bool hasBadDensity = false;
    bool hasBadPressure = false;
    bool hasBadInternalEnergy = false;

    // Problem-cell counts by category.
    int nonFiniteCount = 0;
    int badDensityCount = 0;
    int badPressureCount = 0;
    int badInternalEnergyCount = 0;

    // Positivity-preserving flux-limiter statistics from the latest RHS build.
    // Counts are summed across MPI ranks; theta values use global minima.
    int positivityLimitedFaceCount = 0;
    int positivityDensityLimitedFaceCount = 0;
    int positivityPressureLimitedFaceCount = 0;
    int positivityFailedFaceCount = 0;

    double positivityMinThetaDensity = 1.0;
    double positivityMinThetaPressure = 1.0;
    double positivityMinThetaFinal = 1.0;

    // Minimum physically relevant quantities observed in the scanned region.
    double minRho = 0.0;
    double minPressure = 0.0;
    double minInternalEnergy = 0.0;

    // Optional locations of the minima above. In MPI-reduced reports these may
    // be reset to (-1, -1) when a reliable global location is unavailable.
    int minRhoI = -1;
    int minRhoJ = -1;
    int minPressureI = -1;
    int minPressureJ = -1;
    int minInternalEnergyI = -1;
    int minInternalEnergyJ = -1;

    // Whether the report has been initialized with at least one valid sample.
    bool initialized = false;
};

// Runtime controls for diagnostics output. Under the preferred Solver-side
// wiring, Solver owns one instance of this structure and uses it for enable,
// CSV path, and optional stdout/step-summary controls.
struct StateDiagnosticsOptions {
    bool enable = true;
    std::string csvFile = "solution/state_diagnostics_2d.csv";
    bool printToStdout = true;
    bool includePerStepSummary = true;
};

// Parse diagnostics-related options from the global configuration.
// Expected keys include:
// - stateDiagnostics.enable
// - stateDiagnostics.csv
// - stateDiagnostics.printToStdout
// - stateDiagnostics.includePerStepSummary
StateDiagnosticsOptions parseStateDiagnosticsOptions(const Cfg& cfg);

// Scan the interior cells of a conservative solution field and build a local
// diagnostics report.
//
// Inputs:
// - U: cell-centered conservative variables including ghost cells
// - nx, ny: local interior cell counts
// - ng: number of ghost layers
// - gamma: ratio of specific heats
// - rhoFloor, pFloor: admissibility thresholds used by the scan
StateScanReport scanInteriorStates(const std::vector<Vec4>& U,
                                   int nx,
                                   int ny,
                                   int ng,
                                   double gamma,
                                   double rhoFloor,
                                   double pFloor);

// Reduce a local state-scan report across all MPI ranks and return a global
// report.
StateScanReport reduceStateScanReportMPI(const StateScanReport& local,
                                         const mpi_parallel::MpiParallel& mpi);

// Return true if the report indicates any invalid or physically unacceptable
// state.
bool hasStateFailure(const StateScanReport& report);

// Print a human-readable diagnostics summary for one time step.
void printStateScanReport(const StateScanReport& report,
                          int step,
                          double time,
                          const std::string& prefix = "[state diagnostics]");

// Append one compact diagnostics record to a CSV history file. A header row is
// written automatically when the file is first created.
void appendStateDiagnosticsCsv(const std::string& fileName,
                               const StateScanReport& report,
                               int step,
                               double time,
                               const std::string& tag,
                               bool isRoot);
} // namespace diagnostics
