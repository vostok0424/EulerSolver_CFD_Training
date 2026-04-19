#pragma once

// Diagnostics module for solution-state inspection and reporting.
//
// This header defines:
// - data structures for state-scan summaries
// - runtime options for diagnostics output
// - interfaces for local scanning, MPI reduction, console printing,
//   and CSV history output
//
// Design intent:
// - diagnostics observes solution quality
// - diagnostics does not repair solution states
// - diagnostics may depend on state utilities, but state must not depend
//   on diagnostics

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

// Summary of a state-quality scan over the local interior cells or over a
// globally reduced domain.
//
// The report records:
// - whether invalid states were detected
// - how many invalid states were detected by category
// - minimum physically important quantities found during the scan
// - optional index locations of those minima
//
// The same structure can be used for:
// - purely local reports
// - accumulated reports
// - MPI-reduced global reports
struct StateScanReport {
    // Flags indicating whether at least one problematic state of a given type
    // was detected during the scan.
    bool hasNonFinite = false;
    bool hasBadDensity = false;
    bool hasBadPressure = false;
    bool hasBadInternalEnergy = false;

    // Counts of problematic cells by category.
    int nonFiniteCount = 0;
    int badDensityCount = 0;
    int badPressureCount = 0;
    int badInternalEnergyCount = 0;
    int repairedCellCount = 0;

    // Minimum physically relevant quantities observed in the scanned region.
    double minRho = 0.0;
    double minPressure = 0.0;
    double minInternalEnergy = 0.0;

    // Optional index locations associated with the minima above.
    // These indices are typically local interior indices unless the report has
    // been post-processed into another coordinate convention.
    int minRhoI = -1;
    int minRhoJ = -1;
    int minPressureI = -1;
    int minPressureJ = -1;
    int minInternalEnergyI = -1;
    int minInternalEnergyJ = -1;

    // Whether the report has been initialized with at least one valid sample.
    bool initialized = false;
};

// Runtime options controlling whether diagnostics are enabled and how they are
// written.
struct StateDiagnosticsOptions {
    // Master switch for state diagnostics.
    bool enable = false;
    // Output CSV file used to append step-by-step diagnostics history.
    std::string csvFile = "state_diagnostics.csv";
    // Whether to print a textual summary to standard output.
    bool printToStdout = true;
    // Whether to include a per-step summary entry when diagnostics are enabled.
    bool includePerStepSummary = true;
};

// Parse diagnostics-related options from the global configuration.
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

// Merge one report into another on the current process.
//
// This is useful for combining multiple local scans before any MPI reduction.
void accumulateStateScanReport(StateScanReport& dst,
                               const StateScanReport& src);

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

// Append one diagnostics record to a CSV history file.
//
// If requested, a header row is written automatically when the file is first
// created.
void appendStateDiagnosticsCsv(const std::string& fileName,
                               const StateScanReport& report,
                               int step,
                               double time,
                               const std::string& tag,
                               bool isRoot);
} // namespace diagnostics
