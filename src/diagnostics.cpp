#include "diagnostics.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <mpi.h>
#include "cfg.hpp"
#include "mpi_parallel.hpp"

namespace {

// Conservative-state indexing convention used in this diagnostics module.
// Only density is accessed directly here; pressure and internal energy are
// evaluated through state-layer helper functions to avoid duplicated physics
// logic inside diagnostics.
constexpr int kRho  = 0;


// Convert a 2D structured-grid index pair (i, j) into the corresponding
// flattened storage index for vectors laid out in row-major order.
inline int flatIndex(const int i, const int j, const int nxTot) {
    return j * nxTot + i;
}

// Update one tracked minimum value and its grid location.
// Non-finite candidate values are ignored so that diagnostics minima are
// derived only from physically readable finite states.
inline void updateMinimum(double value,
                          int i,
                          int j,
                          double& currentMin,
                          int& currentI,
                          int& currentJ,
                          bool& initialized) {
    if (!std::isfinite(value)) {
        return;
    }

    if (!initialized || value < currentMin) {
        currentMin = value;
        currentI = i;
        currentJ = j;
        initialized = true;
    }
}

// Return true when the CSV file already exists on disk.
// This is used only by the root rank to decide whether a header row must be
// written before appending a new diagnostics record.
inline bool fileExists(const std::string& fileName) {
    std::ifstream fin(fileName.c_str());
    return fin.good();
}

} // namespace

namespace diagnostics {

// Parse optional state-diagnostics controls from the case configuration.
// Missing entries fall back to conservative defaults so diagnostics can be
// enabled incrementally without requiring all keys to be present.
StateDiagnosticsOptions parseStateDiagnosticsOptions(const Cfg& cfg) {
    StateDiagnosticsOptions opts;

    opts.enable = cfg.getBool("stateDiagnostics.enable", false);
    opts.csvFile = cfg.getString("stateDiagnostics.csv", "state_diagnostics.csv");
    opts.printToStdout = cfg.getBool("stateDiagnostics.printToStdout", true);
    opts.includePerStepSummary = cfg.getBool("stateDiagnostics.includePerStepSummary", true);

    return opts;
}

// Scan the local physical-cell block and summarize state admissibility.
// Ghost cells are excluded deliberately: this routine is intended to assess
// the solver-owned interior region after updates, repairs, and boundary
// application have been completed for the current stage.
StateScanReport scanInteriorStates(const std::vector<Vec4>& U,
                                   const int nx,
                                   const int ny,
                                   const int ng,
                                   const double gamma,
                                   const double rhoFloor,
                                   const double pFloor) {
    StateScanReport rep;

    const int nxTot = nx + 2 * ng;
    const int nyTot = ny + 2 * ng;
    const std::size_t expectedSize = static_cast<std::size_t>(nxTot * nyTot);

    // A storage-size mismatch indicates that the state array cannot be mapped
    // consistently onto the expected structured block. Mark the report as
    // failed and encode the absolute size mismatch in nonFiniteCount so the
    // caller can detect the setup error without continuing the scan.
    if (U.size() != expectedSize) {
        rep.hasNonFinite = true;
        rep.nonFiniteCount = static_cast<int>(expectedSize > U.size() ? expectedSize - U.size() : U.size() - expectedSize);
        return rep;
    }

    bool minRhoInitialized = false;
    bool minPressureInitialized = false;
    bool minEintInitialized = false;

    for (int j = ng; j < ng + ny; ++j) {
        for (int i = ng; i < ng + nx; ++i) {
            const Vec4& cell = U[flatIndex(i, j, nxTot)];

            if (!isFiniteState(cell)) {
                rep.hasNonFinite = true;
                ++rep.nonFiniteCount;
                continue;
            }

            const double rho = cell[kRho];
            const double p = safePressure(cell, gamma);
            const double eint = safeInternalEnergy(cell);

            updateMinimum(rho, i, j, rep.minRho, rep.minRhoI, rep.minRhoJ, minRhoInitialized);
            updateMinimum(p, i, j, rep.minPressure, rep.minPressureI, rep.minPressureJ, minPressureInitialized);
            updateMinimum(eint, i, j, rep.minInternalEnergy, rep.minInternalEnergyI, rep.minInternalEnergyJ, minEintInitialized);

            if (!std::isfinite(rho) || rho <= rhoFloor) {
                rep.hasBadDensity = true;
                ++rep.badDensityCount;
            }
            if (!std::isfinite(p) || p <= pFloor) {
                rep.hasBadPressure = true;
                ++rep.badPressureCount;
            }
            if (!std::isfinite(eint) || eint <= 0.0) {
                rep.hasBadInternalEnergy = true;
                ++rep.badInternalEnergyCount;
            }
        }
    }

    rep.initialized = minRhoInitialized || minPressureInitialized || minEintInitialized;
    return rep;
}


// Reduce per-rank diagnostics reports into one communicator-wide summary.
// Boolean flags are combined with MPI_MAX, counters are summed, and scalar
// minima are reduced exactly with MPI_MIN.
//
// Note: the associated (i, j) locations of minima are not globally reduced as
// authoritative coordinates here. A location is preserved only when the local
// rank owns a minimum value equal to the reduced global minimum; otherwise the
// location is reset to (-1, -1) to avoid reporting misleading coordinates.
StateScanReport reduceStateScanReportMPI(const StateScanReport& local,
                                         const mpi_parallel::MpiParallel& mpi) {
    StateScanReport global = local;
    const MPI_Comm comm = mpi.cartComm();
    if (comm == MPI_COMM_NULL) return global;

    int localFlags[4] = {
        local.hasNonFinite ? 1 : 0,
        local.hasBadDensity ? 1 : 0,
        local.hasBadPressure ? 1 : 0,
        local.hasBadInternalEnergy ? 1 : 0
    };
    int globalFlags[4] = {0, 0, 0, 0};
    MPI_Allreduce(localFlags, globalFlags, 4, MPI_INT, MPI_MAX, comm);

    global.hasNonFinite = (globalFlags[0] != 0);
    global.hasBadDensity = (globalFlags[1] != 0);
    global.hasBadPressure = (globalFlags[2] != 0);
    global.hasBadInternalEnergy = (globalFlags[3] != 0);

    int localCounts[5] = {
        local.nonFiniteCount,
        local.badDensityCount,
        local.badPressureCount,
        local.badInternalEnergyCount,
        local.repairedCellCount
    };
    int globalCounts[5] = {0, 0, 0, 0, 0};
    MPI_Allreduce(localCounts, globalCounts, 5, MPI_INT, MPI_SUM, comm);

    global.nonFiniteCount = globalCounts[0];
    global.badDensityCount = globalCounts[1];
    global.badPressureCount = globalCounts[2];
    global.badInternalEnergyCount = globalCounts[3];
    global.repairedCellCount = globalCounts[4];

    const double huge = std::numeric_limits<double>::max();
    const double localMins[3] = {
        local.initialized ? local.minRho : huge,
        local.initialized ? local.minPressure : huge,
        local.initialized ? local.minInternalEnergy : huge
    };
    double globalMins[3] = {huge, huge, huge};
    MPI_Allreduce(localMins, globalMins, 3, MPI_DOUBLE, MPI_MIN, comm);

    global.minRho = globalMins[0];
    global.minPressure = globalMins[1];
    global.minInternalEnergy = globalMins[2];
    global.initialized = (globalMins[0] < huge) || (globalMins[1] < huge) || (globalMins[2] < huge);

    // The minimum values themselves are globally reduced exactly.
    // Their stored index locations, however, remain conditionally valid: a
    // local (i, j) pair is retained only when this rank owns a minimum equal
    // to the reduced global minimum. Otherwise the location is cleared to
    // (-1, -1) rather than implying a false global coordinate.
    if (!(local.initialized && local.minRho == global.minRho)) {
        global.minRhoI = -1;
        global.minRhoJ = -1;
    }
    if (!(local.initialized && local.minPressure == global.minPressure)) {
        global.minPressureI = -1;
        global.minPressureJ = -1;
    }
    if (!(local.initialized && local.minInternalEnergy == global.minInternalEnergy)) {
        global.minInternalEnergyI = -1;
        global.minInternalEnergyJ = -1;
    }

    return global;
}

// Return true when any hard state-admissibility failure was detected.
// This lightweight predicate is used by callers that only need a yes/no gate
// rather than the full diagnostic breakdown.
bool hasStateFailure(const StateScanReport& report) {
    return report.hasNonFinite ||
           report.hasBadDensity ||
           report.hasBadPressure ||
           report.hasBadInternalEnergy;
}

// Emit one compact human-readable diagnostics summary line.
// This is intended for console monitoring and mirrors the key fields written
// to CSV without introducing additional formatting dependencies.
void printStateScanReport(const StateScanReport& report,
                          const int step,
                          const double time,
                          const std::string& prefix) {
    std::ostringstream oss;
    oss << prefix
        << " step=" << step
        << " time=" << std::setprecision(16) << time
        << " nonFinite=" << report.nonFiniteCount
        << " badDensity=" << report.badDensityCount
        << " badPressure=" << report.badPressureCount
        << " badEint=" << report.badInternalEnergyCount
        << " repaired=" << report.repairedCellCount;

    if (report.initialized) {
        oss << " minRho=" << report.minRho
            << " minP=" << report.minPressure
            << " minEint=" << report.minInternalEnergy;

        if (report.minRhoI >= 0 && report.minRhoJ >= 0) {
            oss << " minRho@(" << report.minRhoI << "," << report.minRhoJ << ")";
        }
        if (report.minPressureI >= 0 && report.minPressureJ >= 0) {
            oss << " minP@(" << report.minPressureI << "," << report.minPressureJ << ")";
        }
        if (report.minInternalEnergyI >= 0 && report.minInternalEnergyJ >= 0) {
            oss << " minEint@(" << report.minInternalEnergyI << "," << report.minInternalEnergyJ << ")";
        }
    }

    std::cout << oss.str() << std::endl;
}

// Append one diagnostics record to the CSV log on the root rank.
// The header row is written only when the file does not yet exist.
void appendStateDiagnosticsCsv(const std::string& fileName,
                               const StateScanReport& report,
                               const int step,
                               const double time,
                               const std::string& tag,
                               const bool isRoot) {
    // File output is intentionally restricted to the root rank so that one
    // communicator-wide reduced report produces exactly one CSV record.
    if (!isRoot) {
        return;
    }

    // Detect whether this append operation also needs to create the CSV header.
    const bool needHeader = !fileExists(fileName);

    std::ofstream fout(fileName.c_str(), std::ios::out | std::ios::app);
    if (!fout) {
        std::cerr << "[state diagnostics] failed to open CSV file: " << fileName << std::endl;
        return;
    }

    if (needHeader) {
        fout << "tag,step,time,hasNonFinite,hasBadDensity,hasBadPressure,hasBadInternalEnergy,"
             << "nonFiniteCount,badDensityCount,badPressureCount,badInternalEnergyCount,repairedCellCount,"
             << "minRho,minPressure,minInternalEnergy,minRhoI,minRhoJ,minPressureI,minPressureJ,minInternalEnergyI,minInternalEnergyJ\n";
    }

    fout << tag << ','
         << step << ','
         << std::setprecision(16) << time << ','
         << (report.hasNonFinite ? 1 : 0) << ','
         << (report.hasBadDensity ? 1 : 0) << ','
         << (report.hasBadPressure ? 1 : 0) << ','
         << (report.hasBadInternalEnergy ? 1 : 0) << ','
         << report.nonFiniteCount << ','
         << report.badDensityCount << ','
         << report.badPressureCount << ','
         << report.badInternalEnergyCount << ','
         << report.repairedCellCount << ','
         << report.minRho << ','
         << report.minPressure << ','
         << report.minInternalEnergy << ','
         << report.minRhoI << ','
         << report.minRhoJ << ','
         << report.minPressureI << ','
         << report.minPressureJ << ','
         << report.minInternalEnergyI << ','
         << report.minInternalEnergyJ << '\n';
}

} // namespace diagnostics
