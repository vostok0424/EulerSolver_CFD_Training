#include "diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <mpi.h>
#include "cfg.hpp"
#include "mpi_parallel.hpp"

namespace {

// Conservative-state indexing convention used throughout the solver.
constexpr int kRho  = 0;


inline int flatIndex(const int i, const int j, const int nxTot) {
    return j * nxTot + i;
}

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

inline bool fileExists(const std::string& fileName) {
    std::ifstream fin(fileName.c_str());
    return fin.good();
}

} // namespace

namespace diagnostics {

StateDiagnosticsOptions parseStateDiagnosticsOptions(const Cfg& cfg) {
    StateDiagnosticsOptions opts;

    opts.enable = cfg.getBool("stateDiagnostics.enable", false);
    opts.csvFile = cfg.getString("stateDiagnostics.csv", "state_diagnostics.csv");
    opts.printToStdout = cfg.getBool("stateDiagnostics.printToStdout", true);
    opts.includePerStepSummary = cfg.getBool("stateDiagnostics.includePerStepSummary", true);

    return opts;
}

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

void accumulateStateScanReport(StateScanReport& dst,
                               const StateScanReport& src) {
    dst.hasNonFinite = dst.hasNonFinite || src.hasNonFinite;
    dst.hasBadDensity = dst.hasBadDensity || src.hasBadDensity;
    dst.hasBadPressure = dst.hasBadPressure || src.hasBadPressure;
    dst.hasBadInternalEnergy = dst.hasBadInternalEnergy || src.hasBadInternalEnergy;

    dst.nonFiniteCount += src.nonFiniteCount;
    dst.badDensityCount += src.badDensityCount;
    dst.badPressureCount += src.badPressureCount;
    dst.badInternalEnergyCount += src.badInternalEnergyCount;
    dst.repairedCellCount += src.repairedCellCount;

    if (src.initialized) {
        if (!dst.initialized || src.minRho < dst.minRho) {
            dst.minRho = src.minRho;
            dst.minRhoI = src.minRhoI;
            dst.minRhoJ = src.minRhoJ;
        }
        if (!dst.initialized || src.minPressure < dst.minPressure) {
            dst.minPressure = src.minPressure;
            dst.minPressureI = src.minPressureI;
            dst.minPressureJ = src.minPressureJ;
        }
        if (!dst.initialized || src.minInternalEnergy < dst.minInternalEnergy) {
            dst.minInternalEnergy = src.minInternalEnergy;
            dst.minInternalEnergyI = src.minInternalEnergyI;
            dst.minInternalEnergyJ = src.minInternalEnergyJ;
        }
        dst.initialized = true;
    }
}

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

    // The global minimum values are reduced exactly, but the associated index
    // locations are kept from the local report only when they match the reduced
    // minima. This avoids introducing solver-wide coordinate assumptions here.
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

bool hasStateFailure(const StateScanReport& report) {
    return report.hasNonFinite ||
           report.hasBadDensity ||
           report.hasBadPressure ||
           report.hasBadInternalEnergy;
}

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

void appendStateDiagnosticsCsv(const std::string& fileName,
                               const StateScanReport& report,
                               const int step,
                               const double time,
                               const std::string& tag,
                               const bool isRoot) {
    if (!isRoot) {
        return;
    }

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
