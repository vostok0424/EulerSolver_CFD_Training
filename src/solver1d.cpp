// solver1d.cpp
// -----------
// 1D finite-volume Euler solver (MPI-capable).
//
// High-level algorithm per time step:
//   1) Fill ghost cells (MPI halo exchange + physical BCs at global edges)
//   2) Reconstruct face states UL/UR from cell-centered U
//   3) Compute numerical fluxes at faces via a Riemann solver (flux module)
//   4) Build RHS = -dF/dx
//   5) Advance in time with an explicit time integrator (Euler/RK2/SSPRK3/RK4)
//   6) Periodically gather and write a single merged legacy VTK file (.vtk)
//
// Data layout (1D):
// - U_ is a 1D array including ghost cells, size nxTot = nx + 2*ng
// - Interior cells are U_[ng .. ng+nx-1]
// - Left ghosts are U_[0 .. ng-1], right ghosts are U_[ng+nx .. nxTot-1]

#include "solver1d.hpp"
#include "io1d.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <type_traits>

namespace {

constexpr const char* kOutputDir1D = "solution";

struct ReducedStateScanReport {
    unsigned long long total{0};
    unsigned long long nonFiniteCount{0};
    unsigned long long badDensityCount{0};
    unsigned long long badPressureCount{0};
    unsigned long long badInternalEnergyCount{0};
    double minRho{0.0};
    double minP{0.0};
};

void accumulateStateFailure(StateScanReport& report, StateStatus status) {
    switch (status) {
        case StateStatus::NonFinite:
            ++report.nonFiniteCount;
            break;
        case StateStatus::NegativeDensity:
        case StateStatus::DensityTooSmall:
            ++report.badDensityCount;
            break;
        case StateStatus::NegativePressure:
        case StateStatus::PressureTooSmall:
            ++report.badPressureCount;
            break;
        case StateStatus::NegativeInternalEnergy:
            ++report.badInternalEnergyCount;
            break;
        case StateStatus::Ok:
        default:
            break;
    }
}

ReducedStateScanReport reduceStateScanReportMPI(const StateScanReport& report, MPI_Comm comm) {
    const unsigned long long localCounts[5] = {
        static_cast<unsigned long long>(report.total),
        static_cast<unsigned long long>(report.nonFiniteCount),
        static_cast<unsigned long long>(report.badDensityCount),
        static_cast<unsigned long long>(report.badPressureCount),
        static_cast<unsigned long long>(report.badInternalEnergyCount)
    };
    unsigned long long globalCounts[5] = {0, 0, 0, 0, 0};

    MPI_Allreduce(localCounts, globalCounts, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);

    const double localMin[2] = {report.minRho, report.minP};
    double globalMin[2] = {0.0, 0.0};
    MPI_Allreduce(localMin, globalMin, 2, MPI_DOUBLE, MPI_MIN, comm);

    ReducedStateScanReport reduced{};
    reduced.total = globalCounts[0];
    reduced.nonFiniteCount = globalCounts[1];
    reduced.badDensityCount = globalCounts[2];
    reduced.badPressureCount = globalCounts[3];
    reduced.badInternalEnergyCount = globalCounts[4];
    reduced.minRho = globalMin[0];
    reduced.minP = globalMin[1];
    return reduced;
}

void ensureOutputDirectoryExists(const mpi_parallel::MpiParallel& mp) {
    namespace fs = std::filesystem;
    if (mp.isRoot()) {
        fs::create_directories(kOutputDir1D);
    }
    mp.barrier();
}

std::string makeStepOutputPath(const std::string& outPrefix, int step, double t) {
    std::ostringstream stepSS;
    stepSS << std::setw(4) << std::setfill('0') << step;

    std::ostringstream tSS;
    tSS << std::fixed << std::setprecision(6) << t;

    const std::filesystem::path outDir = kOutputDir1D;
    return (outDir / (outPrefix + "_step" + stepSS.str() + "_t=" + tSS.str() + ".vtk")).string();
}

std::string makeFinalOutputPath(const std::string& outPrefix, double t) {
    std::ostringstream tSS;
    tSS << std::fixed << std::setprecision(6) << t;

    const std::filesystem::path outDir = kOutputDir1D;
    return (outDir / (outPrefix + "_final_t=" + tSS.str() + ".vtk")).string();
}

void writeMergedVtkFile1D(const mpi_parallel::MpiParallel& mp,
                          const std::string& path,
                          const std::string& label,
                          double t,
                          const std::vector<Vec3>& U,
                          int nx,
                          int ng,
                          int iBeg,
                          int nxGlobal,
                          double x0,
                          double x1,
                          double gamma) {
    if (mp.isRoot()) {
        std::cout << "[1D] Writing merged " << label << " " << path << " at t=" << t << "\n";
    }

    writeVTK1D_GatherMPI(path,
                         U, nx, ng,
                         iBeg,
                         nxGlobal,
                         x0, x1,
                         gamma,
                         mp.cartComm());
}

} // namespace

// MPI halo exchange and MPI gather output pack Vec3 into raw doubles.
// Ensure Vec3 is trivially copyable and tightly packed (no padding).
static_assert(std::is_trivially_copyable_v<Vec3>, "Vec3 must be trivially copyable");
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be exactly 3 doubles (no padding)");

// Constructor: read configuration, build local subdomain, allocate fields,
// and initialise the interior solution either from an IC (ic=...) or from setFields.
Solver1D::Solver1D(const Cfg& cfg, const mpi_parallel::MpiParallel& mp)
    : mp_(mp), recon_(cfg)
{
    // --------------------
    // 1) Read case settings
    // --------------------
    // Global mesh size (cfg)
    grid_.nxGlobal = cfg.getInt("nx", 200);

    // Number of ghost layers. High-order reconstruction requires wider stencils.
    grid_.ng = cfg.getInt("ng", 2);
    if (grid_.ng < 1) throw std::runtime_error("ng must be >=1 for 1D solver");

    // Global physical extents
    grid_.x0 = cfg.getDouble("x0", 0.0);
    grid_.x1 = cfg.getDouble("x1", 1.0);

    // Numerics / control
    gamma_       = cfg.getDouble("gamma", 1.4);
    cfl_         = cfg.getDouble("cfl", 0.5);
    finalTime_   = cfg.getDouble("finalTime", 0.2);
    outputEvery_ = cfg.getInt   ("outputEvery", 50);
    writeFinal_  = cfg.getBool  ("writeFinal", true);
    outPrefix_   = cfg.getString("outPrefix", "run1d");

    // Read the parsed 1D boundary-condition object, including per-side
    // types and any required inlet/outlet parameters.
    bc_ = boundary::read1D(cfg);

    // Enforce a minimum number of ghost layers for the selected reconstruction scheme.
    const int ngReq = recon::requiredGhostCells(recon_.options().scheme);
    if (grid_.ng < ngReq) {
        throw std::runtime_error("ng is too small for the selected reconstruction.scheme");
    }

    // Shared state-layer thresholds and optional diagnostic controls.
    stateLimits_ = recon_.options().stateLimits();
    enableStateDiagnostics_ = cfg.getBool("stateDiagnostics.enable", true);
    stateDiagCsvPath_ = cfg.getString("stateDiagnostics.csv",
                                      "solution/" + outPrefix_ + "_state_diagnostics.csv");

    // -----------------------------
    // 2) Domain decomposition (MPI)
    // -----------------------------
    // Decompose the global interior grid into contiguous x-blocks across px ranks.
    sub_  = mp_.decompose1D(grid_.nxGlobal);
    grid_.iBeg = sub_.iBeg();
    grid_.nx   = sub_.nx();

    // Global grid spacing is defined by the global mesh and used consistently on all ranks.
    const double dxG = grid_.dx();

    // Local physical extents for this block
    const double x0Loc = grid_.x0 + static_cast<double>(grid_.iBeg) * dxG;
    const double x1Loc = grid_.x0 + static_cast<double>(grid_.iBeg + grid_.nx) * dxG;

    // --------------------
    // 3) Allocate solution
    // --------------------
    // Allocate local arrays including ghost cells.
    grid_.nxTot = grid_.nx + 2 * grid_.ng;
    U_.assign(grid_.nxTot, Vec3{});
    RHS_.assign(grid_.nxTot, Vec3{});
    faces_.resize(grid_.nx);

    // -----------------------------
    // 4) Choose numerical methods
    // -----------------------------
    // Flux (Riemann solver) and explicit time integrator are selected by name.
    flux_ = makeFluxD<1>(cfg.getString("flux", "hllc"));
    ti_   = makeTimeIntegratorT<Vec3>(cfg.getString("timeIntegrator", "ssprk3"));

    // -----------------------------
    // 5) Initialise interior fields
    // -----------------------------
    const bool useSetFields = cfg.getBool("setFields.use", false);
    if (!useSetFields) {
        ic_.reset(makeIC1D(cfg.getString("ic", "riemann")));
    }

    // Initialise only this-rank interior block using local physical extents.
    // Ghost cells are filled later by applyBC().
    if (useSetFields) {
        setFields1D(U_, grid_.nx, grid_.ng, x0Loc, x1Loc, gamma_, cfg);
    } else {
        ic_->apply(U_, grid_.nx, grid_.ng, x0Loc, x1Loc, gamma_, cfg);
    }
}

// Fill ghost cells.
//
// Step 1: MPI halo exchange across subdomain interfaces (internal boundaries).
// Step 2: Physical boundary conditions on the *global* left/right edges only.
//         Ranks that are not on a global edge temporarily switch that side to
//         boundary::BcType::Internal so boundary::apply1D(...) leaves those ghosts untouched.
void Solver1D::applyBC(std::vector<Vec3>& U) const {
    // 1) Exchange halos across MPI subdomain interfaces
    mp_.exchangeHalos1D(reinterpret_cast<double*>(U.data()), grid_.nx, grid_.ng, 3);

    // Determine whether this rank touches the global domain boundary.
    // MPI_PROC_NULL indicates there is no neighbor in that direction.
    const auto nbr = mp_.neighbors();
    boundary::Bc1D bcApply = bc_;
    if (nbr.west != MPI_PROC_NULL) {
        bcApply.left.type = boundary::BcType::Internal;
    }
    if (nbr.east != MPI_PROC_NULL) {
        bcApply.right.type = boundary::BcType::Internal;
    }

    boundary::apply1D(U, grid_.nx, grid_.ng, bcApply);
}

// Compute a stable time step from the CFL condition:
//   dt = CFL * dx / max(|u| + a)
// where a is the sound speed. We reduce the maximum wave speed across all ranks.
double Solver1D::computeDt(const std::vector<Vec3>& U) const {
    const double dx = grid_.dx();
    double maxChar = 1e-14;
    for (int i = 0; i < grid_.nx; ++i) {
        const auto Wi = evalFlowVars(U[grid_.ng + i], gamma_);
        maxChar = std::max(maxChar, std::abs(Wi.u) + Wi.a);
    }
    // Global maximum across ranks so all ranks advance with the same dt.
    const double maxCharG = mp_.allreduceMax(maxChar);
    return cfl_ * dx / maxCharG;
}

// Build the semi-discrete RHS: RHS = -dF/dx.
//
// Pipeline:
//   - Reconstruction module produces UL/UR at faces
//   - Flux module computes F at faces via a Riemann solver
//   - Finite-volume divergence produces the cell RHS
void Solver1D::buildRHS(const std::vector<Vec3>& U, std::vector<Vec3>& RHS) {
    if (static_cast<int>(RHS.size()) != grid_.nxTot) {
        RHS.assign(grid_.nxTot, Vec3{});
    } else {
        std::fill(RHS.begin(), RHS.end(), Vec3{});
    }

    // Reconstruction is characteristic-only. The reconstruction module returns
    // conservative face states UL/UR, with characteristic projection details
    // handled internally.
    recon_.reconstructFaces(U, grid_.nx, grid_.ng, gamma_, faces_.UL, faces_.UR);

    // Compute numerical flux at each face.
    for (int i = 0; i < grid_.nx + 1; ++i) {
        const int f = idxFace(i);
        faces_.F[f] = flux_->numericalFlux(faces_.UL[f], faces_.UR[f], 0, gamma_);
    }

    // Divergence of flux: RHS_i = -(F_{i+1/2} - F_{i-1/2}) / dx
    const double dx = grid_.dx();
    for (int i = 0; i < grid_.nx; ++i) {
        for (int k = 0; k < 3; ++k) {
            RHS[grid_.ng + i][k] = -(faces_.F[idxFace(i + 1)][k] - faces_.F[idxFace(i)][k]) / dx;
        }
    }
}

StateScanReport Solver1D::scanInteriorStates(const std::vector<Vec3>& U) const {
    StateScanReport report{};
    report.total = static_cast<std::size_t>(grid_.nx);
    report.minRho = std::numeric_limits<double>::infinity();
    report.minP   = std::numeric_limits<double>::infinity();

    for (int i = 0; i < grid_.nx; ++i) {
        const auto result = checkConservative(U[grid_.ng + i], gamma_, stateLimits_);

        report.minRho = std::min(report.minRho, result.rho);
        report.minP   = std::min(report.minP,   result.p);

        if (!result.ok) {
            accumulateStateFailure(report, result.status);
        }
    }

    if (!std::isfinite(report.minRho)) {
        report.minRho = 0.0;
    }
    if (!std::isfinite(report.minP)) {
        report.minP = 0.0;
    }

    return report;
}

bool Solver1D::shouldWriteStepOutput(int step) const {
    return (outputEvery_ > 0) && (step % outputEvery_ == 0);
}

bool Solver1D::shouldRecordStateDiagnostics(int step) const {
    if (!enableStateDiagnostics_) return false;
    return shouldWriteStepOutput(step);
}

void Solver1D::appendStateDiagnosticsCsv(int step, double t, const StateScanReport& report,
                                         const std::string& tag) const {
    if (!enableStateDiagnostics_) return;

    const ReducedStateScanReport global = reduceStateScanReportMPI(report, mp_.cartComm());

    if (!mp_.isRoot()) return;

    namespace fs = std::filesystem;
    const fs::path csvPath(stateDiagCsvPath_);
    if (csvPath.has_parent_path()) {
        fs::create_directories(csvPath.parent_path());
    }

    const bool needHeader = stateDiagWriteHeader_ || !fs::exists(csvPath);

    std::ofstream ofs(csvPath, std::ios::app);
    if (!ofs) {
        std::cerr << "[1D][state] failed to open CSV: " << csvPath.string() << "\n";
        return;
    }

    if (needHeader) {
        ofs << "tag,step,time,cells,minRho,minP,nonFinite,badDensity,badPressure,badEint\n";
        stateDiagWriteHeader_ = false;
    }

    ofs << tag << ","
        << step << ","
        << std::setprecision(16) << t << ","
        << global.total << ","
        << global.minRho << ","
        << global.minP << ","
        << global.nonFiniteCount << ","
        << global.badDensityCount << ","
        << global.badPressureCount << ","
        << global.badInternalEnergyCount << "\n";
}

// Periodic output.
//
// We gather all ranks' interior cell data to rank 0 and write ONE merged legacy VTK file.
void Solver1D::writeOutput(int step, double t) const {
    if (outputEvery_ <= 0) return;
    if (step % outputEvery_ != 0) return;

    ensureOutputDirectoryExists(mp_);
    const std::string fname = makeStepOutputPath(outPrefix_, step, t);
    writeMergedVtkFile1D(mp_, fname, "step", t,
                         U_, grid_.nx, grid_.ng,
                         grid_.iBeg,
                         grid_.nxGlobal,
                         grid_.x0, grid_.x1,
                         gamma_);
}

// Main time-marching loop.
// - We output an initial field at step=0.
// - The time integrator calls rhsFun(...) for each stage; rhsFun ensures ghost
//   cells are valid before reconstruction/flux evaluation.
void Solver1D::run() {
    double t = 0.0;
    int step = 0;
    std::cout << "[1D][r" << mp_.rank() << "] Starting run, finalTime=" << finalTime_
              << ", global=" << grid_.nxGlobal
              << ", local=" << grid_.nx
              << ", iBeg=" << grid_.iBeg << "\n";

    // Initial RHS build and output.
    applyBC(U_);
    buildRHS(U_, RHS_);
    applyBC(U_); // keep ghosts consistent for output
    if (shouldRecordStateDiagnostics(step)) {
        appendStateDiagnosticsCsv(step, t, scanInteriorStates(U_), "initial");
    }
    writeOutput(step, t);

    // RHS callback used by RK schemes. Called multiple times per time step.
    auto rhsFun = [this](std::vector<Vec3>& Uin, std::vector<Vec3>& Rout) {
        // For every stage, ensure ghosts are valid before characteristic reconstruction/flux.
        applyBC(Uin);
        buildRHS(Uin, Rout);
    };

    // Advance until finalTime (last step is clipped to hit finalTime exactly).
    while (t < finalTime_) {
        double dt = computeDt(U_);
        if (t + dt > finalTime_) dt = finalTime_ - t;

        ti_->step(U_, dt, rhsFun);
        t += dt;
        ++step;
        applyBC(U_);
        if (shouldRecordStateDiagnostics(step)) {
            appendStateDiagnosticsCsv(step, t, scanInteriorStates(U_), "output");
        }
        writeOutput(step, t);
    }

    // Optional final snapshot.
    if (writeFinal_) {
        ensureOutputDirectoryExists(mp_);

        applyBC(U_);
        const bool finalAlreadyRecorded = shouldRecordStateDiagnostics(step);
        if (enableStateDiagnostics_ && !finalAlreadyRecorded) {
            appendStateDiagnosticsCsv(step, t, scanInteriorStates(U_), "final");
        }

        const std::string fname = makeFinalOutputPath(outPrefix_, t);
        writeMergedVtkFile1D(mp_, fname, "final", t,
                             U_, grid_.nx, grid_.ng,
                             grid_.iBeg,
                             grid_.nxGlobal,
                             grid_.x0, grid_.x1,
                             gamma_);
    }
}
