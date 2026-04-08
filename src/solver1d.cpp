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
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

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
    nxGlobal_ = cfg.getInt("nx", 200);

    // Number of ghost layers. High-order reconstruction requires wider stencils.
    ng_ = cfg.getInt("ng", 2);
    if (ng_ < 1) throw std::runtime_error("ng must be >=1 for 1D solver");

    // Global physical extents
    x0_ = cfg.getDouble("x0", 0.0);
    x1_ = cfg.getDouble("x1", 1.0);

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
    if (ng_ < ngReq) {
        throw std::runtime_error("ng is too small for the selected reconstruction.scheme");
    }

    // Shared state-layer thresholds and optional diagnostic controls.
    stateLimits_ = recon_.options().stateLimits();
    enableStateDiagnostics_ = cfg.getBool("stateDiagnostics.enable", true);
    stateDiagnosticsEvery_  = cfg.getInt ("stateDiagnostics.every", 1);
    if (stateDiagnosticsEvery_ < 1) {
        stateDiagnosticsEvery_ = 1;
    }

    // -----------------------------
    // 2) Domain decomposition (MPI)
    // -----------------------------
    // Decompose the global interior grid into contiguous x-blocks across px ranks.
    sub_  = mp_.decompose1D(nxGlobal_);
    iBeg_ = sub_.iBeg();
    nx_   = sub_.nx();

    // Global grid spacing is defined by the global mesh and used consistently on all ranks.
    const double dxG = (x1_ - x0_) / static_cast<double>(nxGlobal_);

    // Local physical extents for this block
    const double x0Loc = x0_ + static_cast<double>(iBeg_) * dxG;
    const double x1Loc = x0_ + static_cast<double>(iBeg_ + nx_) * dxG;

    // --------------------
    // 3) Allocate solution
    // --------------------
    // Allocate local arrays including ghost cells.
    nxTot_ = nx_ + 2 * ng_;
    U_.assign(nxTot_, Vec3{});
    RHS_.assign(nxTot_, Vec3{});

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
        ic_.reset(makeIC1D(cfg.getString("ic", "sod")));
    }

    // Initialise only this-rank interior block using local physical extents.
    // Ghost cells are filled later by applyBC().
    if (useSetFields) {
        setFields1D(U_, nx_, ng_, x0Loc, x1Loc, gamma_, cfg);
    } else {
        ic_->apply(U_, nx_, ng_, x0Loc, x1Loc, gamma_, cfg);
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
    mp_.exchangeHalos1D(reinterpret_cast<double*>(U.data()), nx_, ng_, 3);

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

    boundary::apply1D(U, nx_, ng_, bcApply);
}

// Compute a stable time step from the CFL condition:
//   dt = CFL * dx / max(|u| + a)
// where a is the sound speed. We reduce the maximum wave speed across all ranks.
double Solver1D::computeDt(const std::vector<Vec3>& U) const {
    const double dx = (x1_ - x0_) / nxGlobal_;
    double maxChar = 1e-14;
    for (int i = 0; i < nx_; ++i) {
        const auto Wi = evalFlowVars(U[ng_ + i], gamma_);
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
void Solver1D::buildRHS(const std::vector<Vec3>& U, std::vector<Vec3>& RHS) const {
    if (static_cast<int>(RHS.size()) != nx_ + 2 * ng_) {
        RHS.assign(nx_ + 2 * ng_, Vec3{});
    } else {
        std::fill(RHS.begin(), RHS.end(), Vec3{});
    }

    // Reconstruction is characteristic-only. The reconstruction module returns
    // conservative face states UL/UR, with characteristic projection details
    // handled internally.
    // Reconstruct left/right conservative states at faces i=0..nx.
    std::vector<Vec3> UL, UR;
    recon_.reconstructFaces(U, nx_, ng_, gamma_, UL, UR);

    // Compute numerical flux at each face.
    std::vector<Vec3> F(nx_ + 1);
    for (int i = 0; i < nx_ + 1; ++i) {
        F[i] = flux_->numericalFlux(UL[i], UR[i], 0, gamma_);
    }

    // Divergence of flux: RHS_i = -(F_{i+1/2} - F_{i-1/2}) / dx
    const double dx = (x1_ - x0_) / nxGlobal_;
    for (int i = 0; i < nx_; ++i) {
        for (int k = 0; k < 3; ++k) {
            RHS[ng_ + i][k] = -(F[i + 1][k] - F[i][k]) / dx;
        }
    }
}

StateScanReport Solver1D::scanInteriorStates(const std::vector<Vec3>& U) const {
    StateScanReport report{};
    report.total = static_cast<std::size_t>(nx_);
    report.minRho = std::numeric_limits<double>::infinity();
    report.minP   = std::numeric_limits<double>::infinity();

    for (int i = 0; i < nx_; ++i) {
        const auto result = checkConservative(U[ng_ + i], gamma_, stateLimits_);

        report.minRho = std::min(report.minRho, result.rho);
        report.minP   = std::min(report.minP,   result.p);

        if (!result.ok) {
            switch (result.status) {
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
    }

    if (!std::isfinite(report.minRho)) {
        report.minRho = 0.0;
    }
    if (!std::isfinite(report.minP)) {
        report.minP = 0.0;
    }

    return report;
}

void Solver1D::reportStateDiagnostics(int step, double t, const StateScanReport& report) const {
    if (!enableStateDiagnostics_) return;
    if (stateDiagnosticsEvery_ <= 0) return;
    if (step % stateDiagnosticsEvery_ != 0) return;

    const unsigned long long localCounts[5] = {
        static_cast<unsigned long long>(report.total),
        static_cast<unsigned long long>(report.nonFiniteCount),
        static_cast<unsigned long long>(report.badDensityCount),
        static_cast<unsigned long long>(report.badPressureCount),
        static_cast<unsigned long long>(report.badInternalEnergyCount)
    };
    unsigned long long globalCounts[5] = {0, 0, 0, 0, 0};

    MPI_Allreduce(localCounts, globalCounts, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mp_.cartComm());

    const double localMin[2] = {report.minRho, report.minP};
    double globalMin[2] = {0.0, 0.0};
    MPI_Allreduce(localMin, globalMin, 2, MPI_DOUBLE, MPI_MIN, mp_.cartComm());

    if (mp_.isRoot()) {
        std::cout << "[1D][state] step=" << step
                  << ", t=" << t
                  << ", cells=" << globalCounts[0]
                  << ", minRho=" << globalMin[0]
                  << ", minP=" << globalMin[1]
                  << ", nonFinite=" << globalCounts[1]
                  << ", badDensity=" << globalCounts[2]
                  << ", badPressure=" << globalCounts[3]
                  << ", badEint=" << globalCounts[4]
                  << "\n";
        if (globalCounts[1] + globalCounts[2] + globalCounts[3] + globalCounts[4] > 0) {
            std::cout << "[1D][state] warning: invalid interior states detected by centralized diagnostics.\n";
        }
    }
}

// Periodic output.
//
// We gather all ranks' interior cell data to rank 0 and write ONE merged legacy VTK file.
void Solver1D::writeOutput(int step, double t) const {
    if (outputEvery_ <= 0) return;
    if (step % outputEvery_ != 0) return;

    // Create output directory once on root.
    namespace fs = std::filesystem;
    const fs::path outDir = "solution";
    if (mp_.isRoot()) {
        fs::create_directories(outDir);
    }
    mp_.barrier();

    // File naming convention: <prefix>_stepXXXX_t=TTTTTT.vtk
    // Step is zero-padded for easy sorting.
    std::ostringstream stepSS;
    stepSS << std::setw(4) << std::setfill('0') << step;

    std::ostringstream tSS;
    tSS << std::fixed << std::setprecision(6) << t;

    const fs::path fname = outDir / (outPrefix_ + "_step" + stepSS.str() + "_t=" + tSS.str() + ".vtk");
    if (mp_.isRoot()) {
        std::cout << "[1D] Writing merged " << fname.string() << " at t=" << t << "\n";
    }

    // MPI gather + merged VTK write.
    writeVTK1D_GatherMPI(fname.string(),
                         U_, nx_, ng_,
                         iBeg_,
                         nxGlobal_,
                         x0_, x1_,
                         gamma_,
                         mp_.cartComm());
}

// Main time-marching loop.
// - We output an initial field at step=0.
// - The time integrator calls rhsFun(...) for each stage; rhsFun ensures ghost
//   cells are valid before reconstruction/flux evaluation.
void Solver1D::run() {
    double t = 0.0;
    int step = 0;
    std::cout << "[1D][r" << mp_.rank() << "] Starting run, finalTime=" << finalTime_
              << ", global=" << nxGlobal_
              << ", local=" << nx_
              << ", iBeg=" << iBeg_ << "\n";

    // Initial RHS build and output.
    applyBC(U_);
    buildRHS(U_, RHS_);
    applyBC(U_); // keep ghosts consistent for output
    writeOutput(step, t);
    if (enableStateDiagnostics_) {
        reportStateDiagnostics(step, t, scanInteriorStates(U_));
    }

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
        if (enableStateDiagnostics_) {
            reportStateDiagnostics(step, t, scanInteriorStates(U_));
        }
        writeOutput(step, t);
    }

    // Optional final snapshot.
    if (writeFinal_) {
        namespace fs = std::filesystem;
        const fs::path outDir = "solution";
        if (mp_.isRoot()) {
            fs::create_directories(outDir);
        }
        mp_.barrier();

        applyBC(U_);
        if (enableStateDiagnostics_) {
            reportStateDiagnostics(step, t, scanInteriorStates(U_));
        }
        std::ostringstream tSS;
        tSS << std::fixed << std::setprecision(6) << t;

        const fs::path fname = outDir / (outPrefix_ + "_final_t=" + tSS.str() + ".vtk");
        if (mp_.isRoot()) {
            std::cout << "[1D] Writing merged final " << fname.string() << " at t=" << t << "\n";
        }

        writeVTK1D_GatherMPI(fname.string(),
                             U_, nx_, ng_,
                             iBeg_,
                             nxGlobal_,
                             x0_, x1_,
                             gamma_,
                             mp_.cartComm());
    }
}
