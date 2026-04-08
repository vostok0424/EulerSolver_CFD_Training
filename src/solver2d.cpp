// solver2d.cpp
// -----------
// 2D finite-volume Euler solver (MPI-capable).
//
// High-level algorithm per time step:
//   1) Fill ghost cells (MPI halo exchange + physical BCs at global edges)
//   2) Reconstruct conservative face states UL/UR on x- and y-faces (characteristic reconstruction module)
//   3) Compute numerical fluxes on faces via a Riemann solver (flux module)
//   4) Build RHS = -dF/dx - dG/dy
//   5) Advance in time with an explicit time integrator (Euler/RK2/SSPRK3/RK4)
//   6) Periodically gather and write a single merged legacy VTK file (.vtk)
//
// Data layout (2D):
// - U_ is a flattened ghosted array of size (nx+2*ng)*(ny+2*ng)
// - idx(i,j) = i + (nx+2*ng)*j (row-major)
// - Interior cells are (i,j) = (ng..ng+nx-1, ng..ng+ny-1)

#include "solver2d.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

// MPI halo exchange and MPI gather output pack Vec4 into raw doubles. Ensure it is tightly-packed.
static_assert(std::is_trivially_copyable_v<Vec4>, "Vec4 must be trivially copyable for MPI halo exchange");
static_assert(sizeof(Vec4) == 4 * sizeof(double), "Vec4 must be exactly 4 doubles (no padding)");

// Constructor: read configuration, build local subdomain, allocate fields,
// choose numerical methods (flux/reconstruction/time integrator), and initialise
// interior cells either via an IC (ic=...) or setFields.
Solver2D::Solver2D(const Cfg& cfg, const mpi_parallel::MpiParallel& mp)
    : mp_(mp), recon_(cfg)
{
    // --------------------
    // 1) Read case settings
    // --------------------
    // Global mesh sizes (cfg)
    nxGlobal_ = cfg.getInt("nx", 200);
    nyGlobal_ = cfg.getInt("ny", 50);

    // Number of ghost layers. High-order reconstruction requires wider stencils.
    // Ghost layers
    ng_ = cfg.getInt("ng", 2);
    if (ng_ < 1) throw std::runtime_error("ng must be >=1 for 2D solver");

    // Enforce the minimum ghost width required by the selected reconstruction scheme.
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

    // Global physical extents
    x0_ = cfg.getDouble("x0", 0.0);
    x1_ = cfg.getDouble("x1", 1.0);
    y0_ = cfg.getDouble("y0", 0.0);
    y1_ = cfg.getDouble("y1", 0.2);

    // -----------------------------
    // 2) Domain decomposition (MPI)
    // -----------------------------
    // Decompose the global interior grid into a px-by-py Cartesian process grid.
    // This-rank subdomain (interior indices are global 0-based)
    sub_  = mp_.decompose(nxGlobal_, nyGlobal_);
    iBeg_ = sub_.iBeg();
    jBeg_ = sub_.jBeg();
    nx_   = sub_.nx();
    ny_   = sub_.ny();

    // Global spacing is defined by the global mesh and used consistently on all ranks.
    // Global uniform spacing
    const double dxG = (x1_ - x0_) / static_cast<double>(nxGlobal_);
    const double dyG = (y1_ - y0_) / static_cast<double>(nyGlobal_);

    // Local physical extents for this block (so IC/VTK coordinates are correct)
    const double x0Loc = x0_ + static_cast<double>(iBeg_) * dxG;
    const double x1Loc = x0_ + static_cast<double>(iBeg_ + nx_) * dxG;
    const double y0Loc = y0_ + static_cast<double>(jBeg_) * dyG;
    const double y1Loc = y0_ + static_cast<double>(jBeg_ + ny_) * dyG;

    // -----------------------------
    // 3) Numerics and run control
    // -----------------------------
    // Numerics / control
    gamma_       = cfg.getDouble("gamma", 1.4);
    cfl_         = cfg.getDouble("cfl", 0.5);
    finalTime_   = cfg.getDouble("finalTime", 0.2);
    outputEvery_ = cfg.getInt   ("outputEvery", 50);
    writeFinal_  = cfg.getBool  ("writeFinal", true);
    outPrefix_   = cfg.getString("outPrefix", "run2d");

    // -----------------------------
    // 4) Boundary conditions
    // -----------------------------
    // Read the parsed 2D boundary-condition object, including per-side
    // types and any required inlet/outlet parameters.
    bc_ = boundary::read2D(cfg);

    // --------------------
    // 5) Allocate fields
    // --------------------
    // Allocate local arrays including ghost cells.
    // Allocate local fields (including ghosts)
    nxTot_ = nx_ + 2 * ng_;
    nyTot_ = ny_ + 2 * ng_;
    U_.assign(static_cast<size_t>(nxTot_) * static_cast<size_t>(nyTot_), Vec4{});
    RHS_.assign(static_cast<size_t>(nxTot_) * static_cast<size_t>(nyTot_), Vec4{});
    ULxBuf_.resize(static_cast<size_t>(nx_ + 1) * static_cast<size_t>(ny_));
    URxBuf_.resize(static_cast<size_t>(nx_ + 1) * static_cast<size_t>(ny_));
    ULyBuf_.resize(static_cast<size_t>(nx_) * static_cast<size_t>(ny_ + 1));
    URyBuf_.resize(static_cast<size_t>(nx_) * static_cast<size_t>(ny_ + 1));
    FxBuf_.resize(static_cast<size_t>(nx_ + 1) * static_cast<size_t>(ny_));
    GyBuf_.resize(static_cast<size_t>(nx_) * static_cast<size_t>(ny_ + 1));

    // -----------------------------
    // 6) Choose numerical methods
    // -----------------------------
    flux_ = makeFluxD<2>(cfg.getString("flux", "hllc"));
    ti_   = makeTimeIntegratorT<Vec4>(cfg.getString("timeIntegrator", "ssprk3"));

    // -----------------------------
    // 7) Initialise interior fields
    // -----------------------------
    // Only this-rank interior block is initialised here; ghosts are filled later.
    const bool useSetFields = cfg.getBool("setFields.use", false);
    if (!useSetFields) {
        ic_.reset(makeIC2D(cfg.getString("ic", "sodx")));
    }

    // Initialize only this-rank block using local physical extents.
    if (useSetFields) {
        setFields2D(U_, nx_, ny_, ng_, x0Loc, x1Loc, y0Loc, y1Loc, gamma_, cfg);
    } else {
        ic_->apply(U_, nx_, ny_, ng_, x0Loc, x1Loc, y0Loc, y1Loc, gamma_, cfg);
    }
}

// Fill ghost cells.
//
// Step 1: MPI halo exchange across subdomain interfaces (internal boundaries).
// Step 2: Physical boundary conditions on the *global* domain boundaries only.
//         For interior MPI interfaces we temporarily switch that side to
//         boundary::BcType::Internal so boundary::apply2D(...) performs a no-op there.
void Solver2D::applyBC(std::vector<Vec4>& U) const {
    // 1) Exchange halos across MPI subdomain interfaces
    //    NOTE: This assumes Vec4 is a tightly-packed POD of 4 doubles.
    mp_.exchangeHalos2D(reinterpret_cast<double*>(U.data()), nx_, ny_, ng_, 4);

    // Determine whether this rank touches each global boundary.
    // MPI_PROC_NULL means there is no neighbor in that direction.
    const auto nbr = mp_.neighbors();

    boundary::Bc2D bcApply = bc_;
    if (nbr.west != MPI_PROC_NULL) {
        bcApply.left.type = boundary::BcType::Internal;
    }
    if (nbr.east != MPI_PROC_NULL) {
        bcApply.right.type = boundary::BcType::Internal;
    }
    if (nbr.south != MPI_PROC_NULL) {
        bcApply.bottom.type = boundary::BcType::Internal;
    }
    if (nbr.north != MPI_PROC_NULL) {
        bcApply.top.type = boundary::BcType::Internal;
    }

    boundary::apply2D(U, nx_, ny_, ng_, bcApply);
}

// Compute a stable time step from a 2D CFL estimate:
//   dt = CFL / max( (|u|+a)/dx + (|v|+a)/dy )
// We reduce the maximum across all ranks so everyone advances with the same dt.
double Solver2D::computeDt(const std::vector<Vec4>& U) const {
    const double dx = (x1_ - x0_) / nxGlobal_;
    const double dy = (y1_ - y0_) / nyGlobal_;
    double maxS = 1e-14;

    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const auto W = EosIdealGas<2>::consToPrim(U[idx(ng_+i, ng_+j)], gamma_);
            const double a = EosIdealGas<2>::soundSpeed(W, gamma_);
            const double S = (std::abs(W.u[0]) + a) / dx + (std::abs(W.u[1]) + a) / dy;
            maxS = std::max(maxS, S);
        }
    }
    // Global maximum across ranks -> consistent dt for all ranks.
    const double maxSg = mp_.allreduceMax(maxS);
    return cfl_ / maxSg;
}

// Build the semi-discrete RHS: RHS = -dF/dx - dG/dy.
//
// Pipeline:
//   - Characteristic reconstruction produces conservative UL/UR on x-faces and y-faces
//   - Flux module computes numerical fluxes Fx and Gy by a Riemann solver
//   - Finite-volume divergence produces cell-centered RHS
void Solver2D::buildRHS(const std::vector<Vec4>& U, std::vector<Vec4>& RHS) {
    std::fill(RHS.begin(), RHS.end(), Vec4{});

    // Characteristic reconstruction returns conservative face states on x- and y-faces.
    // U must already have valid ghost cells when calling buildRHS.
    recon_.reconstructFacesX(U, nx_, ny_, ng_, gamma_, ULxBuf_, URxBuf_);
    recon_.reconstructFacesY(U, nx_, ny_, ng_, gamma_, ULyBuf_, URyBuf_);

    // X-faces: compute numerical flux using dir=0 (x-normal).
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_ + 1; ++i) {
            const int f = i + (nx_ + 1) * j;
            FxBuf_[f] = flux_->numericalFlux(ULxBuf_[f], URxBuf_[f], 0, gamma_);
        }
    }

    // Y-faces: compute numerical flux using dir=1 (y-normal).
    for (int j = 0; j < ny_ + 1; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const int f = i + nx_ * j;
            GyBuf_[f] = flux_->numericalFlux(ULyBuf_[f], URyBuf_[f], 1, gamma_);
        }
    }

    const double dx = (x1_ - x0_) / nxGlobal_;
    const double dy = (y1_ - y0_) / nyGlobal_;

    // Divergence of fluxes -> RHS (cell-centered):
    //   RHS = -(FxR-FxL)/dx - (GyT-GyB)/dy
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const Vec4& FxL = FxBuf_[i     + (nx_ + 1) * j];
            const Vec4& FxR = FxBuf_[i + 1 + (nx_ + 1) * j];
            const Vec4& GyB = GyBuf_[i + nx_ * j];
            const Vec4& GyT = GyBuf_[i + nx_ * (j + 1)];

            Vec4 R{};
            for (int k = 0; k < 4; ++k) {
                R[k] = -(FxR[k] - FxL[k]) / dx - (GyT[k] - GyB[k]) / dy;
            }
            RHS[idx(ng_ + i, ng_ + j)] = R;
        }
    }
}

StateScanReport Solver2D::scanInteriorStates(const std::vector<Vec4>& U) const {
    StateScanReport report{};
    report.total = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    report.minRho = std::numeric_limits<double>::infinity();
    report.minP   = std::numeric_limits<double>::infinity();

    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const auto result = checkConservative(U[idx(ng_ + i, ng_ + j)], gamma_, stateLimits_);

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
    }

    if (!std::isfinite(report.minRho)) {
        report.minRho = 0.0;
    }
    if (!std::isfinite(report.minP)) {
        report.minP = 0.0;
    }

    return report;
}

void Solver2D::reportStateDiagnostics(int step, double t, const StateScanReport& report) const {
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
        std::cout << "[2D][state] step=" << step
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
            std::cout << "[2D][state] warning: invalid interior states detected by centralized diagnostics.\n";
        }
    }
}

// Periodic output.
//
// We gather all ranks' interior cell data to rank 0 and write ONE merged legacy VTK file.
void Solver2D::writeOutput(int step, double t) const {
    if (outputEvery_ <= 0) return;
    if (step % outputEvery_ != 0) return;

    namespace fs = std::filesystem;
    const fs::path outDir = "solution";

    // Create output directory once on root; barrier keeps ranks in sync.
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
        std::cout << "[2D] Writing merged " << fname.string() << " at t=" << t << "\n";
    }

    // MPI gather + merged VTK write.
    writeVTK2D_GatherMPI(fname.string(),
                         U_, nx_, ny_, ng_,
                         iBeg_, jBeg_,
                         nxGlobal_, nyGlobal_,
                         x0_, x1_,
                         y0_, y1_,
                         gamma_,
                         mp_.cartComm());
}

// Main time-marching loop.
// - Output an initial snapshot at step=0.
// - The time integrator calls rhsFun(...) multiple times per time step.
//   rhsFun ensures ghost cells are valid before characteristic reconstruction/flux evaluation.
void Solver2D::run() {
    double t = 0.0;
    int step = 0;
    std::cout << "[2D][r" << mp_.rank() << "] Starting run, finalTime=" << finalTime_
              << ", global=" << nxGlobal_ << "x" << nyGlobal_
              << ", local=" << nx_ << "x" << ny_
              << ", begin=(" << iBeg_ << "," << jBeg_ << ")\n";

    // Initial RHS build and output.
    applyBC(U_);
    buildRHS(U_, RHS_);
    applyBC(U_); // keep ghosts consistent for output/diagnostics
    if (enableStateDiagnostics_) {
        reportStateDiagnostics(step, t, scanInteriorStates(U_));
    }
    writeOutput(step, t);

    // RHS callback used by RK schemes.
    auto rhsFun = [this](std::vector<Vec4>& Uin, std::vector<Vec4>& Rout){
        // For every stage, ensure ghosts are valid (MPI halos + physical BCs) before characteristic reconstruction/flux.
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
            std::cout << "[2D] Writing merged final " << fname.string() << " at t=" << t << "\n";
        }

        writeVTK2D_GatherMPI(fname.string(),
                             U_, nx_, ny_, ng_,
                             iBeg_, jBeg_,
                             nxGlobal_, nyGlobal_,
                             x0_, x1_,
                             y0_, y1_,
                             gamma_,
                             mp_.cartComm());
    }
}
