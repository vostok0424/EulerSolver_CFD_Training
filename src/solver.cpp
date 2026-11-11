// solver.cpp
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

#include "solver.hpp"
#include <algorithm>
#include <cctype>
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

constexpr const char* kOutputDir2D = "solution";

VTKDataOutputType parseVTKDataOutputType(const std::string& value) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if ( s == "point" ) {
        return VTKDataOutputType::Point;
    }
    if ( s == "cell" ) {
        return VTKDataOutputType::Cell;
    }

    throw std::runtime_error("Unknown output.vtkDataOutputType: " + value +
                             " (expected point or cell)");
}


const char* vtkDataOutputTypeName(VTKDataOutputType outputType) {
    switch (outputType) {
        case VTKDataOutputType::Point: return "POINT_DATA";
        case VTKDataOutputType::Cell:  return "CELL_DATA";
    }
    return "UNKNOWN";
}

void ensureOutputDirectoryExists(const mpi_parallel::MpiParallel& mp) {
    namespace fs = std::filesystem;
    if (mp.isRoot()) {
        fs::create_directories(kOutputDir2D);
    }
    mp.barrier();
}

std::string makeStepOutputPath(const std::string& outPrefix, int step, double t) {
    std::ostringstream stepSS;
    stepSS << std::setw(4) << std::setfill('0') << step;

    std::ostringstream tSS;
    tSS << std::fixed << std::setprecision(6) << t;

    const std::filesystem::path outDir = kOutputDir2D;
    return (outDir / (outPrefix + "_step" + stepSS.str() + "_t=" + tSS.str() + ".vtk")).string();
}

std::string makeFinalOutputPath(const std::string& outPrefix, double t) {
    std::ostringstream tSS;
    tSS << std::fixed << std::setprecision(6) << t;

    const std::filesystem::path outDir = kOutputDir2D;
    return (outDir / (outPrefix + "_final_t=" + tSS.str() + ".vtk")).string();
}

void writeMergedVtkFile2D(const mpi_parallel::MpiParallel& mp,
                          const std::string& path,
                          const std::string& label,
                          double t,
                          const std::vector<Vec4>& U,
                          int nx,
                          int ny,
                          int ng,
                          int iBeg,
                          int jBeg,
                          int nxGlobal,
                          int nyGlobal,
                          double x0,
                          double x1,
                          double y0,
                          double y1,
                          double gamma,
                          VTKDataOutputType dataOutputType) {
    if (mp.isRoot()) {
        std::cout << "[2D] Writing merged " << label << " " << path
                  << " at t=" << t
                  << " as " << vtkDataOutputTypeName(dataOutputType) << "\n";
    }

    switch (dataOutputType) {
        case VTKDataOutputType::Point:
            writeVTK2D_PointData_GatherMPI(path,
                                           U, nx, ny, ng,
                                           iBeg, jBeg,
                                           nxGlobal, nyGlobal,
                                           x0, x1,
                                           y0, y1,
                                           gamma,
                                           mp.cartComm());
            break;

        case VTKDataOutputType::Cell:
            writeVTK2D_CellData_GatherMPI(path,
                                          U, nx, ny, ng,
                                          iBeg, jBeg,
                                          nxGlobal, nyGlobal,
                                          x0, x1, y0, y1, gamma,
                                          mp.cartComm());
            break;
    }
}

} // namespace

// MPI halo exchange and MPI gather output pack Vec4 into raw doubles. Ensure it is tightly-packed.
static_assert(std::is_trivially_copyable_v<Vec4>, "Vec4 must be trivially copyable for MPI halo exchange");
static_assert(sizeof(Vec4) == 4 * sizeof(double), "Vec4 must be exactly 4 doubles (no padding)");

// Constructor: read configuration, build local subdomain, allocate fields,
// choose numerical methods (flux/reconstruction/time integrator), and initialize
// interior cells through setFields.
Solver::Solver(const Cfg& cfg, const mpi_parallel::MpiParallel& mp)
    : mp_(mp), recon_(cfg)
{
    // --------------------
    // 1) Read case settings
    // --------------------
    // Global mesh sizes (cfg)
    grid_.nxGlobal = cfg.getInt("nx", 200);
    grid_.nyGlobal = cfg.getInt("ny", 50);

    // Number of ghost layers. High-order reconstruction requires wider stencils.
    // Ghost layers
    grid_.ng = cfg.getInt("ng", 2);
    if (grid_.ng < 1) throw std::runtime_error("ng must be >=1 for 2D solver");

    // Enforce the minimum ghost width required by the selected reconstruction scheme.
    const int ngReq = recon::requiredGhostCells(recon_.options().scheme);
    if (grid_.ng < ngReq) {
        throw std::runtime_error("ng is too small for the selected reconstruction.scheme");
    }

    // Global physical extents
    grid_.x0 = cfg.getDouble("x0", 0.0);
    grid_.x1 = cfg.getDouble("x1", 1.0);
    grid_.y0 = cfg.getDouble("y0", 0.0);
    grid_.y1 = cfg.getDouble("y1", 0.2);

    if (grid_.nxGlobal <= 0 || grid_.nyGlobal <= 0 ||
        grid_.x1 <= grid_.x0 || grid_.y1 <= grid_.y0 ||
        !std::isfinite(grid_.dx()) || !std::isfinite(grid_.dy()) ||
        grid_.dx() <= 0.0 || grid_.dy() <= 0.0)
        throw std::runtime_error("Invalid grid dimensions or physical extents");

    // -----------------------------
    // 2) Domain decomposition (MPI)
    // -----------------------------
    // Decompose the global interior grid into a px-by-py Cartesian process grid.
    // This-rank subdomain (interior indices are global 0-based)
    sub_  = mp_.decompose(grid_.nxGlobal, grid_.nyGlobal);
    grid_.iBeg = sub_.iBeg();
    grid_.jBeg = sub_.jBeg();
    grid_.nx   = sub_.nx();
    grid_.ny   = sub_.ny();

    // Global spacing is defined by the global mesh and used consistently on all ranks.
    // Global uniform spacing
    const double dxG = grid_.dx();
    const double dyG = grid_.dy();

    // Local physical extents for this block, used by setFields and VTK output.
    const double x0Loc = grid_.x0 + static_cast<double>(grid_.iBeg) * dxG;
    const double x1Loc = grid_.x0 + static_cast<double>(grid_.iBeg + grid_.nx) * dxG;
    const double y0Loc = grid_.y0 + static_cast<double>(grid_.jBeg) * dyG;
    const double y1Loc = grid_.y0 + static_cast<double>(grid_.jBeg + grid_.ny) * dyG;

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
    if (gamma_ <= 1.0 || cfl_ <= 0.0 || finalTime_ < 0.0)
        throw std::runtime_error("Require gamma > 1, cfl > 0, finalTime >= 0");
    if (grid_.nx < grid_.ng || grid_.ny < grid_.ng)
        throw std::runtime_error("Local nx and ny must be >= ng for halo exchange and wall reflection");

    vtkDataOutputType_ = parseVTKDataOutputType(
        cfg.getString("output.vtkDataOutputType", "point"));

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
    grid_.nxTot = grid_.nx + 2 * grid_.ng;
    grid_.nyTot = grid_.ny + 2 * grid_.ng;
    U_.assign(static_cast<size_t>(grid_.nxTot) * static_cast<size_t>(grid_.nyTot), Vec4{});
    RHS_.assign(static_cast<size_t>(grid_.nxTot) * static_cast<size_t>(grid_.nyTot), Vec4{});
    faces_.resize(grid_.nx, grid_.ny);

    // -----------------------------
    // 6) Choose numerical methods
    // -----------------------------
    flux_ = makeFluxD<2>(cfg.getString("flux", "hllc"));
    ti_   = makeTimeIntegratorT<Vec4>(cfg.getString("timeIntegrator", "ssprk3"));

    // -----------------------------
    // 7) Initialise interior fields
    // -----------------------------
    // Only this-rank interior block is initialised here; ghosts are filled later.
  

    // Initialize only this-rank block using local physical extents.
    setFields2D(U_,
                grid_.nx, grid_.ny, grid_.ng,
                x0Loc, x1Loc,
                y0Loc, y1Loc,
                gamma_,
                cfg);
}

// Fill ghost cells for one solver state.
//
// Step 1: exchange MPI halos across subdomain interfaces.
// Step 2: apply physical boundary conditions only on ranks that touch the
// global domain boundary. Interior MPI interfaces are marked as Internal so
// boundary::apply2D(...) leaves them untouched.
void Solver::applyBC(std::vector<Vec4>& U) const {
    // 1) Exchange halos across MPI subdomain interfaces
    //    NOTE: This assumes Vec4 is a tightly-packed POD of 4 doubles.
    mp_.exchangeHalos2D(reinterpret_cast<double*>(U.data()), grid_.nx, grid_.ny, grid_.ng, 4);

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

    boundary::apply2D(U, grid_.nx, grid_.ny, grid_.ng, bcApply);
}

void Solver::validateInterior(const std::vector<Vec4>& U) const {
    for (int j = 0; j < grid_.ny; ++j) {
        for (int i = 0; i < grid_.nx; ++i) {
            const auto& Q = U[idx(grid_.ng + i, grid_.ng + j)];
            try {
                (void)evalFlowVars(Q, gamma_);
            } catch (const std::exception& e) {
                std::ostringstream msg;
                msg << "cell (" << grid_.iBeg+i << "," << grid_.jBeg+j
                    << "), U=[" << std::setprecision(17) << Q[0] << "," << Q[1]
                    << "," << Q[2] << "," << Q[3] << "]: " << e.what();
                throw std::runtime_error(msg.str());
            }
        }
    }
}

// Compute a stable time step from the 2D CFL estimate
//   dt = CFL / max( (|u|+a)/dx + (|v|+a)/dy )
// and reduce the spectral-radius bound across all MPI ranks so every rank uses
// the same accepted time step.
double Solver::computeDt(const std::vector<Vec4>& U) const {
    const double dx = grid_.dx();
    const double dy = grid_.dy();
    double maxS = 0.0;

    for (int j = 0; j < grid_.ny; ++j) {
        for (int i = 0; i < grid_.nx; ++i) {
            const auto W = evalFlowVars(U[idx(grid_.ng + i, grid_.ng + j)], gamma_);
            const double S = (std::abs(W.u) + W.a) / dx + (std::abs(W.v) + W.a) / dy;
            if (!std::isfinite(S) || S <= 0.0)
                throw std::runtime_error("CFL: invalid spectral radius at cell (" +
                    std::to_string(grid_.iBeg+i) + "," + std::to_string(grid_.jBeg+j) + ")");
            maxS = std::max(maxS, S);
        }
    }
    // Global maximum across ranks -> consistent dt for all ranks.
    const double maxSg = mp_.allreduceMax(maxS);
    const double dt = cfl_ / maxSg;
    if (!std::isfinite(dt) || dt <= 0.0) throw std::runtime_error("CFL: invalid time step");
    return dt;
}

// Build the semi-discrete finite-volume RHS: RHS = -dF/dx - dG/dy.
//
// Pipeline:
//   - characteristic reconstruction produces conservative UL/UR on x-faces and y-faces
//   - the flux module computes numerical fluxes by a Riemann solver
//   - flux differences produce the cell-centered RHS on the local interior block
void Solver::buildRHS(const std::vector<Vec4>& U, std::vector<Vec4>& RHS) {
    std::fill(RHS.begin(), RHS.end(), Vec4{});


    // Characteristic reconstruction returns conservative face states on x- and y-faces.
    // U must already have valid ghost cells when calling buildRHS.
    recon_.reconstructFacesX(U,
                             grid_.nx,
                             grid_.ny,
                             grid_.ng,
                             gamma_,
                             faces_.x.UL,
                             faces_.x.UR);
    recon_.reconstructFacesY(U,
                             grid_.nx,
                             grid_.ny,
                             grid_.ng,
                             gamma_,
                             faces_.y.UL,
                             faces_.y.UR);

    const double dx = grid_.dx();
    const double dy = grid_.dy();

    auto faceFlux = [&](const Vec4& left, const Vec4& right, int dir, int i, int j) {
        try {
            (void)evalFlowVars(left, gamma_);
            (void)evalFlowVars(right, gamma_);
            const auto F = flux_->numericalFlux(left, right, dir, gamma_);
            if (!isFiniteState(F)) throw std::runtime_error("non-finite numerical flux");
            return F;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(dir == 0 ? "x" : "y") +
                "-face (" + std::to_string(grid_.iBeg+i) + "," +
                std::to_string(grid_.jBeg+j) + "): " + e.what());
        }
    };

    // Compute the configured numerical flux directly on each face.
    for (int j = 0; j < grid_.ny; ++j) {
        for (int i = 0; i < grid_.nx + 1; ++i) {
            const int f = idxFaceX(i, j);
            faces_.x.F[f] = faceFlux(faces_.x.UL[f], faces_.x.UR[f], 0, i, j);
        }
    }
    for (int j = 0; j < grid_.ny + 1; ++j) {
        for (int i = 0; i < grid_.nx; ++i) {
            const int f = idxFaceY(i, j);
            faces_.y.F[f] = faceFlux(faces_.y.UL[f], faces_.y.UR[f], 1, i, j);
        }
    }

    // Divergence of fluxes -> RHS (cell-centered):
    //   RHS = -(FxR-FxL)/dx - (GyT-GyB)/dy
    for (int j = 0; j < grid_.ny; ++j) {
        for (int i = 0; i < grid_.nx; ++i) {
            const Vec4& FxL = faces_.x.F[idxFaceX(i,     j)];
            const Vec4& FxR = faces_.x.F[idxFaceX(i + 1, j)];
            const Vec4& GyB = faces_.y.F[idxFaceY(i, j)];
            const Vec4& GyT = faces_.y.F[idxFaceY(i, j + 1)];

            Vec4 R{};
            for (int k = 0; k < 4; ++k) {
                R[k] = -(FxR[k] - FxL[k]) / dx - (GyT[k] - GyB[k]) / dy;
            }
            if (!isFiniteState(R)) throw std::runtime_error("Non-finite flux divergence");
            RHS[idx(grid_.ng + i, grid_.ng + j)] = R;
        }
    }
}

bool Solver::shouldWriteStepOutput(const int step) const {
    return outputEvery_ > 0 && (step % outputEvery_ == 0);
}

// Write one scheduled merged step snapshot when this step hits the configured
// output cadence. All ranks contribute interior-cell data and rank 0 writes
// the merged legacy VTK file.
void Solver::writeOutput(int step, double t) const {
    if (!shouldWriteStepOutput(step)) return;

    ensureOutputDirectoryExists(mp_);
    const std::string fname = makeStepOutputPath(outPrefix_, step, t);
    writeMergedVtkFile2D(mp_, fname, "step", t,
                         U_, grid_.nx, grid_.ny, grid_.ng,
                         grid_.iBeg, grid_.jBeg,
                         grid_.nxGlobal, grid_.nyGlobal,
                         grid_.x0, grid_.x1, grid_.y0, grid_.y1, gamma_,
                         vtkDataOutputType_);
}

// Write the terminal merged snapshot independent of the regular output cadence.
void Solver::writeFinalOutput(const double t) const {
    if (!writeFinal_) {
        return;
    }

    ensureOutputDirectoryExists(mp_);
    const std::string fname = makeFinalOutputPath(outPrefix_, t);
    writeMergedVtkFile2D(mp_, fname, "final", t,
                         U_, grid_.nx, grid_.ny, grid_.ng,
                         grid_.iBeg, grid_.jBeg,
                         grid_.nxGlobal, grid_.nyGlobal,
                         grid_.x0, grid_.x1, grid_.y0, grid_.y1, gamma_,
                         vtkDataOutputType_);
}

// Refresh ghost cells and write any scheduled snapshot.
void Solver::processRegularOutputPhase(const int step, const double t) {
    applyBC(U_);
    writeOutput(step, t);
}

// Main time-marching loop.
// - Build an initial RHS and process the initial output phase.
// - Advance in time with the selected explicit integrator.
// - After each accepted step, process the regular output phase.
// - Optionally emit one terminal output phase at the final time.
//
// The time integrator calls rhsFun(...) multiple times per time step; rhsFun
// refreshes ghost cells before reconstruction and flux evaluation.
void Solver::run() {
    double t = 0.0;
    int step = 0;
    std::cout << "[2D][r" << mp_.rank() << "] Starting run, finalTime=" << finalTime_
              << ", global=" << grid_.nxGlobal << "x" << grid_.nyGlobal
              << ", local=" << grid_.nx << "x" << grid_.ny
              << ", begin=(" << grid_.iBeg << "," << grid_.jBeg << ")\n";

    validateInterior(U_);
    // Build the initial RHS before the initial output phase.
    applyBC(U_);
    buildRHS(U_, RHS_);
    processRegularOutputPhase(step, t);

    // RHS callback used by explicit multi-stage integrators.
    int stage = 0;
    auto rhsFun = [this, &stage](std::vector<Vec4>& Uin, std::vector<Vec4>& Rout){
        // For every stage, refresh ghost cells (MPI halos + physical BCs)
        // before characteristic reconstruction and flux evaluation.
        ++stage;
        validateInterior(Uin);
        applyBC(Uin);
        buildRHS(Uin, Rout);
    };

    // Advance until finalTime. The last accepted step is clipped so the solver
    // lands exactly on the requested terminal time.
    while (t < finalTime_) {
        double dt = computeDt(U_);
        if (t + dt > finalTime_) dt = finalTime_ - t;

        if (!std::isfinite(dt) || dt <= 0.0 || t + dt <= t)
            throw std::runtime_error("Time step does not advance the physical clock");
        stage = 0;
        try {
            ti_->step(U_, dt, rhsFun);
            validateInterior(U_);
        } catch (const std::exception& e) {
            throw std::runtime_error("step=" + std::to_string(step) +
                ", t=" + std::to_string(t) + ", RHS stage=" + std::to_string(stage) + ": " + e.what());
        }
        t += dt;
        ++step;

        processRegularOutputPhase(step, t);
    }

    // Optionally write the terminal snapshot.
    if (writeFinal_) {
        applyBC(U_);
        writeFinalOutput(t);
    }
}
