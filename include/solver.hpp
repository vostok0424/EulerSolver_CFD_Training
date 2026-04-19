#pragma once

// solver.hpp
// -------------
// 2D finite-volume Euler solver (cell-centered, explicit time stepping).
//
// High-level workflow (see solver.cpp):
//   1) Read cfg, mesh, run-control, and boundary-condition settings.
//   2) Create the numerical modules:
//        - IC (initial condition) or setFields overrides
//        - characteristic reconstruction for face states
//        - numerical flux evaluation on x/y faces
//        - explicit time integrator for time advancement
//   3) Build an initial RHS and process the initial diagnostics/output phase.
//   4) Advance in time with repeated:
//        - CFL-based dt computation
//        - explicit time-integrator stage calls through rhsFun
//        - regular diagnostics/output processing on the configured cadence
//   5) Optionally emit one final diagnostics/output phase at the terminal time.
//
// MPI:
// - The 2D solver uses a Cartesian px-by-py rank layout (mpi.px, mpi.py).
// - Each rank stores its local interior cells plus ng ghost cells on each side.
// - MPI halo exchange updates interior subdomain interfaces; physical BCs are
//   applied only on ranks that touch the global domain boundary.
// - In serial runs, the single rank owns the whole domain.

#include "state.hpp"
#include "cfg.hpp"
#include "flux.hpp"
#include "time_integrator.hpp"
#include "mpi_parallel.hpp"
#include "reconstruction.hpp"
#include "ic.hpp"
#include "setFields.hpp"
#include "io.hpp"
#include "boundary.hpp"
#include "diagnostics.hpp"

#include <memory>
#include <string>
#include <vector>

// GridBlock2DInfo
// ---------------
// Compact description of the global 2D Cartesian mesh and this rank's owned
// local block, including ghost-layer thickness, logical sizes, and physical
// extents.
struct GridBlock2DInfo {
    int nxGlobal{};
    int nyGlobal{};

    int iBeg{};
    int jBeg{};
    int nx{};
    int ny{};

    int ng{};
    int nxTot{};
    int nyTot{};

    double x0{};
    double x1{};
    double y0{};
    double y1{};

    double dx() const { return (x1 - x0) / static_cast<double>(nxGlobal); }
    double dy() const { return (y1 - y0) / static_cast<double>(nyGlobal); }
};

// DirectionalFaceBuffers2D
// ------------------------
// Reusable face-centered storage for one coordinate direction:
// - UL / UR: reconstructed left/right conservative face states
// - F      : numerical flux stored at each face in that direction
struct DirectionalFaceBuffers2D {
    std::vector<Vec4> UL;
    std::vector<Vec4> UR;
    std::vector<Vec4> F;

    void resize(std::size_t nFaces) {
        UL.resize(nFaces);
        UR.resize(nFaces);
        F.resize(nFaces);
    }
};

// FaceBuffers2D
// -------------
// Direction-grouped face buffers for x-normal and y-normal faces.
struct FaceBuffers2D {
    DirectionalFaceBuffers2D x;
    DirectionalFaceBuffers2D y;

    void resize(int nx, int ny) {
        x.resize(static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny));
        y.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1));
    }
};

// Solver
// ------
// Owns the 2D solution/RHS arrays, reusable face buffers, and the numerical
// modules needed to advance the local MPI block.
//
// Notes on indexing:
// - U_ is a flattened ghosted 2D array of size grid_.nxTot * grid_.nyTot.
// - Interior cells i=0..grid_.nx-1, j=0..grid_.ny-1 are stored at
//     idx(grid_.ng + i, grid_.ng + j)
// - Ghost layers occupy the outer grid_.ng cells on each side.
// - idx(i,j) uses row-major storage: i changes fastest.
class Solver {
public:
    // Construct the solver from the case configuration and MPI helper.
    // The solver stores a reference to mp, so mp must outlive the solver.
    Solver(const Cfg& cfg, const mpi_parallel::MpiParallel& mp);

    // Run the full simulation: initial RHS build, time stepping, diagnostics,
    // and scheduled/final output.
    void run();

private:
    // MPI topology helper plus this-rank subdomain metadata.
    // Subdomain indices are global (0-based) for the interior region.
    const mpi_parallel::MpiParallel& mp_;
    mpi_parallel::Subdomain2D sub_{};

    // Global mesh and this-rank local block metadata, including ghost sizes.
    GridBlock2DInfo grid_{};

    // Primary conservative solution (cell-centered, ghosted) and RHS storage.
    std::vector<Vec4> U_;
    std::vector<Vec4> RHS_;

    // Reusable face-centered buffers for reconstructed states and fluxes.
    FaceBuffers2D faces_{};

    // Parsed 2D boundary-condition object (types + per-side parameters).
    boundary::Bc2D bc_;
    // Reconstruction module: builds left/right face states from cell values.
    recon::Reconstruction2D recon_;

    // Pluggable numerical modules selected from cfg.
    std::unique_ptr<FluxD<2>> flux_;
    std::unique_ptr<TimeIntegratorT<Vec4>> ti_;
    std::unique_ptr<IC> ic_;

    // Gas model and run-control parameters.
    double gamma_{};
    double cfl_{};
    double finalTime_{};
    int outputEvery_{};
    bool writeFinal_{};
    std::string outPrefix_{};

    // Shared state-layer thresholds and diagnostics controls used for interior
    // state checks and reporting.
    StateLimits stateLimits_{};
    bool enableStateDiagnostics_{true};
    std::string stateDiagCsvPath_;

    // Flatten (i,j) into a 1D row-major index for ghosted cell-centered arrays.
    int idx(int i, int j) const { return i + grid_.nxTot * j; }

    // Flatten local x-face and y-face indices into 1D row-major storage.
    int idxFaceX(int i, int j) const { return i + (grid_.nx + 1) * j; }
    int idxFaceY(int i, int j) const { return i + grid_.nx * j; }

    // Fill ghost cells by MPI halo exchange on internal interfaces and
    // physical boundary conditions on global domain edges.
    // Must be called before reconstruction/flux evaluation.
    void applyBC(std::vector<Vec4>& U) const;
    // Compute a stable explicit CFL time step from the current interior state.
    double computeDt(const std::vector<Vec4>& U) const;
    // Build the spatial finite-volume RHS for the given state vector.
    // Expects ghost cells already valid (call applyBC first).
    // Uses: reconstruction -> numerical flux -> flux divergence.
    void buildRHS(const std::vector<Vec4>& U, std::vector<Vec4>& RHS);

    // Diagnostics/output orchestration helpers.
    // These keep run() focused on the main advance loop while centralizing
    // cadence checks, diagnostics recording, and merged snapshot writing.
    bool shouldWriteStepOutput(int step) const;
    bool shouldRecordStateDiagnostics(int step) const;
    void recordStateDiagnostics(int step, double t, const std::string& tag) const;
    void processRegularOutputPhase(int step, double t, const std::string& diagnosticsTag);
    void writeOutput(int step, double t) const;
    void writeFinalOutput(double t) const;
};
