#pragma once

// solver.hpp
// -------------
// 2D finite-volume Euler solver (cell-centered, explicit time stepping).
//
// High-level workflow (see solver.cpp):
//   1) Read cfg and set up mesh size, gas constants, and numerical options.
//   2) Create modules via factories:
//        - IC (initial condition)
//        - setFields (optional region overrides)
//        - characteristic reconstruction for face states
//        - flux (Rusanov/HLLC/AUSM/...) for face fluxes
//        - time integrator (Euler/RK2/RK4/...) for time advancement
//        - boundary conditions for ghost cells
//   3) Time loop:
//        - applyBC(U)
//        - buildRHS(U, RHS)
//        - dt = computeDt(U)
//        - time_integrator.step(U, dt, rhsFun)
//        - scan interior states via the diagnostics layer (optional diagnostics)
//        - writeOutput(step, t)
//
// MPI:
// - The 2D solver uses a Cartesian px-by-py rank layout (mpi.px, mpi.py).
// - Each rank stores its local interior cells plus ng ghost cells on each side.
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
// Compact description of the global 2D Cartesian mesh and this rank's local
// owned block, including ghost-layer thickness and physical extents.
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
// - UL / UR: reconstructed left/right face states
// - F      : numerical flux at each face
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
// --------
// Owns the 2D solution arrays and orchestrates the simulation.
//
// Notes on indexing:
// - U_ is a flattened 2D array of size nxTot_ * nyTot_.
// - Interior cells i=0..nx_-1, j=0..ny_-1 are stored at:
//     I = idx(ng_ + i, ng_ + j)
// - Ghost layers occupy the outer ng_ cells on each side.
//
// idx(i,j) uses row-major storage: i changes fastest.
class Solver {
public:
    // Construct the solver from a cfg object and an MPI helper (mp).
    // The solver stores a reference to mp; mp must outlive the solver.
    Solver(const Cfg& cfg, const mpi_parallel::MpiParallel& mp);

    // Run the full simulation: initialise, time-step, and write outputs.
    void run();

private:
    // MPI topology + this-rank subdomain.
    // Subdomain indices are global (0-based) for the interior region.
    const mpi_parallel::MpiParallel& mp_;
    mpi_parallel::Subdomain2D sub_{};

    // Global mesh and this-rank local block metadata.
    GridBlock2DInfo grid_{};

    // Primary conservative solution (cell-centered) and RHS storage.
    std::vector<Vec4> U_;
    std::vector<Vec4> RHS_;

    // Reusable face-centered reconstruction/flux buffers.
    FaceBuffers2D faces_{};

    // Parsed 2D boundary-condition object (types + per-side parameters).
    boundary::Bc2D bc_;
    // Reconstruction module: builds face left/right states from cell values.
    recon::Reconstruction2D recon_;

    // Pluggable numerical modules selected by cfg.
    std::unique_ptr<FluxD<2>> flux_;
    std::unique_ptr<TimeIntegratorT<Vec4>> ti_;
    std::unique_ptr<IC> ic_;

    // Run-control parameters.
    double gamma_{};
    double cfl_{};
    double finalTime_{};
    int outputEvery_{};
    bool writeFinal_{};
    std::string outPrefix_{};

    // Shared state-layer thresholds/diagnostic options used for interior-state checks.
    StateLimits stateLimits_{};
    bool enableStateDiagnostics_{true};
    std::string stateDiagCsvPath_;

    // Flatten (i,j) into a 1D index for ghosted cell-centered arrays.
    int idx(int i, int j) const { return i + grid_.nxTot * j; }

    // Flatten local x-face and y-face indices into 1D storage.
    int idxFaceX(int i, int j) const { return i + (grid_.nx + 1) * j; }
    int idxFaceY(int i, int j) const { return i + grid_.nx * j; }

    // Fill ghost cells using MPI halo exchange on internal interfaces and
    // physical boundary conditions on domain edges.
    // Must be called before reconstruction/flux evaluation.
    void applyBC(std::vector<Vec4>& U) const;
    // Compute a stable explicit time step (CFL condition) from the current state.
    double computeDt(const std::vector<Vec4>& U) const;
    // Build spatial RHS for the given state vector.
    // Expects ghost cells already valid (call applyBC first).
    // Uses: reconstruction -> numerical flux -> finite-volume divergence.
    void buildRHS(const std::vector<Vec4>& U, std::vector<Vec4>& RHS);


    // Return true when this step should write regular field output.
    bool shouldWriteStepOutput(int step) const;
    // Return true when state diagnostics should be recorded (bound to output steps).
    bool shouldRecordStateDiagnostics(int step) const;
    // Scan, MPI-reduce, and append one state-diagnostics record when enabled.
    void recordStateDiagnostics(int step, double t, const std::string& tag) const;

    // Write legacy VTK (.vtk) output for the current step/time.
    // In MPI runs, this may gather to rank 0 and write a single merged file.
    void writeOutput(int step, double t) const;
};
