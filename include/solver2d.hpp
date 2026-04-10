#pragma once

// solver2d.hpp
// -------------
// 2D finite-volume Euler solver (cell-centered, explicit time stepping).
//
// High-level workflow (see solver2d.cpp):
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
//        - scan interior states via the centralized state layer (optional diagnostics)
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
#include "ic2d.hpp"
#include "setFields.hpp"
#include "io2d.hpp"
#include "boundary.hpp"

#include <memory>
#include <string>
#include <vector>

// Solver2D
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
class Solver2D {
public:
    // Construct the solver from a cfg object and an MPI helper (mp).
    // The solver stores a reference to mp; mp must outlive the solver.
    Solver2D(const Cfg& cfg, const mpi_parallel::MpiParallel& mp);

    // Run the full simulation: initialise, time-step, and write outputs.
    void run();

private:
    // MPI topology + this-rank subdomain.
    // Subdomain indices are global (0-based) for the interior region.
    const mpi_parallel::MpiParallel& mp_;
    mpi_parallel::Subdomain2D sub_{};

    // Global mesh sizes (from cfg). In MPI runs, these are the full-domain sizes.
    int nxGlobal_{}, nyGlobal_{};

    // Local mesh sizes for this rank (interior only) and totals including ghosts.
    int nx_{}, ny_{}, ng_{};
    int nxTot_{}, nyTot_{};

    // Global start indices of this rank's interior block (0-based).
    int iBeg_{}, jBeg_{};

    // Local physical coordinates for this rank's interior domain.
    double x0_{}, x1_{}, y0_{}, y1_{};
    double gamma_{};
    double cfl_{};
    double finalTime_{};

    int outputEvery_{};
    bool writeFinal_{};
    std::string outPrefix_;

    // Shared state-layer thresholds/diagnostic options used for interior-state checks.
    StateLimits stateLimits_{};
    bool enableStateDiagnostics_{true};
    std::string stateDiagCsvPath_;
    mutable bool stateDiagWriteHeader_{true};

    // Parsed 2D boundary-condition object (types + per-side parameters).
    boundary::Bc2D bc_;
    // Reconstruction module: builds face left/right states from cell values.
    recon::Reconstruction2D recon_;

    // Primary conservative solution (cell-centered) and RHS storage.
    std::vector<Vec4> U_;
    std::vector<Vec4> RHS_;

    // Reusable RHS work buffers to avoid repeated allocation inside buildRHS().
    std::vector<Vec4> ULxBuf_;
    std::vector<Vec4> URxBuf_;
    std::vector<Vec4> ULyBuf_;
    std::vector<Vec4> URyBuf_;
    std::vector<Vec4> FxBuf_;
    std::vector<Vec4> GyBuf_;

    // Pluggable numerical modules selected by cfg.
    std::unique_ptr<FluxD<2>> flux_;
    std::unique_ptr<TimeIntegratorT<Vec4>> ti_;
    std::unique_ptr<IC2D> ic_;

    // Flatten (i,j) into a 1D index for ghosted arrays (0..nxTot_-1, 0..nyTot_-1).
    int idx(int i, int j) const { return i + nxTot_ * j; }

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

    // Scan interior cell states over the local physical block using the centralized state layer.
    StateScanReport scanInteriorStates(const std::vector<Vec4>& U) const;

    // Return true when this step should write regular field output.
    bool shouldWriteStepOutput(int step) const;
    // Return true when state diagnostics should be recorded (bound to output steps).
    bool shouldRecordStateDiagnostics(int step) const;
    // Append one compact state-diagnostic record to the CSV file on the root rank.
    void appendStateDiagnosticsCsv(int step, double t, const StateScanReport& report, const std::string& tag) const;

    // Write legacy VTK (.vtk) output for the current step/time.
    // In MPI runs, this may gather to rank 0 and write a single merged file.
    void writeOutput(int step, double t) const;
};
