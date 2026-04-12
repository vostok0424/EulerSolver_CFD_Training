#pragma once

// solver1d.hpp
// -------------
// 1D finite-volume Euler solver (cell-centered, explicit time stepping).
//
// Workflow (see solver1d.cpp):
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
// - The 1D solver splits the domain along x (requires mpi.py = 1).
// - Each rank stores its local interior cells plus ng ghost cells on both sides.
// - In serial runs, the single rank owns the whole domain.

// Notes on indexing:
// - U_ is sized grid_.nxTot = grid_.nx + 2*grid_.ng.
// - Interior cell i (0..grid_.nx-1) is stored at U_[grid_.ng + i].
// - Ghost cells occupy [0..grid_.ng-1] (left) and
//   [grid_.ng + grid_.nx .. grid_.nxTot-1] (right).

#include "state.hpp"
#include "cfg.hpp"
#include "flux.hpp"
#include "time_integrator.hpp"
#include "mpi_parallel.hpp"
#include "reconstruction.hpp"
#include "ic1d.hpp"
#include "setFields.hpp"
#include "io1d.hpp"
#include "boundary.hpp"

#include <memory>
#include <string>
#include <vector>

// GridBlock1DInfo
// ---------------
// Compact description of the global 1D Cartesian mesh and this rank's local
// owned block, including ghost-layer thickness and physical extents.
struct GridBlock1DInfo {
    int nxGlobal{};

    int iBeg{};
    int nx{};

    int ng{};
    int nxTot{};

    double x0{};
    double x1{};

    double dx() const { return (x1 - x0) / static_cast<double>(nxGlobal); }
};

// FaceBuffers1D
// -------------
// Reusable face-centered storage:
// - UL / UR: reconstructed left/right face states
// - F      : numerical flux at each face
struct FaceBuffers1D {
    std::vector<Vec3> UL;
    std::vector<Vec3> UR;
    std::vector<Vec3> F;

    void resize(int nx) {
        const std::size_t nFaces = static_cast<std::size_t>(nx + 1);
        UL.resize(nFaces);
        UR.resize(nFaces);
        F.resize(nFaces);
    }
};

// Solver1D
// --------
// Owns the 1D solution arrays and orchestrates the simulation.
//
// Notes on indexing:
// - U_ is sized grid_.nxTot = grid_.nx + 2*grid_.ng.
// - Interior cell i (0..grid_.nx-1) is stored at U_[grid_.ng + i].
// - Ghost cells occupy [0..grid_.ng-1] (left) and
//   [grid_.ng + grid_.nx .. grid_.nxTot-1] (right).
class Solver1D {
public:
    // Construct the solver from a cfg object and an MPI helper (mp).
    // The solver stores a reference to mp; mp must outlive the solver.
    Solver1D(const Cfg& cfg, const mpi_parallel::MpiParallel& mp);

    // Run the full simulation: initialise, time-step, and write outputs.
    void run();

private:
    // MPI topology + this-rank subdomain.
    // Subdomain indices are global (0-based) for the interior region.
    const mpi_parallel::MpiParallel& mp_;
    mpi_parallel::Subdomain1D sub_{};

    // Global mesh and this-rank local block metadata.
    GridBlock1DInfo grid_{};

    // Primary conservative solution (cell-centered) and RHS storage.
    std::vector<Vec3> U_;
    std::vector<Vec3> RHS_;

    // Reusable face-centered reconstruction/flux buffers.
    FaceBuffers1D faces_{};

    // Parsed 1D boundary-condition object (types + per-side parameters).
    boundary::Bc1D bc_;

    // Reconstruction module: builds face left/right states from cell values.
    recon::Reconstruction1D recon_;

    // Pluggable numerical modules selected by cfg.
    std::unique_ptr<FluxD<1>> flux_;
    std::unique_ptr<TimeIntegratorT<Vec3>> ti_;
    std::unique_ptr<IC1D> ic_;

    // Run-control parameters.
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

    // Flatten local cell and face indices into 1D storage.
    int idx(int i) const { return i; }
    int idxFace(int i) const { return i; }

    // Fill ghost cells using MPI halo exchange on internal interfaces and
    // physical boundary conditions on domain edges.
    // Must be called before reconstruction/flux evaluation.
    void applyBC(std::vector<Vec3>& U) const;

    // Compute a stable explicit time step (CFL condition) from the current state.
    double computeDt(const std::vector<Vec3>& U) const;

    // Build spatial RHS for the given state vector.
    // Expects ghost cells already valid (call applyBC first).
    // Uses: reconstruction -> numerical flux -> finite-volume divergence.
    void buildRHS(const std::vector<Vec3>& U, std::vector<Vec3>& RHS);

    // Scan interior cell states [ng_, ng_ + nx_) using the centralized state layer.
    StateScanReport scanInteriorStates(const std::vector<Vec3>& U) const;

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
