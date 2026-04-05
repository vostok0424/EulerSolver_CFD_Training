#pragma once

// ic1d.hpp
// Initial conditions (IC) for the 1D Euler solver.
//
// The solver builds an IC object from the cfg file (see makeIC1D),
// then calls IC1D::apply(...) once to fill the conservative state U.
//
// Conventions:
// - U is cell-centered and includes ghost cells.
// - Only interior cells [ng .. ng+nx-1] must be initialized; ghost cells are
//   typically filled later by the boundary-condition module.
// - The state is conservative: (rho, rho*u, rho*E).
// - x0/x1 define the physical extent of THIS rank's local subdomain.
//
// This design makes it easy to add new ICs:
//   1) Create a new struct deriving from IC1D and implement apply().
//   2) Register it in makeIC1D(name) in ic1d.cpp.

#include "state.hpp"
#include "cfg.hpp"
#include <vector>

// IC1D
// ----
// Abstract interface for a 1D initial condition.
//
// apply(...): fill U with an initial conservative field based on cfg options.
//
// Parameters:
// - nx, ng: number of interior cells and ghost cells
// - x0, x1: local physical extent (important in MPI runs)
// - gamma: ratio of specific heats
// - cfg: configuration key-value access
struct IC1D {
    virtual ~IC1D() = default;

    // Fill the solution vector U with the desired initial condition.
    // Implementations should at minimum initialize interior cells.
    virtual void apply(std::vector<Vec3>& U,
                       int nx, int ng,
                       double x0, double x1,
                       double gamma,
                       const Cfg& cfg) const = 0;
};

// IC1D_Sod
// --------
// Classic Sod shock-tube initial condition (Riemann problem):
// left/right states separated by a discontinuity.
// Parameters are read from cfg (see ic1d.cpp), with sensible defaults.
struct IC1D_Sod final : public IC1D {
    void apply(std::vector<Vec3>& U,
               int nx, int ng,
               double x0, double x1,
               double gamma,
               const Cfg& cfg) const override;
};

// Factory: construct an IC object by name.
//
// Example cfg:
//   ic = sod
//
// The returned pointer is owned by the caller.
IC1D* makeIC1D(const std::string& name);
