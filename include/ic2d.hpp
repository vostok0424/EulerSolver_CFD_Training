#pragma once

// ic2d.hpp
// Initial conditions (IC) for the 2D Euler solver.
//
// The solver constructs an IC object from the cfg file (see makeIC2D),
// then calls IC2D::apply(...) once to fill the conservative state U.
//
// Conventions:
// - U is cell-centered and includes ghost cells.
// - Only interior cells [ng..ng+nx-1] x [ng..ng+ny-1] must be initialized; ghost
//   cells are typically filled later by the boundary-condition module.
// - The state is conservative: (rho, rho*u, rho*v, rho*E).
// - x0/x1/y0/y1 are the physical extents of THIS rank's local subdomain (important in MPI).
//
// To add a new IC:
//   1) Create a new struct deriving from IC2D and implement apply().
//   2) Register it in makeIC2D(name) in ic2d.cpp.

#include "state.hpp"
#include "cfg.hpp"
#include <vector>

// IC2D
// ----
// Abstract interface for a 2D initial condition.
//
// apply(...): fill U with an initial conservative field based on cfg options.
//
// Parameters:
// - nx, ny: number of interior cells in x/y
// - ng: number of ghost cells
// - x0, x1, y0, y1: local physical extent (important in MPI runs)
// - gamma: ratio of specific heats
// - cfg: configuration key-value access
struct IC2D {
    virtual ~IC2D() = default;
    // Fill the solution vector U with the desired initial condition.
    // Implementations should at minimum initialize interior cells.
    virtual void apply(std::vector<Vec4>& U,
                       int nx, int ny, int ng,
                       double x0, double x1,
                       double y0, double y1,
                       double gamma,
                       const Cfg& cfg) const = 0;
};

// IC2D_SodX
// ---------
// 2D shock-tube initial condition with the discontinuity normal to the x-direction.
//
// The initial state is piecewise-constant in x (left/right), and uniform in y.
// Parameters are read from cfg (see ic2d.cpp), with sensible defaults.
struct IC2D_SodX final : public IC2D {
    void apply(std::vector<Vec4>& U,
               int nx, int ny, int ng,
               double x0, double x1,
               double y0, double y1,
               double gamma,
               const Cfg& cfg) const override;
};

// Factory: construct an IC object by name.
//
// Example cfg:
//   ic = sodx
//
// The returned pointer is owned by the caller.
IC2D* makeIC2D(const std::string& name);
