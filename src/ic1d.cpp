#include "ic1d.hpp"
#include <stdexcept>

// ic1d.cpp
// --------
// Implementations of 1D initial conditions.
//
// The solver allocates U with ghost cells and calls IC1D::apply(...) once to
// initialize the interior cells. Boundary conditions will fill ghost cells later.
//
// This file currently provides:
//   - IC1D_Sod: classic Sod shock-tube Riemann problem
//   - makeIC1D: factory to construct an IC object by name

void IC1D_Sod::apply(std::vector<Vec3>& U,
                     int nx, int ng,
                     double x0, double x1,
                     double gamma,
                     const Cfg& cfg) const
{
    // Grid spacing for the interior domain on this rank.
    // Cell centers are located at x = x0 + (i+0.5)*dx.
    const double dx = (x1 - x0) / nx;

    // Discontinuity location. Default is the mid-point of the domain.
    const double xMid = cfg.getDouble("ic.sod.xMid", 0.5*(x0+x1));

    // Left/right primitive states (rho, u, p) read from cfg.
    // Defaults correspond to the standard Sod problem.
    Prim1 WL{}, WR{};
    WL.rho = cfg.getDouble("ic.sod.rhoL", 1.0);
    WL.u[0]= cfg.getDouble("ic.sod.uL",   0.0);
    WL.p   = cfg.getDouble("ic.sod.pL",   1.0);

    WR.rho = cfg.getDouble("ic.sod.rhoR", 0.125);
    WR.u[0]= cfg.getDouble("ic.sod.uR",   0.0);
    WR.p   = cfg.getDouble("ic.sod.pR",   0.1);

    // Convert primitive states to conservative vectors stored in U.
    const Vec3 UL = EosIdealGas<1>::primToCons(WL, gamma);
    const Vec3 UR = EosIdealGas<1>::primToCons(WR, gamma);

    // Initialise interior cells: choose UL or UR depending on the cell center.
    // Only interior indices [ng .. ng+nx-1] are written here.
    for (int i=0;i<nx;++i) {
        // Cell-centered x coordinate.
        const double x = x0 + (i+0.5)*dx;
        // Piecewise-constant initial condition.
        U[ng+i] = (x < xMid) ? UL : UR;
    }
}

// Factory: create a 1D initial-condition object by name.
// The caller owns the returned pointer.
IC1D* makeIC1D(const std::string& name) {
    // Register new ICs here.
    if (name == "sod") return new IC1D_Sod();
    throw std::runtime_error("Unknown 1D IC: " + name);
}
