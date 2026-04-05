#include "ic2d.hpp"
#include <stdexcept>

// ic2d.cpp
// --------
// Implementations of 2D initial conditions.
//
// The solver allocates U with ghost cells and calls IC2D::apply(...) once to
// initialise the interior cells. Boundary conditions will fill ghost cells later.
//
// This file currently provides:
//   - IC2D_SodX: 2D shock-tube with a discontinuity normal to the x-direction
//   - makeIC2D:  factory to construct an IC object by name

void IC2D_SodX::apply(std::vector<Vec4>& U,
                      int nx, int ny, int ng,
                      double x0, double x1,
                      double y0, double y1,
                      double gamma,
                      const Cfg& cfg) const
{
    // Total sizes including ghost cells.
    // U is stored as a flattened 2D array with row-major order.
    const int nxTot = nx + 2*ng;
    // Helper to flatten (i,j) indices into a 1D vector index.
    // i = 0..nxTot-1, j = 0..nyTot-1 over the ghosted array.
    auto idx = [&](int i, int j){ return i + nxTot*j; };

    // Interior grid spacing. Cell centers are at:
    //   x = x0 + (i+0.5)*dx,  y = y0 + (j+0.5)*dy
    const double dx = (x1 - x0) / nx;
    const double dy = (y1 - y0) / ny;
    // This IC varies only in x (uniform in y). dy is computed for completeness.
    (void)dy;
    // Discontinuity location. Default is the mid-point of the local domain.
    const double xMid = cfg.getDouble("ic2d.sodx.xMid", 0.5*(x0+x1));

    // Left/right primitive states (rho, u, v, p) read from cfg.
    // Defaults correspond to the standard Sod problem extended to 2D.
    Prim2 WL{}, WR{};
    WL.rho = cfg.getDouble("ic2d.sodx.rhoL", 1.0);
    WL.u[0]= cfg.getDouble("ic2d.sodx.uL",   0.0);
    WL.u[1]= cfg.getDouble("ic2d.sodx.vL",   0.0);
    WL.p   = cfg.getDouble("ic2d.sodx.pL",   1.0);

    WR.rho = cfg.getDouble("ic2d.sodx.rhoR", 0.125);
    WR.u[0]= cfg.getDouble("ic2d.sodx.uR",   0.0);
    WR.u[1]= cfg.getDouble("ic2d.sodx.vR",   0.0);
    WR.p   = cfg.getDouble("ic2d.sodx.pR",   0.1);

    // Convert primitive states to conservative vectors stored in U.
    const Vec4 UL = EosIdealGas<2>::primToCons(WL, gamma);
    const Vec4 UR = EosIdealGas<2>::primToCons(WR, gamma);

    // Initialise interior cells only:
    //   i = 0..nx-1, j = 0..ny-1 map to U[idx(ng+i, ng+j)].
    // Ghost cells are left untouched here and will be filled by BCs.
    for (int j=0;j<ny;++j) {
        // j is used in idx(...) below; no unused-variable suppression needed.
        for (int i=0;i<nx;++i) {
            // Cell-centered x coordinate (IC depends only on x).
            const double x = x0 + (i+0.5)*dx;
            // Piecewise-constant initial condition in x.
            U[idx(ng+i, ng+j)] = (x < xMid) ? UL : UR;
        }
    }
}

// Factory: create a 2D initial-condition object by name.
// The caller owns the returned pointer.
IC2D* makeIC2D(const std::string& name) {
    // Register new ICs here.
    if (name == "sodx") return new IC2D_SodX();
    throw std::runtime_error("Unknown 2D IC: " + name);
}
