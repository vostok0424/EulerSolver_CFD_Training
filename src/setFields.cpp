// setFields.cpp
// -------------
// Optional initial-field overrides (similar to OpenFOAM's setFields concept).
//
// Purpose:
// - Start from a uniform background primitive state (rho,u[,v],p).
// - Then apply a list of rectangular regions (in 1D: x-interval; in 2D: x-y box)
//   that overwrite the initial state.
//
// Notes:
// - This module writes ONLY interior cells. Ghost cells are filled later by boundary conditions.
// - Region indices in cfg start from 1: setFields.region1.*, region2.*, ...
// - If multiple regions overlap, later regions overwrite earlier ones (r increasing).

#include "setFields.hpp"

void setFields1D(std::vector<Vec3>& U,
                 int nx, int ng,
                 double x0, double x1,
                 double gamma,
                 const Cfg& cfg)
{
    // Uniform grid spacing over the interior domain.
    const double dx = (x1 - x0) / static_cast<double>(nx);

    // Background primitive state (defaults are a simple uniform flow).
    Prim1 bg{};
    bg.rho = cfg.getDouble("setFields.bg.rho", 1.0);
    bg.u[0]= cfg.getDouble("setFields.bg.u",   0.0);
    bg.p   = cfg.getDouble("setFields.bg.p",   1.0);

    // Convert background primitive state to conservative form stored in U.
    const Vec3 Ubg = EosIdealGas<1>::primToCons(bg, gamma);

    // Fill interior cells [ng .. ng+nx-1] with the background state.
    for (int i=0;i<nx;++i) U[ng+i] = Ubg;

    // Apply region overrides (piecewise-constant patches).
    const int nRegions = cfg.getInt("setFields.nRegions", 0);
    for (int r=1;r<=nRegions;++r) {
        // Each region r has keys like:
        //   setFields.region<r>.xMin, xMax, rho, u, p
        const std::string base = "setFields.region" + std::to_string(r) + ".";
        const double xMin = cfg.getDouble(base+"xMin", x0);
        const double xMax = cfg.getDouble(base+"xMax", x1);

        // Region state: start from background and override provided keys.
        Prim1 W = bg;
        W.rho = cfg.getDouble(base+"rho", bg.rho);
        W.u[0]= cfg.getDouble(base+"u",   bg.u[0]);
        W.p   = cfg.getDouble(base+"p",   bg.p);
        const Vec3 Ureg = EosIdealGas<1>::primToCons(W, gamma);

        // Overwrite cells whose centers fall inside [xMin, xMax].
        for (int i=0;i<nx;++i) {
            // Cell-centered coordinate.
            const double x = x0 + (i+0.5)*dx;
            if (x>=xMin && x<=xMax) U[ng+i] = Ureg;
        }
    }
}

void setFields2D(std::vector<Vec4>& U,
                 int nx, int ny, int ng,
                 double x0, double x1,
                 double y0, double y1,
                 double gamma,
                 const Cfg& cfg)
{
    // Ghosted array indexing helper (row-major): idx(i,j) = i + (nx+2*ng)*j.
    const int nxTot = nx + 2*ng;
    auto idx = [&](int i, int j){ return i + nxTot*j; };

    // Uniform spacing over the interior domain.
    const double dx = (x1 - x0) / static_cast<double>(nx);
    const double dy = (y1 - y0) / static_cast<double>(ny);

    // Background primitive state (rho,u,v,p) for 2D.
    Prim2 bg{};
    bg.rho = cfg.getDouble("setFields.bg.rho", 1.0);
    bg.u[0]= cfg.getDouble("setFields.bg.u",   0.0);
    bg.u[1]= cfg.getDouble("setFields.bg.v",   0.0);
    bg.p   = cfg.getDouble("setFields.bg.p",   1.0);

    // Convert background primitive state to conservative form stored in U.
    const Vec4 Ubg = EosIdealGas<2>::primToCons(bg, gamma);

    // Fill interior cells only.
    for (int j=0;j<ny;++j)
        for (int i=0;i<nx;++i)
            U[idx(ng+i, ng+j)] = Ubg;

    // Apply rectangular region overrides.
    const int nRegions = cfg.getInt("setFields.nRegions", 0);
    for (int r=1;r<=nRegions;++r) {
        // Each region r has keys like:
        //   setFields.region<r>.xMin, xMax, yMin, yMax, rho, u, v, p
        const std::string base = "setFields.region" + std::to_string(r) + ".";
        const double xMin = cfg.getDouble(base+"xMin", x0);
        const double xMax = cfg.getDouble(base+"xMax", x1);
        const double yMin = cfg.getDouble(base+"yMin", y0);
        const double yMax = cfg.getDouble(base+"yMax", y1);

        // Region state: start from background and override provided keys.
        Prim2 W = bg;
        W.rho = cfg.getDouble(base+"rho", bg.rho);
        W.u[0]= cfg.getDouble(base+"u",   bg.u[0]);
        W.u[1]= cfg.getDouble(base+"v",   bg.u[1]);
        W.p   = cfg.getDouble(base+"p",   bg.p);
        const Vec4 Ureg = EosIdealGas<2>::primToCons(W, gamma);

        // Overwrite cells whose centers fall inside the box [xMin,xMax] x [yMin,yMax].
        for (int j=0;j<ny;++j) {
            // Cell-centered y coordinate.
            const double y = y0 + (j+0.5)*dy;
            if (y<yMin || y>yMax) continue;
            for (int i=0;i<nx;++i) {
                // Cell-centered x coordinate.
                const double x = x0 + (i+0.5)*dx;
                if (x<xMin || x>xMax) continue;
                U[idx(ng+i, ng+j)] = Ureg;
            }
        }
    }
}
