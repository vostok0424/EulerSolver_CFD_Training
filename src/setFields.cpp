// setFields.cpp
// -------------
// Optional initial-field overrides (similar to OpenFOAM's setFields concept).
//
// Purpose:
// - Start from a uniform background primitive state (rho,u,v,p).
// - Then apply a list of rectangular regions (x-y boxes)
//
// Notes:
// - This module writes ONLY interior cells. Ghost cells are filled later by boundary conditions.
// - Region indices in cfg start from 1: setFields.region1.*, setFields.region2.*, ...
// - If multiple regions overlap, later regions overwrite earlier ones (r increasing).


#include "setFields.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

enum class ShockDir {
    PlusX,
    MinusX,
    PlusY,
    MinusY
};

ShockDir parseShockDir(const std::string& s)
{
    if (s == "+x" || s == "x" || s == "+X" || s == "X") return ShockDir::PlusX;
    if (s == "-x" || s == "-X") return ShockDir::MinusX;
    if (s == "+y" || s == "y" || s == "+Y" || s == "Y") return ShockDir::PlusY;
    if (s == "-y" || s == "-Y") return ShockDir::MinusY;
    throw std::runtime_error("setFields: invalid shockDir='" + s + "' (expected +x, -x, +y, or -y)");
}

Prim2 buildRegionPrim2D(const Prim2& bg, double gamma, const Cfg& cfg, const std::string& base)
{
    Prim2 W = bg;
    const double shockMach = cfg.getDouble(base + "shockMach", -1.0);
    if (shockMach > 0.0) {
        if (shockMach <= 1.0) {
            throw std::runtime_error("setFields: shockMach must be > 1 for " + base);
        }

        W.rho = cfg.getDouble(base + "rho", bg.rho);
        W.u[0] = cfg.getDouble(base + "u", 0.0);
        W.u[1] = cfg.getDouble(base + "v", 0.0);
        W.p = cfg.getDouble(base + "p", bg.p);
        const ShockDir dir = parseShockDir(cfg.getString(base + "shockDir", "+x"));

        double nx = 0.0;
        double ny = 0.0;
        switch (dir) {
            case ShockDir::PlusX:  nx =  1.0; ny =  0.0; break;
            case ShockDir::MinusX: nx = -1.0; ny =  0.0; break;
            case ShockDir::PlusY:  nx =  0.0; ny =  1.0; break;
            case ShockDir::MinusY: nx =  0.0; ny = -1.0; break;
        }

        const double rho1 = W.rho;
        const double u1 = W.u[0];
        const double v1 = W.u[1];
        const double p1 = W.p;
        const double a1 = std::sqrt(gamma * p1 / rho1);
        const double un1 = u1 * nx + v1 * ny;
        const double utx = u1 - un1 * nx;
        const double uty = v1 - un1 * ny;
        const double Vs = un1 + shockMach * a1;
        const double w1 = shockMach * a1;
        const double rhoRatio = ((gamma + 1.0) * shockMach * shockMach)
                              / ((gamma - 1.0) * shockMach * shockMach + 2.0);
        const double pRatio = 1.0 + (2.0 * gamma / (gamma + 1.0))
                                    * (shockMach * shockMach - 1.0);
        const double rho2 = rho1 * rhoRatio;
        const double p2 = p1 * pRatio;
        const double w2 = w1 * (rho1 / rho2);
        const double un2 = Vs - w2;

        W.rho = rho2;
        W.u[0] = utx + un2 * nx;
        W.u[1] = uty + un2 * ny;
        W.p = p2;
        return W;
    }

    W.rho = cfg.getDouble(base + "rho", bg.rho);
    W.u[0] = cfg.getDouble(base + "u", bg.u[0]);
    W.u[1] = cfg.getDouble(base + "v", bg.u[1]);
    W.p = cfg.getDouble(base + "p", bg.p);
    return W;
}

} // namespace

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

        // Region state: either direct primitive override or shock-defined post-shock state.
        const Prim2 W = buildRegionPrim2D(bg, gamma, cfg, base);
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
