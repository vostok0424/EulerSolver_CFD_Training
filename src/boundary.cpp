
#include "boundary.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

// boundary.cpp
// -----------
// Boundary-condition (BC) application for 2D Euler on a Cartesian grid.
//
// This module fills ghost cells based on boundary types selected in the cfg file.
// The solver is expected to:
//   1) Read BC data once via read2D(cfg)
//   2) Perform MPI halo exchange (for internal interfaces)
//   3) Call apply2D(...) to fill physical boundary ghost cells
//
// Supported BC names (case-sensitive; no aliases):
//   - "internal"      : MPI subdomain interface (ghost cells come from halo exchange)
//   - "zeroGradient"  : copy nearest interior cell into ghosts
//   - "inlet"         : fixed primitive state (rho, u, v, p) from cfg
//   - "outlet"        : subsonic pressure outlet (impose p, extrapolate rho and velocity)
//   - "slipWall"      : inviscid slip wall (reflect normal momentum)
//   - "symmetry"      : same treatment as slipWall here (reflect normal momentum)
//
// Notes:
// - This code is intentionally strict (training code): BC names must match exactly.
// - Periodic/cyclic boundaries are explicitly rejected.
// - All inlet/outlet parameters are now stored in explicit BC objects rather than
//   file-scope static state.

namespace boundary {

// ---- strict name handling ----
// We only trim whitespace. We do NOT fold case and we do NOT accept aliases.
// This keeps cfg usage explicit and avoids silent behavior changes.

static inline std::string trim(std::string s) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

// Normalize a boundary name:
// - trim whitespace
// - if empty, default to "zeroGradient"
static inline std::string normalizeName(const std::string& s) {
    const std::string v = trim(s);
    if (v.empty()) return "zeroGradient";
    return v;
}

static inline void rejectIfPeriodic(const std::string& n) {
    const std::string v = normalizeName(n);
    if (v == "periodic" || v == "cyclic") {
        throw std::runtime_error(
            "boundary: periodic/cyclic is not supported in this training code.");
    }
}

static inline BcType parseBcType(const std::string& n, const std::string& where) {
    const std::string v = normalizeName(n);
    rejectIfPeriodic(v);

    if (v == "inlet") return BcType::Inlet;
    if (v == "outlet") return BcType::Outlet;
    if (v == "zeroGradient") return BcType::ZeroGradient;
    if (v == "slipWall") return BcType::SlipWall;
    if (v == "symmetry") return BcType::Symmetry;
    if (v == "internal") return BcType::Internal;

    throw std::runtime_error("boundary: unsupported BC name in " + where + ": " + n);
}

static inline bool isWallLike(BcType t) {
    return (t == BcType::SlipWall || t == BcType::Symmetry);
}

static inline std::string pickName(const Cfg& cfg, const std::string& key,
                                   const std::string& fallback) {
    if (cfg.has(key)) return normalizeName(cfg.getString(key, ""));
    return normalizeName(fallback);
}

static inline void requireKey(const Cfg& cfg, const std::string& key) {
    if (!cfg.has(key)) {
        throw std::runtime_error("boundary: missing required cfg key: " + key);
    }
}

static inline double readDoubleStrict(const Cfg& cfg, const std::string& key) {
    requireKey(cfg, key);
    return cfg.getDouble(key, 0.0);
}

static inline bool hasAnyPrefix(const Cfg& cfg, const std::string& prefix,
                                const std::initializer_list<std::string>& keys) {
    for (const auto& k : keys) {
        if (cfg.has(prefix + k)) return true;
    }
    return false;
}


static Prim2 readInletPrim2(const Cfg& cfg, const std::string& side) {
    const std::string p1 = "bc." + side + ".inlet.";
    const std::string p0 = "bc.inlet.";

    const bool sideHas = hasAnyPrefix(cfg, p1, {"rho", "u", "v", "p"});
    const std::string p = sideHas ? p1 : p0;

    Prim2 W{};
    W.rho = readDoubleStrict(cfg, p + "rho");
    W.u[0] = readDoubleStrict(cfg, p + "u");
    W.u[1] = readDoubleStrict(cfg, p + "v");
    W.p = readDoubleStrict(cfg, p + "p");
    return W;
}

static double readOutletP(const Cfg& cfg, const std::string& side) {
    const std::string k1 = "bc." + side + ".outlet.p";
    const std::string k0 = "bc.outlet.p";
    if (cfg.has(k1)) return readDoubleStrict(cfg, k1);
    return readDoubleStrict(cfg, k0);
}


Bc2D read2D(const Cfg& cfg) {
    const std::string base = normalizeName(cfg.getString("bc", "zeroGradient"));

    Bc2D b;
    b.gamma = cfg.getDouble("gamma", 1.4);

    const std::string leftName = pickName(cfg, "bc.left", base);
    const std::string rightName = pickName(cfg, "bc.right", base);
    const std::string bottomName = pickName(cfg, "bc.bottom", base);
    const std::string topName = pickName(cfg, "bc.top", base);

    b.left.type = parseBcType(leftName, "read2D.left");
    b.right.type = parseBcType(rightName, "read2D.right");
    b.bottom.type = parseBcType(bottomName, "read2D.bottom");
    b.top.type = parseBcType(topName, "read2D.top");

    if (b.left.type == BcType::Inlet) {
        b.left.inlet = readInletPrim2(cfg, "left");
    } else if (b.left.type == BcType::Outlet) {
        b.left.pout = readOutletP(cfg, "left");
    }

    if (b.right.type == BcType::Inlet) {
        b.right.inlet = readInletPrim2(cfg, "right");
    } else if (b.right.type == BcType::Outlet) {
        b.right.pout = readOutletP(cfg, "right");
    }

    if (b.bottom.type == BcType::Inlet) {
        b.bottom.inlet = readInletPrim2(cfg, "bottom");
    } else if (b.bottom.type == BcType::Outlet) {
        b.bottom.pout = readOutletP(cfg, "bottom");
    }

    if (b.top.type == BcType::Inlet) {
        b.top.inlet = readInletPrim2(cfg, "top");
    } else if (b.top.type == BcType::Outlet) {
        b.top.pout = readOutletP(cfg, "top");
    }

    return b;
}

// ---- outlet state helper ----
// The outlet BC here is a simple pressure outlet:
// - If the boundary-normal Mach number is supersonic (M>=1): extrapolate everything.
// - If subsonic: impose pressure p_out and extrapolate density and velocity.


static inline Vec4 outletState2D(const Vec4& Uin, double pout, int normalDir,
                                 double gamma) {
    const Prim2 W = EosIdealGas<2>::consToPrim(Uin, gamma);
    const double a = EosIdealGas<2>::soundSpeed(W, gamma);
    const double un = W.u[normalDir];
    const double M = (a > 0.0) ? std::abs(un) / a : 0.0;

    if (M >= 1.0) {
        return Uin;
    }

    Prim2 Wo = W;
    Wo.p = pout;
    return EosIdealGas<2>::primToCons(Wo, gamma);
}


void apply2D(std::vector<Vec4>& U, int nx, int ny, int ng, const Bc2D& bc) {
    const double gamma = bc.gamma;
    const int nxTot = nx + 2 * ng;
    auto idx = [&](int i, int j) { return i + nxTot * j; };

    // LEFT / RIGHT (vary with j)
    for (int j = ng; j < ny + ng; ++j) {
        // LEFT
        if (bc.left.type == BcType::Internal) {
            // MPI subdomain interface: do nothing
        } else if (bc.left.type == BcType::ZeroGradient) {
            for (int g = 0; g < ng; ++g) U[idx(ng - 1 - g, j)] = U[idx(ng, j)];
        } else if (bc.left.type == BcType::Inlet) {
            const Vec4 Uin = EosIdealGas<2>::primToCons(bc.left.inlet, gamma);
            for (int g = 0; g < ng; ++g) U[idx(ng - 1 - g, j)] = Uin;
        } else if (bc.left.type == BcType::Outlet) {
            const Vec4 Uadj = U[idx(ng, j)];
            const Vec4 Uo = outletState2D(Uadj, bc.left.pout, /*normalDir=*/0, gamma);
            for (int g = 0; g < ng; ++g) U[idx(ng - 1 - g, j)] = Uo;
        } else if (isWallLike(bc.left.type)) {
            for (int g = 0; g < ng; ++g) {
                Vec4 G = U[idx(ng + g, j)];
                G[1] = -G[1];
                U[idx(ng - 1 - g, j)] = G;
            }
        } else {
            throw std::runtime_error("boundary: unsupported left BC type in apply2D.");
        }

        // RIGHT
        if (bc.right.type == BcType::Internal) {
            // MPI subdomain interface: do nothing
        } else if (bc.right.type == BcType::ZeroGradient) {
            for (int g = 0; g < ng; ++g) U[idx(nx + ng + g, j)] = U[idx(nx + ng - 1, j)];
        } else if (bc.right.type == BcType::Inlet) {
            const Vec4 Uin = EosIdealGas<2>::primToCons(bc.right.inlet, gamma);
            for (int g = 0; g < ng; ++g) U[idx(nx + ng + g, j)] = Uin;
        } else if (bc.right.type == BcType::Outlet) {
            const Vec4 Uadj = U[idx(nx + ng - 1, j)];
            const Vec4 Uo = outletState2D(Uadj, bc.right.pout, /*normalDir=*/0, gamma);
            for (int g = 0; g < ng; ++g) U[idx(nx + ng + g, j)] = Uo;
        } else if (isWallLike(bc.right.type)) {
            for (int g = 0; g < ng; ++g) {
                Vec4 G = U[idx(nx + ng - 1 - g, j)];
                G[1] = -G[1];
                U[idx(nx + ng + g, j)] = G;
            }
        } else {
            throw std::runtime_error("boundary: unsupported right BC type in apply2D.");
        }
    }

    // BOTTOM / TOP (vary with i)
    for (int i = 0; i < nxTot; ++i) {
        // BOTTOM
        if (bc.bottom.type == BcType::Internal) {
            // MPI subdomain interface: do nothing
        } else if (bc.bottom.type == BcType::ZeroGradient) {
            for (int g = 0; g < ng; ++g) U[idx(i, ng - 1 - g)] = U[idx(i, ng)];
        } else if (bc.bottom.type == BcType::Inlet) {
            const Vec4 Uin = EosIdealGas<2>::primToCons(bc.bottom.inlet, gamma);
            for (int g = 0; g < ng; ++g) U[idx(i, ng - 1 - g)] = Uin;
        } else if (bc.bottom.type == BcType::Outlet) {
            const Vec4 Uadj = U[idx(i, ng)];
            const Vec4 Uo = outletState2D(Uadj, bc.bottom.pout, /*normalDir=*/1, gamma);
            for (int g = 0; g < ng; ++g) U[idx(i, ng - 1 - g)] = Uo;
        } else if (isWallLike(bc.bottom.type)) {
            for (int g = 0; g < ng; ++g) {
                Vec4 G = U[idx(i, ng + g)];
                G[2] = -G[2];
                U[idx(i, ng - 1 - g)] = G;
            }
        } else {
            throw std::runtime_error("boundary: unsupported bottom BC type in apply2D.");
        }

        // TOP
        if (bc.top.type == BcType::Internal) {
            // MPI subdomain interface: do nothing
        } else if (bc.top.type == BcType::ZeroGradient) {
            for (int g = 0; g < ng; ++g) U[idx(i, ny + ng + g)] = U[idx(i, ny + ng - 1)];
        } else if (bc.top.type == BcType::Inlet) {
            const Vec4 Uin = EosIdealGas<2>::primToCons(bc.top.inlet, gamma);
            for (int g = 0; g < ng; ++g) U[idx(i, ny + ng + g)] = Uin;
        } else if (bc.top.type == BcType::Outlet) {
            const Vec4 Uadj = U[idx(i, ny + ng - 1)];
            const Vec4 Uo = outletState2D(Uadj, bc.top.pout, /*normalDir=*/1, gamma);
            for (int g = 0; g < ng; ++g) U[idx(i, ny + ng + g)] = Uo;
        } else if (isWallLike(bc.top.type)) {
            for (int g = 0; g < ng; ++g) {
                Vec4 G = U[idx(i, ny + ng - 1 - g)];
                G[2] = -G[2];
                U[idx(i, ny + ng + g)] = G;
            }
        } else {
            throw std::runtime_error("boundary: unsupported top BC type in apply2D.");
        }
    }
}

} // namespace boundary
