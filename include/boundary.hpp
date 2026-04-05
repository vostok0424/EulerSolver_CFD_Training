#pragma once
#include "state.hpp"
#include "cfg.hpp"
#include <vector>

/*
  boundary
  --------
  Common Euler boundary conditions for structured 1D/2D grids with ghost cells.

  Supported names:
    - inlet        : inflow boundary (impose primitive variables from cfg)
    - outlet       : outflow boundary (impose pressure for subsonic, extrapolate for supersonic)
    - zeroGradient : zero-gradient (copy nearest interior cell)
    - slipWall     : inviscid slip wall (reflect normal momentum)
    - symmetry     : symmetry plane (same treatment as slip for Euler)
    - internal     : no-op (leave ghost cells unchanged). Intended for MPI subdomain interfaces.

  Naming rules:
    - Names are case-sensitive.
    - No aliases or automatic canonicalization are applied (only whitespace is trimmed).

  Required cfg keys (only when the corresponding BC is selected; `internal` requires none):
    - gamma
        Used for primitive/conservative conversions and outlet Mach check.

    - inlet (1D):
        bc.inlet.rho, bc.inlet.u, bc.inlet.p
        or per-side overrides:
        bc.left.inlet.rho,  bc.left.inlet.u,  bc.left.inlet.p
        bc.right.inlet.rho, bc.right.inlet.u, bc.right.inlet.p

    - inlet (2D):
        bc.inlet.rho, bc.inlet.u, bc.inlet.v, bc.inlet.p
        or per-side overrides:
        bc.left.inlet.rho,   bc.left.inlet.u,   bc.left.inlet.v,   bc.left.inlet.p
        bc.right.inlet.rho,  bc.right.inlet.u,  bc.right.inlet.v,  bc.right.inlet.p
        bc.bottom.inlet.rho, bc.bottom.inlet.u, bc.bottom.inlet.v, bc.bottom.inlet.p
        bc.top.inlet.rho,    bc.top.inlet.u,    bc.top.inlet.v,    bc.top.inlet.p

    - outlet (1D/2D):
        bc.outlet.p
        or per-side overrides:
        bc.left.outlet.p, bc.right.outlet.p, bc.bottom.outlet.p, bc.top.outlet.p

  Periodic/cyclic is intentionally NOT supported (will throw if requested).
*/

namespace boundary {

enum class BcType {
    Inlet,
    Outlet,
    ZeroGradient,
    SlipWall,
    Symmetry,
    Internal
};

struct SideBc1D {
    BcType type{BcType::ZeroGradient};
    Prim1 inlet{};
    double pout{0.0};
};

struct SideBc2D {
    BcType type{BcType::ZeroGradient};
    Prim2 inlet{};
    double pout{0.0};
};

struct Bc1D {
    double gamma{1.4};
    SideBc1D left{};
    SideBc1D right{};
};

struct Bc2D {
    double gamma{1.4};
    SideBc2D left{};
    SideBc2D right{};
    SideBc2D bottom{};
    SideBc2D top{};
};

// Read from cfg (priority: per-side key -> global bc -> zeroGradient)
// 1D keys: bc.left bc.right bc
Bc1D read1D(const Cfg& cfg);
// 2D keys: bc.left bc.right bc.bottom bc.top bc
Bc2D read2D(const Cfg& cfg);

// Apply to ghost cells (only fills ghosts; interior unchanged)
void apply1D(std::vector<Vec3>& U, int nx, int ng, const Bc1D& bc);
void apply2D(std::vector<Vec4>& U, int nx, int ny, int ng, const Bc2D& bc);

} // namespace boundary
