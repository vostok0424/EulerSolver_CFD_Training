#pragma once
#include "state.hpp"
#include "cfg.hpp"
#include <vector>
#include <string>

/*
  setFields.hpp
  ------------
  OpenFOAM-like field initialization utility.

  Purpose
  - This module provides a simple, cfg-driven way to overwrite parts of the initial
    field after the base initial condition (IC) has been created.
  - Typical workflow in the solver:
      1) Build the base IC (e.g., Sod) into U.
      2) If setFields.use == true, call setFields*() to apply region overrides.
      3) Apply boundary conditions (ghost cells) before the first time step.

  Data model
  - U is cell-centered and includes ghost cells.
  - setFields only modifies interior cells; ghost cells are handled later by BCs.
  - Variables are specified in primitive form (rho, u, v, p) and converted to
    conservative storage in U.

  Config keys

  Common:
    setFields.use = true|false
    setFields.bg.rho, setFields.bg.u, setFields.bg.p

  1D regions:
    setFields.nRegions = N
    setFields.regionK.xMin, setFields.regionK.xMax
    setFields.regionK.rho,  setFields.regionK.u,  setFields.regionK.p
    or shock-defined mode:
    setFields.regionK.shockMach, setFields.regionK.rho, setFields.regionK.p
    optional: setFields.regionK.u, setFields.regionK.shockDir

  2D regions:
    setFields.nRegions = N
    setFields.regionK.xMin, setFields.regionK.xMax
    setFields.regionK.yMin, setFields.regionK.yMax
    setFields.regionK.rho,  setFields.regionK.u,  setFields.regionK.v, setFields.regionK.p
    or shock-defined mode:
    setFields.regionK.shockMach, setFields.regionK.rho, setFields.regionK.p
    optional: setFields.regionK.u, setFields.regionK.v, setFields.regionK.shockDir

  Notes
  - Region indices K are 0-based or 1-based depending on the implementation in setFields.cpp.
    (See setFields.cpp for the exact convention used.)
  - Overlaps: if multiple regions overlap, later regions typically overwrite earlier ones.
  - If setFields.regionK.shockMach is present, that region is interpreted in incident-shock mode.
    In this mode, rho/p/(u,v) describe the ahead-of-shock primitive state, and the
    region is filled with the computed post-shock state.
  - shockDir is intended to specify the shock propagation direction (for example +x, -x,
    +y, -y). The exact accepted values are defined in setFields.cpp.
*/

// setFields also supports a shock-defined region mode.
// When a region provides setFields.regionK.shockMach, the region primitive state is not
// taken directly from rho/u(/v)/p. Instead, rho/p and optional ahead-state velocity
// components are interpreted as the pre-shock state, and the region is initialized with
// the corresponding post-shock state computed from the incident shock Mach number.

// Apply cfg-defined region overrides to a 1D field U.
// Modifies interior cells only (ghost cells are not written here).
void setFields1D(std::vector<Vec3>& U,
                 int nx, int ng,
                 double x0, double x1,
                 double gamma,
                 const Cfg& cfg);

// Apply cfg-defined region overrides to a 2D field U.
// Modifies interior cells only (ghost cells are not written here).
void setFields2D(std::vector<Vec4>& U,
                 int nx, int ny, int ng,
                 double x0, double x1,
                 double y0, double y1,
                 double gamma,
                 const Cfg& cfg);
