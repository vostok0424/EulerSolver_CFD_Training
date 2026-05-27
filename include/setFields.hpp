#pragma once
#include "state.hpp"
#include "cfg.hpp"
#include <vector>
#include <string>

/*
  setFields.hpp
  -------------
  OpenFOAM-like initial-field setup utility.

  Purpose
  - This module provides a cfg-driven way to construct the complete initial
    conservative field for the 2D Euler solver.
  - The initialization procedure is:
      1) Fill all local interior cells with the background primitive state
         specified by setFields.bg.*.
      2) Apply rectangular region overrides in ascending region index.
      3) Apply boundary conditions later in the solver to fill ghost cells.
  - This is now the single initial-field path used by the solver; the legacy
    built-in IC module has been removed.

  Data model
  - U is cell-centered and includes ghost cells.
  - setFields2D() modifies interior cells only; ghost cells are handled later
    by the boundary-condition module.
  - Input values are specified in primitive form (rho, u, v, p) and converted
    to conservative storage in U.
  - If multiple regions overlap, later regions overwrite earlier ones.

  Config keys

  Background state:
    setFields.bg.rho
    setFields.bg.u
    setFields.bg.v
    setFields.bg.p

  Region count:
    setFields.nRegions = N

  2D rectangular regions:
    setFields.regionK.xMin
    setFields.regionK.xMax
    setFields.regionK.yMin
    setFields.regionK.yMax

  Direct primitive region state:
    setFields.regionK.rho
    setFields.regionK.u
    setFields.regionK.v
    setFields.regionK.p

  Optional shock-defined region state:
    setFields.regionK.shockMach
    setFields.regionK.shockDir

  Notes
  - Region indices K start from 1.
  - If setFields.regionK.shockMach > 1, the region is interpreted in
    incident-shock mode. In this mode, rho/p/(u,v) describe the ahead-of-shock
    primitive state, and the region is filled with the computed post-shock state.
  - shockDir specifies the shock propagation direction, for example +x, -x,
    +y, or -y. The accepted values are defined in setFields.cpp.
*/

// Build the 2D initial conservative field from cfg-defined background and
// rectangular region settings. Modifies interior cells only; ghost cells are
// filled later by boundary conditions.
void setFields2D(std::vector<Vec4>& U,
                 int nx, int ny, int ng,
                 double x0, double x1,
                 double y0, double y1,
                 double gamma,
                 const Cfg& cfg);
