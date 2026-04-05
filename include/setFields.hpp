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

  2D regions:
    setFields.nRegions = N
    setFields.regionK.xMin, setFields.regionK.xMax
    setFields.regionK.yMin, setFields.regionK.yMax
    setFields.regionK.rho,  setFields.regionK.u,  setFields.regionK.v, setFields.regionK.p

  Notes
  - Region indices K are 0-based or 1-based depending on the implementation in setFields.cpp.
    (See setFields.cpp for the exact convention used.)
  - Overlaps: if multiple regions overlap, later regions typically overwrite earlier ones.
*/

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
