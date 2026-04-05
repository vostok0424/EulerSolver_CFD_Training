#pragma once
#include "state.hpp"
#include <mpi.h>
#include <string>
#include <vector>

// Write a 2D solution as a legacy VTK rectilinear-grid file (.vtk).
//
// The solver state U is cell-centered and includes ghost cells.
// This writer outputs interior-cell data only, stored as CELL_DATA on a
// 2D RECTILINEAR_GRID with x- and y-coordinates defined at cell edges.
//
// Output fields:
//   - rho      : density
//   - velocity : vector field (u, v, 0)
//   - p        : pressure
//   - rho_u    : x-momentum
//   - rho_v    : y-momentum
//   - E        : total energy from the conservative state
//
// Notes:
// - Only interior cells [ng .. ng+nx-1] x [ng .. ng+ny-1] are written.
// - Grid dimensions in VTK are (nx+1, ny+1, 1).
// - (x0, x1) and (y0, y1) define the physical domain bounds of the interior mesh.
void writeVTK2D(const std::string& filename,
                const std::vector<Vec4>& U,
                int nx, int ny, int ng,
                double x0, double x1,
                double y0, double y1,
                double gamma);

// MPI (Plan A): gather all ranks' interior cell data on rank 0, assemble the
// global 2D field, and write a single legacy VTK rectilinear-grid file (.vtk).
//
// - `Ulocal` is this-rank solution including ghosts; only interior
//   [ng .. ng+nxLocal-1] x [ng .. ng+nyLocal-1] is gathered.
// - (iBeg, jBeg) are this-rank interior block global start indices (0-based).
// - `nxGlobal`, `nyGlobal` are the full global cell counts.
// - Only rank 0 writes `filename`; other ranks participate in gathers and return.
void writeVTK2D_GatherMPI(const std::string& filename,
                          const std::vector<Vec4>& Ulocal,
                          int nxLocal, int nyLocal, int ng,
                          int iBeg, int jBeg,
                          int nxGlobal, int nyGlobal,
                          double x0, double x1,
                          double y0, double y1,
                          double gamma,
                          MPI_Comm comm);
