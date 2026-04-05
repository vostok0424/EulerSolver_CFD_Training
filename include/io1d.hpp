#pragma once
#include "state.hpp"
#include "cfg.hpp"
#include <mpi.h>
#include <string>
#include <vector>

// Write 1D solution as a legacy VTK rectilinear-grid file (.vtk).
//
// The solver state U is cell-centered and includes ghost cells.
// This writer outputs interior-cell data only, stored as CELL_DATA on a
// 1D RECTILINEAR_GRID with x-coordinates defined at cell edges.
//
// Output fields:
//   - rho   : density
//   - u     : velocity
//   - p     : pressure
//   - rho_u : momentum
//   - E     : total energy per unit mass derived from the conservative state
//
// Notes:
// - Only interior cells [ng .. ng+nx-1] are written.
// - Grid dimensions in VTK are (nx+1, 1, 1).
// - x0 and x1 define the physical domain bounds of the interior mesh.
void writeVTK1D(const std::string& filename,
                const std::vector<Vec3>& U,
                int nx, int ng,
                double x0, double x1,
                double gamma);

// MPI (Plan A): gather all ranks' interior cell data on rank 0, assemble the
// global 1D field, and write a single legacy VTK rectilinear-grid file (.vtk).
//
// - `Ulocal` is this-rank solution including ghosts; only interior
//   [ng .. ng+nxLocal-1] is gathered.
// - `iBeg` is this-rank interior block global start index (0-based).
// - `nxGlobal` is the full global cell count.
// - Only rank 0 writes `filename`; other ranks participate in gathers and return.
void writeVTK1D_GatherMPI(const std::string& filename,
                          const std::vector<Vec3>& Ulocal,
                          int nxLocal, int ng,
                          int iBeg,
                          int nxGlobal,
                          double x0, double x1,
                          double gamma,
                          MPI_Comm comm);
