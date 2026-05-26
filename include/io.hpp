#pragma once
#include "state.hpp"
#include <mpi.h>
#include <string>
#include <vector>

// VTK data writting mode for legacy rectilinear-grid output.
//
// - point: convert cell-centered data to POINT_DATA by averaging neighbouring
//          cell values onto grid points. This is convenient for smooth-looking
//          visualization in ParaView.
// - cell : write the conservative cell-centered solution directly as CELL_DATA.
//          This preserves the finite-volume storage location and is better for
//          debugging discontinuities, conservation, and cell-wise diagnostics.
enum class VTKDataOutputType {
    Point,
    Cell
};

// Write a 2D solution as a legacy VTK rectilinear-grid file (.vtk).
//
// The solver state U is cell-centered and includes ghost cells. This default
// point-data writer keeps the current behaviour: it writes POINT_DATA by averaging
// interior cell-centered values onto grid points.
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
// - Only interior cells [ng .. ng+nx-1] x [ng .. ng+ny-1] are used.
// - Grid dimensions in VTK are (nx+1, ny+1, 1).
// - (x0, x1) and (y0, y1) define the physical domain bounds of the interior mesh.
void writeVTK2D_PointData(const std::string& filename,
                          const std::vector<Vec4>& U,
                          int nx, int ny, int ng,
                          double x0, double x1,
                          double y0, double y1,
                          double gamma);

// Write a 2D solution as CELL_DATA on a legacy VTK rectilinear grid (.vtk).
//
// This function writes one value per finite-volume cell without point averaging.
// It is intended for exact cell-wise inspection of the numerical solution.
void writeVTK2D_CellData(const std::string& filename,
                         const std::vector<Vec4>& U,
                         int nx, int ny, int ng,
                         double x0, double x1,
                         double y0, double y1,
                         double gamma);

// MPI (Plan A): gather all ranks' interior cell data on rank 0, assemble the
// global 2D field, and write a single legacy VTK rectilinear-grid file (.vtk).
//
// This point-data MPI writer keeps the current behaviour: rank 0 writes POINT_DATA
// after converting gathered cell-centered values onto grid points.
//
// - `Ulocal` is this-rank solution including ghosts; only interior
//   [ng .. ng+nxLocal-1] x [ng .. ng+nyLocal-1] is gathered.
// - (iBeg, jBeg) are this-rank interior block global start indices (0-based).
// - `nxGlobal`, `nyGlobal` are the full global cell counts.
// - Only rank 0 writes `filename`; other ranks participate in gathers and return.
void writeVTK2D_PointData_GatherMPI(const std::string& filename,
                                    const std::vector<Vec4>& Ulocal,
                                    int nxLocal, int nyLocal, int ng,
                                    int iBeg, int jBeg,
                                    int nxGlobal, int nyGlobal,
                                    double x0, double x1,
                                    double y0, double y1,
                                    double gamma,
                                    MPI_Comm comm);

// MPI version of CELL_DATA VTK output. All ranks gather interior cell data on
// rank 0, then rank 0 writes one value per global finite-volume cell.
void writeVTK2D_CellData_GatherMPI(const std::string& filename,
                                   const std::vector<Vec4>& Ulocal,
                                   int nxLocal, int nyLocal, int ng,
                                   int iBeg, int jBeg,
                                   int nxGlobal, int nyGlobal,
                                   double x0, double x1,
                                   double y0, double y1,
                                   double gamma,
                                   MPI_Comm comm);
