#pragma once
#include "structures.hpp"
#include <mpi.h>
#include <string>
#include <vector>


// Select grid-point averages (point) or direct interior-cell values (cell).
// Output fields are rho, velocity, p, rho_u, rho_v and E; E is energy per unit volume.
enum class VTKDataOutputType
{
    point,
    cell
};


// Write interior-cell fields to a VTK rectilinear grid after averaging them onto grid points.
// state (in) - row-major field including ng ghost cells; only the nx-by-ny interior is written.
// x0, x1, y0, y1 (in) - physical bounds of the interior domain; gamma closes the gas state.
void write_vtk_2d_point_data( const std::string &filename, const std::vector<Conservative_vector> &state, int nx,
    int ny, int ng, double x0, double x1, double y0, double y1, double gamma );


// Write one set of field values per interior cell to a VTK rectilinear grid.
void write_vtk_2d_cell_data( const std::string &filename, const std::vector<Conservative_vector> &state, int nx, int ny,
    int ng, double x0, double x1, double y0, double y1, double gamma );


// Gather interior states on communicator rank 0 and write a single POINT_DATA file.
// All ranks in comm must participate; i_beg and j_beg locate each local interior in the global mesh.
void write_vtk_2d_point_data_gather_mpi( const std::string &filename, const std::vector<Conservative_vector> &ulocal,
    int nx_local, int ny_local, int ng, int i_beg, int j_beg, int nx_global, int ny_global, double x0, double x1,
    double y0, double y1, double gamma, MPI_Comm comm );


// Gather interior states on communicator rank 0 and write a single CELL_DATA file.
void write_vtk_2d_cell_data_gather_mpi( const std::string &filename, const std::vector<Conservative_vector> &ulocal,
    int nx_local, int ny_local, int ng, int i_beg, int j_beg, int nx_global, int ny_global, double x0, double x1,
    double y0, double y1, double gamma, MPI_Comm comm );
