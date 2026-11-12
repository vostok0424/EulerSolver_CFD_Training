#pragma once
#include "structures.hpp"
#include <mpi.h>
#include <string>
#include <vector>


// Write interior-cell fields to a VTK rectilinear grid after averaging them onto grid points.
//
// filename (in) - destination VTK file path
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
void write_vtk_2d_point_data( const std::string &filename, const std::vector<Conservative_vector> &state, int nx,
    int ny, int ng, double x0, double x1, double y0, double y1, double gamma );


// Write one set of field values per interior cell to a VTK rectilinear grid.
//
// filename (in) - destination VTK file path
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
void write_vtk_2d_cell_data( const std::string &filename, const std::vector<Conservative_vector> &state, int nx, int ny,
    int ng, double x0, double x1, double y0, double y1, double gamma );


// Gather interior states on communicator rank 0 and write a single POINT_DATA file.
//
// filename (in) - destination VTK file path
// ulocal (in) - ghosted conservative field; only interior cells contribute to this operation
// nx_local (in) - local number of interior cells along x
// ny_local (in) - local number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// i_beg (in) - global x index of the first local interior cell
// j_beg (in) - global y index of the first local interior cell
// nx_global (in) - global number of interior cells along x
// ny_global (in) - global number of interior cells along y
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
// comm (in) - communicator containing all ranks participating in the collective write
//
// All ranks in comm must participate; i_beg and j_beg locate each local interior in the global mesh.
void write_vtk_2d_point_data_gather_mpi( const std::string &filename, const std::vector<Conservative_vector> &ulocal,
    int nx_local, int ny_local, int ng, int i_beg, int j_beg, int nx_global, int ny_global, double x0, double x1,
    double y0, double y1, double gamma, MPI_Comm comm );


// Gather interior states on communicator rank 0 and write a single CELL_DATA file.
//
// filename (in) - destination VTK file path
// ulocal (in) - ghosted conservative field; only interior cells contribute to this operation
// nx_local (in) - local number of interior cells along x
// ny_local (in) - local number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// i_beg (in) - global x index of the first local interior cell
// j_beg (in) - global y index of the first local interior cell
// nx_global (in) - global number of interior cells along x
// ny_global (in) - global number of interior cells along y
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
// comm (in) - communicator containing all ranks participating in the collective write
void write_vtk_2d_cell_data_gather_mpi( const std::string &filename, const std::vector<Conservative_vector> &ulocal,
    int nx_local, int ny_local, int ng, int i_beg, int j_beg, int nx_global, int ny_global, double x0, double x1,
    double y0, double y1, double gamma, MPI_Comm comm );
