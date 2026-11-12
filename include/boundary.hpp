#pragma once
#include "structures.hpp"
#include <vector>


// Fill physical-boundary ghost cells; leave interior cells and internal MPI boundaries unchanged.
//
// state (in/out) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// bc (in) - physical boundary types, inlet states, outlet pressures and gas model
void apply_boundary_conditions( std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    const BoundaryConditions2D &bc );
