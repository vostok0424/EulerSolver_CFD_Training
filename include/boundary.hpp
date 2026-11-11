#pragma once
#include "structures.hpp"
#include "cfg.hpp"
#include <vector>


// Select inlet, outlet, extrapolation, reflection or an internal MPI interface.
enum class BoundaryType
{
    inlet,
    outlet,
    zero_gradient,
    slip_wall,
    symmetry,
    internal
};


// Store one side's boundary type and its inlet state or prescribed outlet pressure.
struct BoundarySide2D
{
    BoundaryType type{ BoundaryType::zero_gradient };
    Primitive_vector inlet{};
    double pout{ 0.0 };
};


// Store gamma and the boundary settings for the four sides of the local grid.
struct BoundaryConditions2D
{
    double gamma{ 1.4 };
    BoundarySide2D left{};
    BoundarySide2D right{};
    BoundarySide2D bottom{};
    BoundarySide2D top{};
};


// Read boundary types and their required inlet or outlet parameters.
// Side keys bc.left/right/bottom/top override bc; the default is zeroGradient.
// Inlets require rho, u, v and p under bc.<side>.inlet. or bc.inlet.; outlets require outlet.p.
BoundaryConditions2D read_boundary_conditions( const Cfg &cfg );


// Fill physical-boundary ghost cells; leave interior cells and internal MPI boundaries unchanged.
void apply_boundary_conditions( std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    const BoundaryConditions2D &bc );
