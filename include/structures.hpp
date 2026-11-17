#pragma once

#include "constants.hpp"
#include <array>
#include <optional>
#include <string>
#include <vector>


// Store rho, rho*u, rho*v and rho*E. The specific_* names denote conserved quantities per unit volume.
struct Conservative_vector
{
    double specific_mass{};
    double specific_x_momentum{};
    double specific_y_momentum{};
    double specific_energy{};
};


// Point states in increasing transverse coordinate; low-order schemes use slot zero.
using FaceQuadratureState = std::array<Conservative_vector, face_quadrature_count>;


// Store density, x/y velocity and pressure.
struct Primitive_vector
{
    double density{};
    double x_velocity{};
    double y_velocity{};
    double pressure{};
};


// Store mass, x/y momentum and energy fluxes in the direction selected by the flux calculation.
struct Flux_vector
{
    double mass_flux{};
    double x_momentum_flux{};
    double y_momentum_flux{};
    double energy_flux{};
};


// Store primitive variables, sound speed and total enthalpy per unit mass for one evaluated state.
struct EvaluatedState
{
    double density{};
    double x_velocity{};
    double y_velocity{};
    double pressure{};
    double sound_speed{};
    double specific_total_enthalpy{};
};


// Runtime mesh metadata: global physical bounds and this rank's interior and ghosted cell counts.
// Distinct from GridParameters, which holds only input controls.
struct GridBlock2DInfo
{
    int nx_global{};
    int ny_global{};

    int i_beg{};
    int j_beg{};
    int nx{};
    int ny{};

    int ng{};
    int nx_tot{};
    int ny_tot{};

    double x0{};
    double x1{};
    double y0{};
    double y1{};
};


// Select first-order constant, second-order MUSCL or fifth-order normal WENO-Z reconstruction.
enum class ReconstructionScheme
{
    first_order,
    muscl,
    weno_z5
};


// Select the MUSCL slope: centered (none), minmod or van Leer.
enum class SlopeLimiter
{
    none,
    minmod,
    van_leer
};


// Store the reconstruction scheme and its numerical parameters.
struct ReconstructionOptions
{
    ReconstructionScheme scheme = ReconstructionScheme::first_order;

    // Slope limiter used only by MUSCL.
    SlopeLimiter limiter = SlopeLimiter::van_leer;
};


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


// Select the normal-shock propagation direction.
enum class ShockDir
{
    PlusX,
    MinusX,
    PlusY,
    MinusY
};

// One rectangular initialization override. Missing bounds use the local domain bounds.
// primitive is a direct state, or the upstream state when shock_mach > 0 (which must then exceed one).
struct SetFieldsRegion
{
    std::optional<double> x_min, x_max, y_min, y_max;
    Primitive_vector primitive{};
    double shock_mach = -1.0;
    ShockDir shock_dir = ShockDir::PlusX;
};

// Apply the background first, then regions in vector order; later regions overwrite overlaps.
struct SetFieldsOptions
{
    Primitive_vector background{};
    std::vector<SetFieldsRegion> regions;
};


// Select grid-point averages (point) or direct interior-cell values (cell).
// Output fields are rho, velocity, p, rho_u, rho_v and E; E is energy per unit volume.
enum class VTKDataOutputType
{
    point,
    cell
};


// Global input mesh controls; physical bounds exclude ghost cells. Defaults are assigned by the reader.
struct GridParameters
{
    int nx, ny, ng;
    double x0, x1, y0, y1;
};

// Gas, time, numerical-method names and output controls consumed during Solver construction.
struct SolverParameters
{
    double gamma, cfl, final_time;
    int output_every;
    double output_time_interval;
    bool write_final;
    bool positivity_preserving = true;
    std::string out_prefix, flux, time_integrator;
    VTKDataOutputType vtk_data_output_type;
};

// Complete typed input shared between the reading entry point and solver construction.
// Use only after read_parameters returns SUCCESS; a failed read can leave partial assignments.
struct CaseParameters
{
    int mpi_px, mpi_py;
    GridParameters grid;
    SolverParameters solver;
    ReconstructionOptions reconstruction;
    BoundaryConditions2D boundary;
    SetFieldsOptions set_fields;
};
