#pragma once

#include <cstddef>


// Positive regularization used by the WENO-Z nonlinear weights.
#define WENO_Z_EPS 1e-12

static_assert( WENO_Z_EPS > 0.0, "WENO_Z_EPS must be positive" );


// Base admissibility thresholds and strict Rusanov auxiliary-state CFL margin.
#define POSITIVITY_DENSITY_FLOOR 1e-13
#define POSITIVITY_PRESSURE_FLOOR 1e-13
#define POSITIVITY_MAX_CFL 0.45


// Four-point Gauss-Legendre rule for a face average on [-1/2, 1/2].
constexpr int face_quadrature_count = 4;
constexpr double face_quadrature_nodes[4] = { -0.43056815579702629, -0.16999052179242813, 0.16999052179242813, 0.43056815579702629 };
constexpr double face_quadrature_weights[4] = { 0.17392742256872673, 0.32607257743127327, 0.32607257743127327, 0.17392742256872673 };


constexpr std::size_t conservative_variable_count = 4;

// Select the coordinate axis; integer-valued interfaces also reject invalid direction values.
enum CoordinateDirection
{
    x_direction = 0,
    y_direction = 1
};


// Define the shared component order for serialization and conservative eigenvectors.
enum ConservativeComponent
{
    density_component = 0,
    x_momentum_component = 1,
    y_momentum_component = 2,
    total_energy_component = 3
};


// Report calculation success or failure; NEGATIVE_DENSITY and NEGATIVE_PRESSURE include zero.
enum ReturnCodes
{
    SUCCESS = 0,
    NON_FINITE_STATE,
    INVALID_GAMMA,
    NEGATIVE_DENSITY,
    NEGATIVE_PRESSURE,
    INVALID_INTERNAL_ENERGY,
    INVALID_SOUND_VELOCITY,
    INVALID_ENTHALPY,
    INVALID_DIRECTION,
    NON_FINITE_FLUX,
    INVALID_WAVE_SPEED,
    INVALID_STAR_STATE,
    INVALID_FLUX_SCHEME,
    // Configuration-reading failures; callers must not use partially filled parameters.
    PARAMETER_FILE_ERROR,
    PARAMETER_PARSE_ERROR,
    MISSING_PARAMETER,
    INVALID_PARAMETER_VALUE
};
