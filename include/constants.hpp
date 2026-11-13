#pragma once

#include <cstddef>


constexpr int spatial_dimension = 2;
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
    INVALID_THERMODYNAMIC_STATE,
    INVALID_WAVE_SPEED,
    INVALID_STAR_STATE,
    VACUUM_STATE,
    INVALID_PRESSURE_ESTIMATE,
    INVALID_NEWTON_ITERATION,
    PRESSURE_ITERATION_NOT_CONVERGED,
    INVALID_FLUX_SCHEME,
    // Configuration-reading failures; callers must not use partially filled parameters.
    PARAMETER_FILE_ERROR,
    PARAMETER_PARSE_ERROR,
    MISSING_PARAMETER,
    INVALID_PARAMETER_VALUE
};
