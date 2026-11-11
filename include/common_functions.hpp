#pragma once

#include "structures.hpp"
#include <cmath>
#include <string>


// Check whether all values in the array are finite.
bool all_values_finite( const double *values, std::size_t count );

// Check whether all four conservative components are finite.
bool is_finite_state( const Conservative_vector &state );

// Check whether all four flux components are finite.
bool is_finite_flux( const Flux_vector &flux );

// Throw with the operation name and numeric return code when a calculation fails.
void require_success( ReturnCodes code, const char *operation );

// Remove leading and trailing whitespace from a string.
std::string trim_string( std::string value );

// Convert a string to lowercase using the current character locale.
std::string to_lowercase( std::string value );


// Compute the conserved variables from density, velocity and pressure.
// Flow calculations return SUCCESS only after assigning output. On failure, output is unchanged.
ReturnCodes calc_conservative_variables( const Primitive_vector &primitive, double gamma, Conservative_vector &output );

// Compute density, velocity and pressure from the conserved variables.
ReturnCodes calc_primitive_variables( const Conservative_vector &state, double gamma, Primitive_vector &output );

// Compute the ideal-gas speed of sound.
ReturnCodes calc_sound_velocity( const Primitive_vector &primitive, double gamma, double &output );

// Compute primitive variables, sound speed and specific total enthalpy.
ReturnCodes calc_flow_variables( const Conservative_vector &state, double gamma, EvaluatedState &output );

// Compute the squared velocity magnitude from the conserved variables.
ReturnCodes velocity_squared( const Conservative_vector &state, double &output );

// Recover pressure from the conserved variables.
ReturnCodes pressure_from_conservative( const Conservative_vector &state, double gamma, double &output );

// Compute internal energy per unit mass by subtracting kinetic energy.
ReturnCodes specific_internal_energy_from_conservative( const Conservative_vector &state, double &output );


// Compute the Euler flux in the selected coordinate direction.
// evaluated (in) - cached flow variables from the same conservative state; direction is 0 for x or 1 for y.
ReturnCodes calc_physical_flux( const Conservative_vector &state, const EvaluatedState &evaluated, int direction,
    Flux_vector &output );

// Compute the Euler flux in the selected coordinate direction.
ReturnCodes calc_physical_flux( const Conservative_vector &state, int direction, double gamma, Flux_vector &output );

// Compute the Euler flux in the selected coordinate direction.
ReturnCodes calc_physical_flux( const Primitive_vector &primitive, int direction, double gamma, Flux_vector &output );
