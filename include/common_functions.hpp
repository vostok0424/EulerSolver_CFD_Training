#pragma once

#include "structures.hpp"
#include <cmath>
#include <string>


// Check whether all values in the array are finite.
//
// values (in) - array of numeric values to inspect
// count (in) - number of array entries to inspect
bool all_values_finite( const double *values, std::size_t count );

// Check whether all four conservative components are finite.
//
// state (in) - conserved variables of the flow state
bool is_finite_state( const Conservative_vector &state );

// Check whether all four flux components are finite.
//
// flux (in) - flux vector of conserved variables
bool is_finite_flux( const Flux_vector &flux );

// Throw with the operation name and numeric return code when a calculation fails.
//
// code (in) - operation status; SUCCESS indicates valid numerical outputs
// operation (in) - flow operation name included in the failure report
void require_success( ReturnCodes code, const char *operation );

// Remove leading and trailing whitespace from a string.
//
// value (in) - input text copied before transformation
std::string trim_string( std::string value );

// Convert a string to lowercase using the current character locale.
//
// value (in) - input text copied before transformation
std::string to_lowercase( std::string value );


// Compute the conserved variables from density, velocity and pressure.
//
// primitive (in) - density, velocity and pressure of the flow state
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - computed conservative state
//
// Flow calculations return SUCCESS only after assigning output. On failure, output is unchanged.
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_conservative_variables( const Primitive_vector &primitive, double gamma, Conservative_vector &output );

// Compute density, velocity and pressure from the conserved variables.
//
// state (in) - conserved variables of the flow state
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - computed primitive state
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_primitive_variables( const Conservative_vector &state, double gamma, Primitive_vector &output );

// Compute the ideal-gas speed of sound.
//
// primitive (in) - density, velocity and pressure of the flow state
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - computed speed of sound
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_sound_velocity( const Primitive_vector &primitive, double gamma, double &output );

// Compute primitive variables, sound speed and specific total enthalpy.
//
// state (in) - conserved variables of the flow state
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - computed primitive variables, sound speed and specific total enthalpy
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_flow_variables( const Conservative_vector &state, double gamma, EvaluatedState &output );

// Compute the squared velocity magnitude from the conserved variables.
//
// state (in) - conserved variables of the flow state
// output (out) - computed squared velocity magnitude
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes velocity_squared( const Conservative_vector &state, double &output );

// Recover pressure from the conserved variables.
//
// state (in) - conserved variables of the flow state
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - computed pressure
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes pressure_from_conservative( const Conservative_vector &state, double gamma, double &output );

// Compute internal energy per unit mass by subtracting kinetic energy.
//
// state (in) - conserved variables of the flow state
// output (out) - computed internal energy per unit mass
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes specific_internal_energy_from_conservative( const Conservative_vector &state, double &output );


// Compute the Euler flux in the selected coordinate direction.
//
// state (in) - conserved variables of the flow state
// evaluated (in) - cached flow variables evaluated from the same conservative state
// direction (in) - coordinate direction: x_direction or y_direction
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_physical_flux( const Conservative_vector &state, const EvaluatedState &evaluated, int direction,
    Flux_vector &output );

// Compute the Euler flux in the selected coordinate direction.
//
// state (in) - conserved variables of the flow state
// direction (in) - coordinate direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_physical_flux( const Conservative_vector &state, int direction, double gamma, Flux_vector &output );

// Compute the Euler flux in the selected coordinate direction.
//
// primitive (in) - density, velocity and pressure of the flow state
// direction (in) - coordinate direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_physical_flux( const Primitive_vector &primitive, int direction, double gamma, Flux_vector &output );
