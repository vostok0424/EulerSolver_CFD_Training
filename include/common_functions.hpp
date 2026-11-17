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
// known_pressure (in, optional) - verified pressure of this exact state and gamma; nullptr recovers it
ReturnCodes calc_flow_variables( const Conservative_vector &state, double gamma, EvaluatedState &output,
    const double *known_pressure = nullptr );

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


// Add conservative states componentwise.
//
// left (in) - left operand of the conservative-state operation
// right (in) - right operand of the conservative-state operation
//
// State algebra is componentwise and performs no physical-state validation or repair.
Conservative_vector operator+( const Conservative_vector &left, const Conservative_vector &right );

// Subtract conservative states componentwise.
//
// left (in) - left operand of the conservative-state operation
// right (in) - right operand of the conservative-state operation
Conservative_vector operator-( const Conservative_vector &left, const Conservative_vector &right );

// Scale all conservative components by a scalar.
//
// factor (in) - scalar multiplier of all conservative components
// state (in) - conserved variables of the flow state
Conservative_vector operator*( double factor, const Conservative_vector &state );

// Divide all conservative components by a scalar.
//
// state (in) - conserved variables of the flow state
// divisor (in) - scalar divisor of all conservative components
Conservative_vector operator/( const Conservative_vector &state, double divisor );

// Accumulate a conservative increment into the supplied state.
//
// state (in/out) - conserved variables of the flow state
// increment (in) - componentwise conservative increment
Conservative_vector &operator+=( Conservative_vector &state, const Conservative_vector &increment );


// Store rho, rho*u, rho*v and rho*E in four consecutive doubles.
//
// state (in) - conserved variables of the flow state
// output (out) - buffer for four doubles ordered as density, x/y momentum and total energy
void pack_state( const Conservative_vector &state, double *output );

// Recover a conservative state from four doubles in rho, rho*u, rho*v, rho*E order.
//
// input (in) - four packed conservative components in density, x/y momentum, energy order
Conservative_vector unpack_state( const double *input );

// Return the four conserved components in serialization order.
//
// state (in) - conserved variables of the flow state
std::array<double, conservative_variable_count> state_components( const Conservative_vector &state );

// Return flux components in mass, x-momentum, y-momentum and energy order.
//
// flux (in) - flux vector of conserved variables
std::array<double, conservative_variable_count> flux_components( const Flux_vector &flux );


// Read a velocity component; throw on an invalid coordinate direction.
//
// primitive (in) - primitive flow state
// direction (in) - x_direction or y_direction
double velocity_component( const Primitive_vector &primitive, int direction );

// Return uniform global x spacing; the solver validates the mesh.
//
// grid (in) - global mesh bounds and cell counts
double grid_spacing_x( const GridBlock2DInfo &grid );

// Return uniform global y spacing; the solver validates the mesh.
//
// grid (in) - global mesh bounds and cell counts
double grid_spacing_y( const GridBlock2DInfo &grid );
