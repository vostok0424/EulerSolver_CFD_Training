#pragma once

#include "structures.hpp"
#include <string>

// Select the numerical flux for the two-dimensional Euler equations.
enum class FluxScheme
{
    rusanov,
    hllc,
    ausm,
    godunov
};

// Read rusanov, hllc, ausm or godunov; unknown names throw a configuration error.
//
// name (in) - requested numerical-method name
FluxScheme read_flux_scheme( const std::string &name );

// Compute the Rusanov flux vector at a discontinuity interface.
//
// left_state (in) - conserved variables to the left of the interface
// right_state (in) - conserved variables to the right of the interface
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_rusanov_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

// Compute the HLLC flux vector at a discontinuity interface.
//
// left_state (in) - conserved variables to the left of the interface
// right_state (in) - conserved variables to the right of the interface
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_hllc_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

// Compute the AUSM flux vector at a discontinuity interface.
//
// left_state (in) - conserved variables to the left of the interface
// right_state (in) - conserved variables to the right of the interface
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_ausm_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

// Compute the exact-directional Godunov flux vector at a discontinuity interface.
//
// left_state (in) - conserved variables to the left of the interface
// right_state (in) - conserved variables to the right of the interface
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_godunov_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

// Compute the flux vector at a discontinuity interface using the selected scheme.
//
// scheme (in) - selected numerical flux scheme
// left_state (in) - conserved variables to the left of the interface
// right_state (in) - conserved variables to the right of the interface
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_numerical_flux( FluxScheme scheme, const Conservative_vector &left_state,
    const Conservative_vector &right_state, int dir, double gamma, Flux_vector &output );
