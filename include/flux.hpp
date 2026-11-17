#pragma once

#include "structures.hpp"
#include <string>

// Select the numerical flux for the two-dimensional Euler equations.
enum class FluxScheme
{
    rusanov,
    hllc
};

// Read rusanov or hllc; unknown names throw a configuration error.
//
// name (in) - requested numerical-method name
FluxScheme read_flux_scheme( const std::string &name );

// Compute the Rusanov flux vector at a discontinuity interface.
//
// Implementation: local Lax-Friedrichs flux with dissipation speed max(|u_n,L| + c_L, |u_n,R| + c_R).
//
// Reference: V. V. Rusanov (1961), "The calculation of the interaction of non-stationary shock waves
// with barriers", Zh. Vychisl. Mat. Mat. Fiz. 1(2), 267-279.
// https://www.mathnet.ru/eng/zvmmf7987
//
// left_state (in) - conserved variables to the left of the interface
// right_state (in) - conserved variables to the right of the interface
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
// left_flow, right_flow (in, optional) - evaluated states matching the exact supplied states and gamma;
// nullptr evaluates that side internally. Supplied cache data must belong to the same immutable stage.
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_rusanov_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output, const EvaluatedState *left_flow = nullptr,
    const EvaluatedState *right_flow = nullptr );

// Compute the HLLC flux vector at a discontinuity interface.
//
// Implementation: three-wave HLLC with S_L = min(u_n,L - c_L, u_n,R - c_R) and
// S_R = max(u_n,L + c_L, u_n,R + c_R); tangential velocity is carried through each star state.
//
// Reference: E. F. Toro, M. Spruce and W. Speares (1994), "Restoration of the contact surface
// in the HLL-Riemann solver", Shock Waves 4, 25-34. https://doi.org/10.1007/BF01414629
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
