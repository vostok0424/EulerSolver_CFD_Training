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
FluxScheme read_flux_scheme( const std::string &name );

// left_state, right_state (in) - conservative states on the two sides of the face
// dir (in) - normal direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - numerical flux; unchanged on failure for all functions below
ReturnCodes calc_rusanov_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

ReturnCodes calc_hllc_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

ReturnCodes calc_ausm_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

ReturnCodes calc_godunov_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

// Select a flux function with an explicit scheme switch.
ReturnCodes calc_numerical_flux( FluxScheme scheme, const Conservative_vector &left_state,
    const Conservative_vector &right_state, int dir, double gamma, Flux_vector &output );
