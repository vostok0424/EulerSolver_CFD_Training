#pragma once

#include "structures.hpp"


// Limit one shared face flux using the density/pressure cut-off of Hu, Adams and Shu (2013).
// Reference: https://doi.org/10.1016/j.jcp.2013.01.024, equations (11)-(16), (28).
//
// left_average, right_average (in) - admissible stage cell averages on either side of the face
// low_order_flux (in) - common positivity-preserving Rusanov baseline, computed from cell averages
// high_order_flux (in) - common high-order face-average flux, after transverse quadrature
// update_factor (in) - 2*dt/(alpha_d*dx_d), nonnegative; includes the multidimensional partition
// gamma (in) - finite ideal-gas heat-capacity ratio greater than one
// density_floor, pressure_floor (in) - finite positive admissibility thresholds
// limited_flux (out) - low_order_flux + theta*(high_order_flux-low_order_flux)
// theta (out) - shared density/pressure limiting factor in [0,1]
//
// Returns SUCCESS only if both auxiliary states U_left-factor*flux and U_right+factor*flux
// satisfy the floors. Caller supplies the baseline wave-speed/CFL conditions, SSP stage step size,
// admissible reconstructed Riemann inputs and identical final fluxes across MPI interfaces.
// This function neither chooses dt nor repairs invalid low-order baselines or non-finite inputs.
// INVALID_PARAMETER_VALUE reports invalid factors/floors; NEGATIVE_DENSITY/NEGATIVE_PRESSURE
// also report positive values below the supplied floors. Outputs are unchanged on failure.
ReturnCodes limit_positive_flux( const Conservative_vector &left_average,
    const Conservative_vector &right_average, const Flux_vector &low_order_flux,
    const Flux_vector &high_order_flux, double update_factor, double gamma,
    double density_floor, double pressure_floor, Flux_vector &limited_flux, double &theta );


// Contract a finite reconstructed point toward its admissible source cell average.
// average (in) - source cell mean, satisfying the supplied positive floors
// gamma, density_floor, pressure_floor (in) - ideal-gas ratio and positive thresholds
// point (in/out) - finite reconstructed state; unchanged on failure
// Returns SUCCESS when the point is admissible. Does not modify cell averages.
ReturnCodes limit_positive_point( const Conservative_vector &average, double gamma,
    double density_floor, double pressure_floor, Conservative_vector &point );
