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
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_rusanov_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output );

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

// Compute the AUSM flux vector at a discontinuity interface.
//
// Implementation: AUSM-type quadratic Mach and cubic pressure splitting, arithmetic-mean sound speed,
// and density-weighted split mass flux; velocity and enthalpy are upwinded by the mass-flux sign.
// This is an adaptation of the cited AUSM splitting approach with the interface choices stated above.
//
// Reference: M.-S. Liou and C. J. Steffen Jr. (1993), "A New Flux Splitting Scheme",
// J. Comput. Phys. 107(1), 23-39. https://doi.org/10.1006/jcph.1993.1122
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
// Implementation: ideal-gas normal Riemann solution sampled at x/t = 0; the sample selects tangential velocity.
// Newton iteration uses a PVRS initial pressure guess; vacuum, invalid iterates and nonconvergence return
// failure codes rather than applying pressure clipping or an approximate-solver fallback.
//
// References: S. K. Godunov (1959), "A difference method for numerical calculation of discontinuous
// solutions of the equations of hydrodynamics", Mat. Sb. (N.S.) 47(89)(3), 271-306.
// https://www.mathnet.ru/eng/sm4873
// E. F. Toro (2009), "Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction",
// 3rd ed., Springer, Ch. 4 (exact Euler Riemann solution) and Ch. 6 (Godunov method).
// https://doi.org/10.1007/b79761
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
