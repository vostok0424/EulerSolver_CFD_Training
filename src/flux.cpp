#include "common_functions.hpp"
#include "flux.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>


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
// left_flow, right_flow (in, optional) - same-stage states matching the supplied averages and gamma;
// null pointers request internal evaluation for that side
//
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_rusanov_flux( const Conservative_vector &left_state, const Conservative_vector &right_state, int dir,
    double gamma, Flux_vector &output, const EvaluatedState *left_flow, const EvaluatedState *right_flow )
{
    if ( dir != x_direction && dir != y_direction )
    {
        return INVALID_DIRECTION;
    }
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }
    ReturnCodes code = SUCCESS;
    EvaluatedState evaluated_left{}, evaluated_right{};
    if ( left_flow == nullptr )
    {
        code = calc_flow_variables( left_state, gamma, evaluated_left );
        if ( code != SUCCESS )
        {
            return code;
        }
    }
    if ( right_flow == nullptr )
    {
        code = calc_flow_variables( right_state, gamma, evaluated_right );
        if ( code != SUCCESS )
        {
            return code;
        }
    }
    const EvaluatedState &wl = left_flow == nullptr ? evaluated_left : *left_flow;
    const EvaluatedState &wr = right_flow == nullptr ? evaluated_right : *right_flow;
    const double un_l = ( dir == x_direction ) ? wl.x_velocity : wl.y_velocity;
    const double un_r = ( dir == x_direction ) ? wr.x_velocity : wr.y_velocity;
    const double smax = std::max( std::abs( un_l ) + wl.sound_speed, std::abs( un_r ) + wr.sound_speed );

    Flux_vector fl{};

    code = calc_physical_flux( left_state, wl, dir, fl );
    if ( code != SUCCESS )
    {
        return code;
    }

    Flux_vector fr{};

    code = calc_physical_flux( right_state, wr, dir, fr );
    if ( code != SUCCESS )
    {
        return code;
    }

    // Average the physical fluxes and subtract acoustic dissipation across the state jump.
    Flux_vector flux{};

    flux.mass_flux = 0.5 * ( fl.mass_flux + fr.mass_flux ) - ( 0.5 * smax ) * ( right_state.specific_mass - left_state.specific_mass );
    flux.x_momentum_flux = 0.5 * ( fl.x_momentum_flux + fr.x_momentum_flux ) - ( 0.5 * smax ) * ( right_state.specific_x_momentum - left_state.specific_x_momentum );
    flux.y_momentum_flux = 0.5 * ( fl.y_momentum_flux + fr.y_momentum_flux ) - ( 0.5 * smax ) * ( right_state.specific_y_momentum - left_state.specific_y_momentum );
    flux.energy_flux = 0.5 * ( fl.energy_flux + fr.energy_flux ) - ( 0.5 * smax ) * ( right_state.specific_energy - left_state.specific_energy );
    
    if ( !is_finite_flux( flux ) )
    {
        return NON_FINITE_FLUX;
    }

    output = flux;
    return SUCCESS;
}


// Compute a HLLC star state while preserving the tangential velocity.
//
// state (in) - conserved variables of the flow state
// primitive (in) - cached primitive variables, sound speed and enthalpy from the same conservative state
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// s (in) - estimated HLLC bounding-wave speed
// Sstar (in) - HLLC contact-wave speed
// output (out) - computed conservative state
// code (out) - operation status; SUCCESS indicates valid numerical outputs
static void calc_hllc_star_state( const Conservative_vector &state, const EvaluatedState &primitive, int dir,
    double gamma, double s, double Sstar, Conservative_vector &output, ReturnCodes &code )
{
    code = SUCCESS;

    const double rho = primitive.density;
    const double un = ( dir == x_direction ) ? primitive.x_velocity : primitive.y_velocity;
    const double ut = ( dir == x_direction ) ? primitive.y_velocity : primitive.x_velocity;
    const double p = primitive.pressure;
    const double e = state.specific_energy;

    if ( !std::isfinite( s - Sstar ) || s == Sstar )
    {
        code = INVALID_WAVE_SPEED;
        return;
    }
    const double rho_star = rho * ( s - un ) / ( s - Sstar );
    const double p_star = p + rho * ( s - un ) * ( Sstar - un );
    const double e_star = ( ( s - un ) * e - p * un + p_star * Sstar ) / ( s - Sstar );

    Conservative_vector us{};

    us.specific_mass = rho_star;

    if ( dir == x_direction )
    {
        us.specific_x_momentum = rho_star * Sstar;
        us.specific_y_momentum = rho_star * ut;
    }
    else
    {
        us.specific_x_momentum = rho_star * ut;
        us.specific_y_momentum = rho_star * Sstar;
    }

    us.specific_energy = e_star;

    if ( !std::isfinite( p_star ) || p_star <= 0.0 )
    {
        code = INVALID_STAR_STATE;
        return;
    }
    {
        EvaluatedState checked_state{};

        code = calc_flow_variables( us, gamma, checked_state );
        if ( code != SUCCESS )
        {
            return;
        }
    }

    output = us;
    code = SUCCESS;
    return;
}


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
    double gamma, Flux_vector &output )
{
    ReturnCodes code = SUCCESS;

    if ( dir != x_direction && dir != y_direction )
    {
        return INVALID_DIRECTION;
    }
    EvaluatedState wl{};

    code = calc_flow_variables( left_state, gamma, wl );
    if ( code != SUCCESS )
    {
        return code;
    }

    EvaluatedState wr{};

    code = calc_flow_variables( right_state, gamma, wr );
    if ( code != SUCCESS )
    {
        return code;
    }

    const double a_l = wl.sound_speed;
    const double a_r = wr.sound_speed;

    const double un_l = ( dir == x_direction ) ? wl.x_velocity : wl.y_velocity;
    const double un_r = ( dir == x_direction ) ? wr.x_velocity : wr.y_velocity;

    const double sl = std::min( un_l - a_l, un_r - a_r );
    const double sr = std::max( un_l + a_l, un_r + a_r );

    Flux_vector fl{}, fr{};

    if ( sl >= 0.0 )
    {
        code = calc_physical_flux( left_state, wl, dir, fl );
        if ( code != SUCCESS )
        {
            return code;
        }
        output = fl;
        return SUCCESS;
    }

    if ( sr <= 0.0 )
    {
        code = calc_physical_flux( right_state, wr, dir, fr );
        if ( code != SUCCESS )
        {
            return code;
        }
        output = fr;
        return SUCCESS;
    }

    const double rho_l = wl.density, rho_r = wr.density;
    const double p_l = wl.pressure, p_r = wr.pressure;

    // Use the momentum jump conditions to compute the contact-wave speed.
    const double num = p_r - p_l + rho_l * un_l * ( sl - un_l ) - rho_r * un_r * ( sr - un_r );
    const double den = rho_l * ( sl - un_l ) - rho_r * ( sr - un_r );

    if ( !std::isfinite( den ) || den == 0.0 )
    {
        return INVALID_WAVE_SPEED;
    }
    const double sm = num / den;

    if ( !std::isfinite( sm ) )
    {
        return INVALID_WAVE_SPEED;
    }

    Flux_vector flux{};

    if ( sm >= 0.0 )
    {
        code = calc_physical_flux( left_state, wl, dir, fl );
        if ( code != SUCCESS )
        {
            return code;
        }
        Conservative_vector uls{};
        
        calc_hllc_star_state( left_state, wl, dir, gamma, sl, sm, uls, code );
        
        if ( code != SUCCESS )
        {
            return code;
        }
        
        const Conservative_vector correction = sl * ( uls - left_state );

        flux.mass_flux = fl.mass_flux + correction.specific_mass;
        flux.x_momentum_flux = fl.x_momentum_flux + correction.specific_x_momentum;
        flux.y_momentum_flux = fl.y_momentum_flux + correction.specific_y_momentum;
        flux.energy_flux = fl.energy_flux + correction.specific_energy;
    }
    else
    {
        code = calc_physical_flux( right_state, wr, dir, fr );
        if ( code != SUCCESS )
        {
            return code;
        }
        Conservative_vector urs{};
        
        calc_hllc_star_state( right_state, wr, dir, gamma, sr, sm, urs, code );
        
        if ( code != SUCCESS )
        {
            return code;
        }
        const Conservative_vector correction = sr * ( urs - right_state );

        flux.mass_flux = fr.mass_flux + correction.specific_mass;
        flux.x_momentum_flux = fr.x_momentum_flux + correction.specific_x_momentum;
        flux.y_momentum_flux = fr.y_momentum_flux + correction.specific_y_momentum;
        flux.energy_flux = fr.energy_flux + correction.specific_energy;
    }

    if ( !is_finite_flux( flux ) )
    {
        return NON_FINITE_FLUX;
    }

    output = flux;
    return SUCCESS;
}


// Read rusanov or hllc; unknown names throw a configuration error.
//
// name (in) - requested numerical-method name
//
// Read the case-sensitive flux selection; unknown names are configuration errors.
FluxScheme read_flux_scheme( const std::string &name )
{
    if ( name == "rusanov" )
    {
        return FluxScheme::rusanov;
    }
    if ( name == "hllc" )
    {
        return FluxScheme::hllc;
    }
    throw std::runtime_error( "Unknown flux: " + name );
}


// Compute the flux vector at a discontinuity interface using the selected scheme.
//
// scheme (in) - selected numerical flux scheme
// left_state (in) - conserved variables to the left of the interface
// right_state (in) - conserved variables to the right of the interface
// dir (in) - flux calculation direction: x_direction or y_direction
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - flux vector of conserved variables in the selected direction
//
// Dispatch the selected calculation; output is assigned only on SUCCESS.
// Returns SUCCESS on completion or the numerical failure code; output values are unchanged on failure.
ReturnCodes calc_numerical_flux( FluxScheme scheme, const Conservative_vector &left_state,
    const Conservative_vector &right_state, int dir, double gamma, Flux_vector &output )
{
    switch ( scheme )
    {
    case FluxScheme::rusanov:
        return calc_rusanov_flux( left_state, right_state, dir, gamma, output );
    case FluxScheme::hllc:
        return calc_hllc_flux( left_state, right_state, dir, gamma, output );
    }
    return INVALID_FLUX_SCHEME;
}
