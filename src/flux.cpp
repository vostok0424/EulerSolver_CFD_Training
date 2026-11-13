#include "common_functions.hpp"
#include "flux.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>


// Store a normal Riemann sample; side_tag is -1 for left material and +1 for right material.
struct ExactSample1D
{
    double density{};
    double x_velocity{};
    double pressure{};
    int side_tag{};
};


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

    if ( sl >= 0.0 )
    {
        output = fl;
        return SUCCESS;
    }

    if ( sr <= 0.0 )
    {
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

    Conservative_vector uls{}, urs{};
    
    calc_hllc_star_state( left_state, wl, dir, gamma, sl, sm, uls, code );
    if ( code != SUCCESS )
    {
        return code;
    }
    calc_hllc_star_state( right_state, wr, dir, gamma, sr, sm, urs, code );
    if ( code != SUCCESS )
    {
        return code;
    }

    Flux_vector flux{};

    if ( sm >= 0.0 )
    {
        const Conservative_vector correction = sl * ( uls - left_state );

        flux.mass_flux = fl.mass_flux + correction.specific_mass;
        flux.x_momentum_flux = fl.x_momentum_flux + correction.specific_x_momentum;
        flux.y_momentum_flux = fl.y_momentum_flux + correction.specific_y_momentum;
        flux.energy_flux = fl.energy_flux + correction.specific_energy;
    }
    else
    {
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


// Compute the positive Mach-number split for AUSM.
//
// m (in) - signed normal Mach number
// output (out) - computed signed Mach or pressure splitting coefficient
static void ausm_mach_plus( double m, double &output )
{
    const double a_m = std::abs( m );

    if ( a_m >= 1.0 )
    {
        output = 0.5 * ( m + a_m );
        return;
    }

    output = 0.25 * ( m + 1.0 ) * ( m + 1.0 );
    return;
}


// Compute the negative Mach-number split for AUSM.
//
// m (in) - signed normal Mach number
// output (out) - computed signed Mach or pressure splitting coefficient
static void ausm_mach_minus( double m, double &output )
{
    const double a_m = std::abs( m );

    if ( a_m >= 1.0 )
    {
        output = 0.5 * ( m - a_m );
        return;
    }

    output = -0.25 * ( m - 1.0 ) * ( m - 1.0 );
    return;
}


// Compute the positive pressure weight for AUSM.
//
// m (in) - signed normal Mach number
// output (out) - computed signed Mach or pressure splitting coefficient
static void ausm_pressure_plus( double m, double &output )
{
    const double a_m = std::abs( m );

    if ( a_m >= 1.0 )
    {
        output = 0.5 * ( 1.0 + ( m >= 0.0 ? 1.0 : -1.0 ) );
        return;
    }

    output = 0.25 * ( m + 1.0 ) * ( m + 1.0 ) * ( 2.0 - m );
    return;
}


// Compute the negative pressure weight for AUSM.
//
// m (in) - signed normal Mach number
// output (out) - computed signed Mach or pressure splitting coefficient
static void ausm_pressure_minus( double m, double &output )
{
    const double a_m = std::abs( m );

    if ( a_m >= 1.0 )
    {
        output = 0.5 * ( 1.0 - ( m >= 0.0 ? 1.0 : -1.0 ) );
        return;
    }

    output = 0.25 * ( m - 1.0 ) * ( m - 1.0 ) * ( 2.0 + m );
    return;
}


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
    const double a = 0.5 * a_l + 0.5 * a_r;

    if ( !std::isfinite( a ) || a <= 0.0 )
    {
        return INVALID_SOUND_VELOCITY;
    }

    const double un_l = ( dir == x_direction ) ? wl.x_velocity : wl.y_velocity;
    const double un_r = ( dir == x_direction ) ? wr.x_velocity : wr.y_velocity;
    const double ml = un_l / a;
    const double mr = un_r / a;

    // Split the normal Mach numbers for mass transport and pressure weights for normal momentum.
    double mach_plus = 0.0, mach_minus = 0.0;
    double pressure_plus = 0.0, pressure_minus = 0.0;

    ausm_mach_plus( ml, mach_plus );
    ausm_mach_minus( mr, mach_minus );
    ausm_pressure_plus( ml, pressure_plus );
    ausm_pressure_minus( mr, pressure_minus );

    const double m_dot = a * ( mach_plus * wl.density + mach_minus * wr.density );
    const double p_int = pressure_plus * wl.pressure + pressure_minus * wr.pressure;

    // Transport velocity and total enthalpy from the side selected by the mass-flux sign.
    const bool up_l = ( m_dot >= 0.0 );
    const double u_up0 = up_l ? wl.x_velocity : wr.x_velocity;
    const double u_up1 = up_l ? wl.y_velocity : wr.y_velocity;
    const double hup = up_l ? wl.specific_total_enthalpy : wr.specific_total_enthalpy;

    Flux_vector flux{};

    flux.mass_flux = m_dot;

    if ( dir == x_direction )
    {
        flux.x_momentum_flux = m_dot * u_up0 + p_int;
        flux.y_momentum_flux = m_dot * u_up1;
    }
    else
    {
        flux.x_momentum_flux = m_dot * u_up0;
        flux.y_momentum_flux = m_dot * u_up1 + p_int;
    }

    flux.energy_flux = m_dot * hup;
    if ( !is_finite_flux( flux ) )
    {
        return NON_FINITE_FLUX;
    }

    output = flux;
    return SUCCESS;
}


// Compute the shock or rarefaction pressure function and its pressure derivative.
//
// p (in) - trial pressure for the normal Riemann pressure function
// primitive (in) - density, velocity and pressure of the flow state
// gamma (in) - ideal-gas heat-capacity ratio
// f (out) - pressure-function value for Newton iteration
// df (out) - pressure derivative of the pressure function
// code (out) - operation status; SUCCESS indicates valid numerical outputs
static void godunov_prefun_impl( double p, const Primitive_vector &primitive, double gamma, double &f, double &df,
    ReturnCodes &code )
{
    code = SUCCESS;

    double value = 0.0, derivative = 0.0;

    const double rho = primitive.density;
    const double p_k = primitive.pressure;
    double a_k{};

    code = calc_sound_velocity( Primitive_vector{ rho, primitive.x_velocity, 0.0, p_k }, gamma, a_k );
    if ( code != SUCCESS )
    {
        return;
    }

    // Use the shock relation above the side pressure and the rarefaction relation otherwise.
    if ( p > p_k )
    {
        const double a = 2.0 / ( ( gamma + 1.0 ) * rho );
        const double b = ( gamma - 1.0 ) / ( gamma + 1.0 ) * p_k;
        const double sqrt_term = std::sqrt( a / ( p + b ) );

        value = ( p - p_k ) * sqrt_term;
        derivative = sqrt_term * ( 1.0 - 0.5 * ( p - p_k ) / ( p + b ) );
    }
    else
    {
        const double pratio = p / p_k;
        const double expo = ( gamma - 1.0 ) / ( 2.0 * gamma );

        value = ( 2.0 * a_k / ( gamma - 1.0 ) ) * ( std::pow( pratio, expo ) - 1.0 );
        derivative = ( 1.0 / ( rho * a_k ) ) * std::pow( pratio, -( gamma + 1.0 ) / ( 2.0 * gamma ) );
    }
    if ( !std::isfinite( value ) || !std::isfinite( derivative ) || derivative <= 0.0 )
    {
        code = INVALID_NEWTON_ITERATION;
        return;
    }
    f = value;
    df = derivative;
    code = SUCCESS;
    return;
}


// Estimate the star pressure with the primitive-variable Riemann approximation.
//
// wl (in) - primitive state on the left of the normal Riemann problem
// wr (in) - primitive state on the right of the normal Riemann problem
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - computed pressure
// code (out) - operation status; SUCCESS indicates valid numerical outputs
static void guess_pressure_pvrsimpl( const Primitive_vector &wl, const Primitive_vector &wr, double gamma,
    double &output, ReturnCodes &code )
{
    code = SUCCESS;

    double a_l{};

    code = calc_sound_velocity( Primitive_vector{ wl.density, wl.x_velocity, 0.0, wl.pressure }, gamma, a_l );
    if ( code != SUCCESS )
    {
        return;
    }
    double a_r{};

    code = calc_sound_velocity( Primitive_vector{ wr.density, wr.x_velocity, 0.0, wr.pressure }, gamma, a_r );
    if ( code != SUCCESS )
    {
        return;
    }
    const double p_pv = 0.5 * ( wl.pressure + wr.pressure ) -
                        0.125 * ( wr.x_velocity - wl.x_velocity ) * ( wl.density + wr.density ) * ( a_l + a_r );
    if ( !std::isfinite( p_pv ) || p_pv <= 0.0 )
    {
        code = INVALID_PRESSURE_ESTIMATE;
        return;
    }
    output = p_pv;
    code = SUCCESS;
    return;
}


// Numerical controls for the exact Godunov pressure solve.
static constexpr int godunov_max_iterations = 40;
static constexpr double godunov_residual_relative_tolerance = 1e-10;


// Solve for the star pressure and normal velocity by Newton iteration.
//
// wl (in) - primitive state on the left of the normal Riemann problem
// wr (in) - primitive state on the right of the normal Riemann problem
// gamma (in) - ideal-gas heat-capacity ratio
// p_star (out) - pressure in the Riemann star region
// u_star (out) - normal velocity in the Riemann star region
// code (out) - operation status; SUCCESS indicates valid numerical outputs
static void calc_godunov_star_state( const Primitive_vector &wl, const Primitive_vector &wr, double gamma,
    double &p_star, double &u_star, ReturnCodes &code )
{
    code = SUCCESS;

    double a_l{};

    code = calc_sound_velocity( Primitive_vector{ wl.density, wl.x_velocity, 0.0, wl.pressure }, gamma, a_l );
    if ( code != SUCCESS )
    {
        return;
    }
    double a_r{};

    code = calc_sound_velocity( Primitive_vector{ wr.density, wr.x_velocity, 0.0, wr.pressure }, gamma, a_r );
    if ( code != SUCCESS )
    {
        return;
    }

    if ( ( wr.x_velocity - wl.x_velocity ) > ( 2.0 / ( gamma - 1.0 ) ) * ( a_l + a_r ) )
    {
        code = VACUUM_STATE;
        return;
    }

    double p = 0.0;
    guess_pressure_pvrsimpl( wl, wr, gamma, p, code );
    if ( code != SUCCESS )
    {
        return;
    }

    bool converged = false;

    // Solve f_left(p) + f_right(p) + u_right - u_left = 0; reject invalid Newton steps.
    for ( int it = 0; it < godunov_max_iterations; ++it )
    {
        double f_l, df_l, f_r, df_r;

        godunov_prefun_impl( p, wl, gamma, f_l, df_l, code );
        if ( code != SUCCESS )
        {
            return;
        }
        godunov_prefun_impl( p, wr, gamma, f_r, df_r, code );
        if ( code != SUCCESS )
        {
            return;
        }

        const double g = f_l + f_r + ( wr.x_velocity - wl.x_velocity );
        const double dg = df_l + df_r;

        if ( !std::isfinite( g ) || !std::isfinite( dg ) || dg <= 0.0 )
        {
            code = INVALID_NEWTON_ITERATION;
            return;
        }
        const double residual_scale = a_l + a_r + std::abs( wr.x_velocity - wl.x_velocity );

        if ( !std::isfinite( residual_scale ) || residual_scale <= 0.0 )
        {
            code = INVALID_NEWTON_ITERATION;
            return;
        }
        if ( std::abs( g ) <= godunov_residual_relative_tolerance * residual_scale )
        {
            converged = true;
            break;
        }

        const double p_new = p - g / dg;

        if ( !std::isfinite( p_new ) || p_new <= 0.0 )
        {
            code = INVALID_PRESSURE_ESTIMATE;
            return;
        }
        p = p_new;
    }

    if ( !converged )
    {
        code = PRESSURE_ITERATION_NOT_CONVERGED;
        return;
    }

    double f_l, df_l, f_r, df_r;

    godunov_prefun_impl( p, wl, gamma, f_l, df_l, code );
    if ( code != SUCCESS )
    {
        return;
    }
    godunov_prefun_impl( p, wr, gamma, f_r, df_r, code );
    if ( code != SUCCESS )
    {
        return;
    }

    const double velocity = 0.5 * ( wl.x_velocity + wr.x_velocity + f_r - f_l );
    if ( !std::isfinite( velocity ) )
    {
        code = INVALID_STAR_STATE;
        return;
    }
    p_star = p;
    u_star = velocity;
    code = SUCCESS;
    return;
}


// Sample the exact Riemann solution at the stationary face, x/t = 0.
//
// wl (in) - primitive state on the left of the normal Riemann problem
// wr (in) - primitive state on the right of the normal Riemann problem
// gamma (in) - ideal-gas heat-capacity ratio
// p_star (in) - pressure in the Riemann star region
// u_star (in) - normal velocity in the Riemann star region
// output (out) - normal Riemann sample at interface similarity coordinate zero
// code (out) - operation status; SUCCESS indicates valid numerical outputs
//
// The returned side_tag selects the upstream tangential velocity: -1 left, +1 right.
static void sample_at_s0( const Primitive_vector &wl, const Primitive_vector &wr, double gamma, double p_star,
    double u_star, ExactSample1D &output, ReturnCodes &code )
{
    code = SUCCESS;

    double a_l{};

    code = calc_sound_velocity( Primitive_vector{ wl.density, wl.x_velocity, 0.0, wl.pressure }, gamma, a_l );
    if ( code != SUCCESS )
    {
        return;
    }
    double a_r{};

    code = calc_sound_velocity( Primitive_vector{ wr.density, wr.x_velocity, 0.0, wr.pressure }, gamma, a_r );
    if ( code != SUCCESS )
    {
        return;
    }

    ExactSample1D out{};

    // The contact-wave sign selects the left or right wave pattern at x/t = 0.
    if ( 0.0 <= u_star )
    {
        out.side_tag = -1;

        if ( p_star > wl.pressure )
        {
            const double q_l = std::sqrt( 1.0 + ( gamma + 1.0 ) / ( 2.0 * gamma ) * ( p_star / wl.pressure - 1.0 ) );
            const double sl = wl.x_velocity - a_l * q_l;

            if ( 0.0 <= sl )
            {
                out = { wl.density, wl.x_velocity, wl.pressure, -1 };
            }
            else
            {
                const double pr = p_star / wl.pressure;
                const double rho_star = wl.density * ( pr + ( gamma - 1.0 ) / ( gamma + 1.0 ) ) /
                                        ( ( gamma - 1.0 ) / ( gamma + 1.0 ) * pr + 1.0 );
                out = { rho_star, u_star, p_star, -1 };
            }
        }
        else
        {
            const double shl = wl.x_velocity - a_l;
            const double a_star_l = a_l * std::pow( p_star / wl.pressure, ( gamma - 1.0 ) / ( 2.0 * gamma ) );
            const double stl = u_star - a_star_l;

            if ( 0.0 <= shl )
            {
                out = { wl.density, wl.x_velocity, wl.pressure, -1 };
            }
            else if ( 0.0 >= stl )
            {
                const double rho_star = wl.density * std::pow( p_star / wl.pressure, 1.0 / gamma );

                out = { rho_star, u_star, p_star, -1 };
            }
            else
            {
                const double s = 0.0;
                const double u = ( 2.0 / ( gamma + 1.0 ) ) * ( a_l + 0.5 * ( gamma - 1.0 ) * wl.x_velocity + s );
                const double a = ( 2.0 / ( gamma + 1.0 ) ) * ( a_l + 0.5 * ( gamma - 1.0 ) * ( wl.x_velocity - s ) );
                const double rho = wl.density * std::pow( a / a_l, 2.0 / ( gamma - 1.0 ) );
                const double p = wl.pressure * std::pow( a / a_l, 2.0 * gamma / ( gamma - 1.0 ) );

                out = { rho, u, p, -1 };
            }
        }
    }
    else
    {
        out.side_tag = +1;

        if ( p_star > wr.pressure )
        {
            const double q_r = std::sqrt( 1.0 + ( gamma + 1.0 ) / ( 2.0 * gamma ) * ( p_star / wr.pressure - 1.0 ) );
            const double sr = wr.x_velocity + a_r * q_r;

            if ( 0.0 >= sr )
            {
                out = { wr.density, wr.x_velocity, wr.pressure, +1 };
            }
            else
            {
                const double pr = p_star / wr.pressure;
                const double rho_star = wr.density * ( pr + ( gamma - 1.0 ) / ( gamma + 1.0 ) ) /
                                        ( ( gamma - 1.0 ) / ( gamma + 1.0 ) * pr + 1.0 );
                out = { rho_star, u_star, p_star, +1 };
            }
        }
        else
        {
            const double shr = wr.x_velocity + a_r;
            const double a_star_r = a_r * std::pow( p_star / wr.pressure, ( gamma - 1.0 ) / ( 2.0 * gamma ) );
            const double str = u_star + a_star_r;

            if ( 0.0 >= shr )
            {
                out = { wr.density, wr.x_velocity, wr.pressure, +1 };
            }
            else if ( 0.0 <= str )
            {
                const double rho_star = wr.density * std::pow( p_star / wr.pressure, 1.0 / gamma );

                out = { rho_star, u_star, p_star, +1 };
            }
            else
            {
                const double s = 0.0;
                const double u = ( 2.0 / ( gamma + 1.0 ) ) * ( -a_r + 0.5 * ( gamma - 1.0 ) * wr.x_velocity + s );
                const double a = ( 2.0 / ( gamma + 1.0 ) ) * ( a_r - 0.5 * ( gamma - 1.0 ) * ( wr.x_velocity - s ) );
                const double rho = wr.density * std::pow( a / a_r, 2.0 / ( gamma - 1.0 ) );
                const double p = wr.pressure * std::pow( a / a_r, 2.0 * gamma / ( gamma - 1.0 ) );

                out = { rho, u, p, +1 };
            }
        }
    }

    if ( !std::isfinite( out.density ) || !std::isfinite( out.x_velocity ) || !std::isfinite( out.pressure ) ||
         out.density <= 0.0 || out.pressure <= 0.0 )
    {
        code = INVALID_STAR_STATE;
        return;
    }
    output = out;
    code = SUCCESS;
    return;
}


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

    // Solve along the face normal with normal velocity stored in the 1D x_velocity field.
    Primitive_vector wl1{}, wr1{};

    wl1.density = wl.density;
    wl1.x_velocity = ( dir == x_direction ) ? wl.x_velocity : wl.y_velocity;
    wl1.pressure = wl.pressure;

    wr1.density = wr.density;
    wr1.x_velocity = ( dir == x_direction ) ? wr.x_velocity : wr.y_velocity;
    wr1.pressure = wr.pressure;

    double p_star = 0.0;
    double u_star = 0.0;

    calc_godunov_star_state( wl1, wr1, gamma, p_star, u_star, code );
    if ( code != SUCCESS )
    {
        return code;
    }

    ExactSample1D s0{};
    sample_at_s0( wl1, wr1, gamma, p_star, u_star, s0, code );
    if ( code != SUCCESS )
    {
        return code;
    }

    Primitive_vector w0{};

    w0.density = s0.density;
    w0.pressure = s0.pressure;
    w0.velocity( dir ) = s0.x_velocity;

    // Recover tangential velocity from the material side selected by the exact sample.
    const int tan = ( dir == x_direction ) ? y_direction : x_direction;

    w0.velocity( tan ) = ( s0.side_tag < 0 ) ? ( ( tan == x_direction ) ? wl.x_velocity : wl.y_velocity )
                                             : ( ( tan == x_direction ) ? wr.x_velocity : wr.y_velocity );

    Flux_vector checked_result{};

    code = calc_physical_flux( w0, dir, gamma, checked_result );
    if ( code != SUCCESS )
    {
        return code;
    }

    output = checked_result;
    return SUCCESS;
}


// Read rusanov, hllc, ausm or godunov; unknown names throw a configuration error.
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
    if ( name == "ausm" )
    {
        return FluxScheme::ausm;
    }
    if ( name == "godunov" )
    {
        return FluxScheme::godunov;
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
    case FluxScheme::ausm:
        return calc_ausm_flux( left_state, right_state, dir, gamma, output );
    case FluxScheme::godunov:
        return calc_godunov_flux( left_state, right_state, dir, gamma, output );
    }
    return INVALID_FLUX_SCHEME;
}
