
#include "positive_preserving.hpp"
#include "common_functions.hpp"
#include <algorithm>
#include <cmath>
#include <limits>


// Evaluate pressure for detection, allowing negative finite pressure after density has been checked.
// state (in) - finite conservative state with strictly positive density
// gamma (in) - validated ideal-gas ratio
// pressure (out) - signed pressure; unchanged on failure
static ReturnCodes detection_pressure( const Conservative_vector &state, double gamma, double &pressure )
{
    if ( !is_finite_state( state ) )
    {
        return NON_FINITE_STATE;
    }
    if ( state.specific_mass <= 0.0 )
    {
        return NEGATIVE_DENSITY;
    }
    const double kinetic = 0.5 * ( state.specific_x_momentum / state.specific_mass ) * state.specific_x_momentum +
        0.5 * ( state.specific_y_momentum / state.specific_mass ) * state.specific_y_momentum;
    
    const double result = ( gamma - 1.0 ) * ( state.specific_energy - kinetic );
    
    if ( !std::isfinite( result ) )
    {
        return NON_FINITE_STATE;
    }
    
    pressure = result;
    
    return SUCCESS;
}


// Form the two auxiliary states associated with one oriented face.
// left, right (in) - stage cell averages
// flux (in) - shared face flux
// factor (in) - multidimensional auxiliary update factor
// auxiliary (out) - left outgoing and right incoming auxiliary states
static void auxiliary_states( const Conservative_vector &left, const Conservative_vector &right,
    const Flux_vector &flux, double factor, Conservative_vector ( &auxiliary )[2] )
{
    const Conservative_vector increment{ factor * flux.mass_flux, factor * flux.x_momentum_flux, factor * flux.y_momentum_flux, factor * flux.energy_flux };
    
    auxiliary[0] = left - increment;
    auxiliary[1] = right + increment;
}


// Blend fluxes without changing either endpoint when theta is exactly zero or one.
// low, high (in) - finite baseline and candidate fluxes
// theta (in) - convex combination coefficient
// Returns the common limited face flux.
static Flux_vector blend_flux( const Flux_vector &low, const Flux_vector &high, double theta )
{
    if ( theta == 0.0 )
    {
        return low;
    }
    if ( theta == 1.0 )
    {
        return high;
    }
    return
    {
        ( 1.0 - theta ) * low.mass_flux + theta * high.mass_flux,
        ( 1.0 - theta ) * low.x_momentum_flux + theta * high.x_momentum_flux,
        ( 1.0 - theta ) * low.y_momentum_flux + theta * high.y_momentum_flux,
        ( 1.0 - theta ) * low.energy_flux + theta * high.energy_flux
    };
}


// Compute the scalar chord cut-off, biased toward the admissible baseline when limiting is needed.
// baseline (in) - admissible density or pressure of the low-order auxiliary state
// candidate (in) - finite candidate density or pressure
// floor (in) - required positive lower bound
// Returns one for an admissible candidate, otherwise a coefficient in [0,1].
static double cutoff_factor( double baseline, double candidate, double floor )
{
    if ( candidate >= floor )
    {
        return 1.0;
    }
    // Scale the ratio to avoid overflow when baseline and candidate have large opposite signs.
    const double scale = std::max( std::abs( baseline ), std::abs( candidate ) );
    const double ratio = ( baseline / scale - floor / scale ) / ( baseline / scale - candidate / scale );
    const double margin = 1.0 - 64.0 * std::numeric_limits<double>::epsilon();
    
    const double factor = ratio * margin;
    if ( factor < 0.0 )
    {
        return 0.0;
    }
    if ( factor > 1.0 )
    {
        return 1.0;
    }
    return factor;
}


// Interface and admissibility contract are documented in positive_preserving.hpp.
ReturnCodes limit_positive_flux( const Conservative_vector &left_average,
    const Conservative_vector &right_average, const Flux_vector &low_order_flux,
    const Flux_vector &high_order_flux, double update_factor, double gamma,
    double density_floor, double pressure_floor, Flux_vector &limited_flux, double &theta )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }
    if ( !std::isfinite( update_factor ) || update_factor < 0.0 ||
         !std::isfinite( density_floor ) || density_floor <= 0.0 ||
         !std::isfinite( pressure_floor ) || pressure_floor <= 0.0 )
    {
        return INVALID_PARAMETER_VALUE;
    }
    if ( !is_finite_flux( low_order_flux ) || !is_finite_flux( high_order_flux ) )
    {
        return NON_FINITE_FLUX;
    }
    for ( const auto *state : { &left_average, &right_average } )
    {
        double pressure = 0.0;
        const ReturnCodes code = detection_pressure( *state, gamma, pressure );
        if ( code != SUCCESS )
        {
            return code;
        }
        if ( state->specific_mass < density_floor )
        {
            return NEGATIVE_DENSITY;
        }
        if ( pressure < pressure_floor )
        {
            return NEGATIVE_PRESSURE;
        }
    }

    Conservative_vector baseline[2], candidate[2];
    
    auxiliary_states( left_average, right_average, low_order_flux, update_factor, baseline );
    auxiliary_states( left_average, right_average, high_order_flux, update_factor, candidate );
    
    double baseline_pressure[2];
    double density_theta = 1.0;
   
    for ( int side = 0; side < 2; ++side )
    {
        const ReturnCodes code = detection_pressure( baseline[side], gamma, baseline_pressure[side] );
        
        if ( code != SUCCESS )
        {
            return code;
        }
        if ( baseline[side].specific_mass < density_floor )
        {
            return NEGATIVE_DENSITY;
        }
        if ( baseline_pressure[side] < pressure_floor )
        {
            return NEGATIVE_PRESSURE;
        }
        if ( !is_finite_state( candidate[side] ) )
        {
            return NON_FINITE_STATE;
        }
        
        density_theta = std::min( density_theta, cutoff_factor( baseline[side].specific_mass, candidate[side].specific_mass, density_floor ) );
    }

    const Flux_vector density_flux = blend_flux( low_order_flux, high_order_flux, density_theta );
    
    auxiliary_states( left_average, right_average, density_flux, update_factor, candidate );
    
    double pressure_theta = 1.0;
    
    for ( int side = 0; side < 2; ++side )
    {
        double pressure = 0.0;
        const ReturnCodes code = detection_pressure( candidate[side], gamma, pressure );
        
        if ( code != SUCCESS )
        {
            return code;
        }
        
        pressure_theta = std::min( pressure_theta, cutoff_factor( baseline_pressure[side], pressure, pressure_floor ) );
    }

    const double final_theta = density_theta * pressure_theta;
    const Flux_vector result = blend_flux( low_order_flux, high_order_flux, final_theta );
    
    auxiliary_states( left_average, right_average, result, update_factor, candidate );
    
    // Recheck the actual floating-point result; never claim success solely from the analytical bound.
    for ( const auto &state : candidate )
    {
        double pressure = 0.0;
        const ReturnCodes code = detection_pressure( state, gamma, pressure );
        
        if ( code != SUCCESS )
        {
            return code;
        }
        if ( state.specific_mass < density_floor )
        {
            return NEGATIVE_DENSITY;
        }
        if ( pressure < pressure_floor )
        {
            return NEGATIVE_PRESSURE;
        }
    }
    
    limited_flux = result;
    theta = final_theta;
    return SUCCESS;
}


// Protect Riemann inputs before the flux limiter can be applied.
ReturnCodes limit_positive_point( const Conservative_vector &average, double gamma,
    double density_floor, double pressure_floor, Conservative_vector &point )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }
    if ( !std::isfinite( density_floor ) || density_floor <= 0.0 ||
         !std::isfinite( pressure_floor ) || pressure_floor <= 0.0 )
    {
        return INVALID_PARAMETER_VALUE;
    }
    double pressure = 0.0;
    ReturnCodes code = detection_pressure( average, gamma, pressure );
    if ( code != SUCCESS )
    {
        return code;
    }
    if ( average.specific_mass < density_floor )
    {
        return NEGATIVE_DENSITY;
    }
    if ( pressure < pressure_floor )
    {
        return NEGATIVE_PRESSURE;
    }
    if ( !is_finite_state( point ) )
    {
        return NON_FINITE_STATE;
    }

    const double theta = cutoff_factor( average.specific_mass, point.specific_mass, density_floor );
    Conservative_vector candidate = theta == 1.0 ? point :
        ( 1.0 - theta ) * average + theta * point;
    code = detection_pressure( candidate, gamma, pressure );
    if ( code == SUCCESS && candidate.specific_mass >= density_floor && pressure >= pressure_floor )
    {
        point = candidate;
        return SUCCESS;
    }
    // The admissible set is convex. Keep a verified safe endpoint while bisecting.
    double low = 0.0, high = 1.0;
    Conservative_vector safe = average;
    for ( int iteration = 0; iteration < 60; ++iteration )
    {
        const double middle = 0.5 * ( low + high );
        const Conservative_vector trial = ( 1.0 - middle ) * average + middle * candidate;
        code = detection_pressure( trial, gamma, pressure );
        if ( code == SUCCESS && trial.specific_mass >= density_floor && pressure >= pressure_floor )
        {
            low = middle;
            safe = trial;
        }
        else
        {
            high = middle;
        }
    }
    point = safe;
    return SUCCESS;
}
