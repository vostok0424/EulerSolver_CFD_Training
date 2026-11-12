#include "thermodynamics.hpp"
#include <cmath>
#include <stdexcept>

// Require a finite ideal-gas heat-capacity ratio greater than one.
//
// gamma (in) - ideal-gas heat-capacity ratio
static void require_gamma( double gamma )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
        throw std::runtime_error( "EOS: gamma must be finite and > 1" );
}


// Require a finite positive value; throw with the supplied message on failure.
//
// value (in) - numeric value required to be finite and strictly positive
// message (in) - reason included in the validation error
static void require_positive( double value, const char *message )
{
    if ( !std::isfinite( value ) || value <= 0.0 )
    {
        throw std::runtime_error( message );
    }
}


// Compute pressure from positive internal energy density; gamma must be finite and greater than one.
//
// energy (in) - internal energy per unit volume
// gamma (in) - ideal-gas heat-capacity ratio
//
// Compute pressure from internal energy per unit volume: p = (gamma - 1) * energy.
double IdealGas::pressure_from_internal_energy_density( double energy, double gamma )
{
    require_gamma( gamma );
    require_positive( energy, "EOS: internal energy density must be finite and > 0" );

    const double pressure = ( gamma - 1.0 ) * energy;

    require_positive( pressure, "EOS: pressure must be finite and > 0" );
    return pressure;
}


// Compute positive internal energy density from pressure; gamma must be finite and greater than one.
//
// pressure (in) - gas pressure
// gamma (in) - ideal-gas heat-capacity ratio
//
// Compute internal energy per unit volume: energy = p / (gamma - 1).
double IdealGas::internal_energy_density_from_pressure( double pressure, double gamma )
{
    require_gamma( gamma );
    require_positive( pressure, "EOS: pressure must be finite and > 0" );

    const double energy = pressure / ( gamma - 1.0 );

    require_positive( energy, "EOS: internal energy density must be finite and > 0" );
    return energy;
}


// Compute sqrt(gamma * pressure / density) for finite positive density and pressure.
//
// density (in) - gas density
// pressure (in) - gas pressure
// gamma (in) - ideal-gas heat-capacity ratio
//
// Compute the ideal-gas sound speed: c = sqrt(gamma * pressure / density).
double IdealGas::sound_speed( double density, double pressure, double gamma )
{
    require_gamma( gamma );
    require_positive( density, "EOS: density must be finite and > 0" );
    require_positive( pressure, "EOS: pressure must be finite and > 0" );

    const double c2 = gamma * pressure / density;

    require_positive( c2, "EOS: invalid sound-speed squared" );
    return std::sqrt( c2 );
}
