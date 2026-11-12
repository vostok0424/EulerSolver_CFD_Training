#pragma once


// Evaluate the constant-gamma ideal-gas closure; invalid inputs or results throw exceptions.
struct IdealGas
{
    // Compute pressure from positive internal energy density; gamma must be finite and greater than one.
    //
    // internal_energy_density (in) - internal energy per unit volume
    // gamma (in) - ideal-gas heat-capacity ratio
    static double pressure_from_internal_energy_density( double internal_energy_density, double gamma );

    // Compute positive internal energy density from pressure; gamma must be finite and greater than one.
    //
    // pressure (in) - gas pressure
    // gamma (in) - ideal-gas heat-capacity ratio
    static double internal_energy_density_from_pressure( double pressure, double gamma );

    // Compute sqrt(gamma * pressure / density) for finite positive density and pressure.
    //
    // density (in) - gas density
    // pressure (in) - gas pressure
    // gamma (in) - ideal-gas heat-capacity ratio
    static double sound_speed( double density, double pressure, double gamma );
};
