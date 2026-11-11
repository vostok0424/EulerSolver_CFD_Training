#pragma once


// Evaluate the constant-gamma ideal-gas closure; invalid inputs or results throw exceptions.
struct IdealGas
{
    // Compute pressure from positive internal energy density; gamma must be finite and greater than one.
    static double pressure_from_internal_energy_density( double internal_energy_density, double gamma );

    // Compute positive internal energy density from pressure; gamma must be finite and greater than one.
    static double internal_energy_density_from_pressure( double pressure, double gamma );

    // Compute sqrt(gamma * pressure / density) for finite positive density and pressure.
    static double sound_speed( double density, double pressure, double gamma );
};
