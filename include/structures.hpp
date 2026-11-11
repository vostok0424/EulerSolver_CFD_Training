#pragma once

#include "constants.hpp"
#include <array>
#include <stdexcept>


// Store rho, rho*u, rho*v and rho*E. The specific_* names denote conserved quantities per unit volume.
struct Conservative_vector
{
    double specific_mass{};
    double specific_x_momentum{};
    double specific_y_momentum{};
    double specific_energy{};
};


// Store density, x/y velocity and pressure.
struct Primitive_vector
{
    double density{};
    double x_velocity{};
    double y_velocity{};
    double pressure{};

    // Access the velocity component selected by direction: 0 for x, 1 for y.
    double &velocity( int direction )
    {
        if ( direction == x_direction )
        {
            return x_velocity;
        }

        if ( direction == y_direction )
        {
            return y_velocity;
        }

        throw std::runtime_error( "Invalid velocity direction" );
    }

    // Access the velocity component selected by direction: 0 for x, 1 for y.
    double velocity( int direction ) const
    {
        if ( direction == x_direction )
        {
            return x_velocity;
        }

        if ( direction == y_direction )
        {
            return y_velocity;
        }

        throw std::runtime_error( "Invalid velocity direction" );
    }
};


// Store mass, x/y momentum and energy fluxes in the direction selected by the flux calculation.
struct Flux_vector
{
    double mass_flux{};
    double x_momentum_flux{};
    double y_momentum_flux{};
    double energy_flux{};
};


// Store primitive variables, sound speed and total enthalpy per unit mass for one evaluated state.
struct EvaluatedState
{
    double density{};
    double x_velocity{};
    double y_velocity{};
    double pressure{};
    double sound_speed{};
    double specific_total_enthalpy{};
};


inline Conservative_vector operator+( const Conservative_vector &left, const Conservative_vector &right )
{
    return { left.specific_mass + right.specific_mass, left.specific_x_momentum + right.specific_x_momentum,
        left.specific_y_momentum + right.specific_y_momentum, left.specific_energy + right.specific_energy };
}


inline Conservative_vector operator-( const Conservative_vector &left, const Conservative_vector &right )
{
    return { left.specific_mass - right.specific_mass, left.specific_x_momentum - right.specific_x_momentum,
        left.specific_y_momentum - right.specific_y_momentum, left.specific_energy - right.specific_energy };
}


inline Conservative_vector operator*( double factor, const Conservative_vector &state )
{
    return { factor * state.specific_mass, factor * state.specific_x_momentum, factor * state.specific_y_momentum,
        factor * state.specific_energy };
}


inline Conservative_vector operator/( const Conservative_vector &state, double divisor )
{
    return { state.specific_mass / divisor, state.specific_x_momentum / divisor, state.specific_y_momentum / divisor,
        state.specific_energy / divisor };
}


inline Conservative_vector &operator+=( Conservative_vector &state, const Conservative_vector &increment )
{
    state = state + increment;
    return state;
}


// Store rho, rho*u, rho*v and rho*E in four consecutive doubles.
// output (out) - buffer with room for four doubles; structure padding is not serialized.
inline void pack_state( const Conservative_vector &state, double *output )
{
    output[density_component] = state.specific_mass;
    output[x_momentum_component] = state.specific_x_momentum;
    output[y_momentum_component] = state.specific_y_momentum;
    output[total_energy_component] = state.specific_energy;
}


// Recover a conservative state from four doubles in rho, rho*u, rho*v, rho*E order.
// input (in) - buffer containing at least four doubles.
inline Conservative_vector unpack_state( const double *input )
{
    return { input[density_component], input[x_momentum_component], input[y_momentum_component],
        input[total_energy_component] };
}


// Return the four conserved components in serialization order.
inline std::array<double, conservative_variable_count> state_components( const Conservative_vector &state )
{
    std::array<double, conservative_variable_count> components{};

    pack_state( state, components.data() );
    return components;
}


// Return flux components in mass, x-momentum, y-momentum and energy order.
inline std::array<double, conservative_variable_count> flux_components( const Flux_vector &flux )
{
    return { flux.mass_flux, flux.x_momentum_flux, flux.y_momentum_flux, flux.energy_flux };
}


// Store global mesh bounds and this rank's interior and ghosted cell counts.
struct GridBlock2DInfo
{
    int nx_global{};
    int ny_global{};

    int i_beg{};
    int j_beg{};
    int nx{};
    int ny{};

    int ng{};
    int nx_tot{};
    int ny_tot{};

    double x0{};
    double x1{};
    double y0{};
    double y1{};

    double dx() const
    {
        return ( x1 - x0 ) / static_cast<double>( nx_global );
    }

    double dy() const
    {
        return ( y1 - y0 ) / static_cast<double>( ny_global );
    }
};
