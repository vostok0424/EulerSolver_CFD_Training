#include "structures.hpp"
#include <stdexcept>


// Add conservative states componentwise.
//
// left (in) - left operand of the conservative-state operation
// right (in) - right operand of the conservative-state operation
//
// Componentwise state algebra; physical validity is checked by the consuming numerical routines.
Conservative_vector operator+( const Conservative_vector &left, const Conservative_vector &right )
{
    return { left.specific_mass + right.specific_mass, left.specific_x_momentum + right.specific_x_momentum,
        left.specific_y_momentum + right.specific_y_momentum, left.specific_energy + right.specific_energy };
}


// Subtract conservative states componentwise.
//
// left (in) - left operand of the conservative-state operation
// right (in) - right operand of the conservative-state operation
Conservative_vector operator-( const Conservative_vector &left, const Conservative_vector &right )
{
    return { left.specific_mass - right.specific_mass, left.specific_x_momentum - right.specific_x_momentum,
        left.specific_y_momentum - right.specific_y_momentum, left.specific_energy - right.specific_energy };
}


// Scale all conservative components by a scalar.
//
// factor (in) - scalar multiplier of all conservative components
// state (in) - conserved variables of the flow state
Conservative_vector operator*( double factor, const Conservative_vector &state )
{
    return { factor * state.specific_mass, factor * state.specific_x_momentum, factor * state.specific_y_momentum,
        factor * state.specific_energy };
}


// Divide all conservative components by a scalar.
//
// state (in) - conserved variables of the flow state
// divisor (in) - scalar divisor of all conservative components
Conservative_vector operator/( const Conservative_vector &state, double divisor )
{
    return { state.specific_mass / divisor, state.specific_x_momentum / divisor, state.specific_y_momentum / divisor,
        state.specific_energy / divisor };
}


// Accumulate a conservative increment into the supplied state.
//
// state (in/out) - conserved variables of the flow state
// increment (in) - componentwise conservative increment
Conservative_vector &operator+=( Conservative_vector &state, const Conservative_vector &increment )
{
    state = state + increment;
    return state;
}


// Store rho, rho*u, rho*v and rho*E in four consecutive doubles.
//
// state (in) - conserved variables of the flow state
// output (out) - buffer for four doubles ordered as density, x/y momentum and total energy
//
// Pack explicit components rather than object bytes so MPI does not depend on padding or layout.
void pack_state( const Conservative_vector &state, double *output )
{
    output[density_component] = state.specific_mass;
    output[x_momentum_component] = state.specific_x_momentum;
    output[y_momentum_component] = state.specific_y_momentum;
    output[total_energy_component] = state.specific_energy;
}


// Recover a conservative state from four doubles in rho, rho*u, rho*v, rho*E order.
//
// input (in) - four packed conservative components in density, x/y momentum, energy order
//
// Reconstruct the named state from the same serialization order used by pack_state.
Conservative_vector unpack_state( const double *input )
{
    return { input[density_component], input[x_momentum_component], input[y_momentum_component],
        input[total_energy_component] };
}


// Return the four conserved components in serialization order.
//
// state (in) - conserved variables of the flow state
//
// Expose the serialization order for componentwise numerical checks.
std::array<double, conservative_variable_count> state_components( const Conservative_vector &state )
{
    std::array<double, conservative_variable_count> components{};

    pack_state( state, components.data() );
    return components;
}


// Return flux components in mass, x-momentum, y-momentum and energy order.
//
// flux (in) - flux vector of conserved variables
//
// Use the same density/momentum/energy component ordering for flux inspection.
std::array<double, conservative_variable_count> flux_components( const Flux_vector &flux )
{
    return { flux.mass_flux, flux.x_momentum_flux, flux.y_momentum_flux, flux.energy_flux };
}


// Return the required ghost width: one for first order, two for MUSCL and three for WENO5.
//
// s (in) - selected reconstruction scheme
//
// Return stencil half-width requirements; reject an invalid scheme instead of selecting a fallback.
int required_ghost_cells( ReconstructionScheme s )
{
    switch ( s )
    {
    case ReconstructionScheme::first_order:
        return 1;
    case ReconstructionScheme::muscl:
        return 2;
    case ReconstructionScheme::weno5:
        return 3;
    }

    throw std::runtime_error( "Invalid reconstruction scheme enum" );
}


// Check whether the selected scheme is MUSCL or WENO5.
//
// s (in) - selected reconstruction scheme
//
// Identify the two schemes that require characteristic high-order reconstruction.
bool is_high_order( ReconstructionScheme s )
{
    return s == ReconstructionScheme::muscl || s == ReconstructionScheme::weno5;
}


// Access the velocity component selected by direction: 0 for x, 1 for y.
//
// direction (in) - coordinate direction: x_direction or y_direction
//
// Return a writable coordinate component; invalid directions raise an exception.
double &Primitive_vector::velocity( int direction )
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
//
// direction (in) - coordinate direction: x_direction or y_direction
//
// Read a coordinate component with the same direction checks as the writable overload.
double Primitive_vector::velocity( int direction ) const
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


// Uniform global spacing; the solver validates positive global counts and ordered bounds.
//
// Use global bounds and counts so every MPI block has the same cell spacing.
double GridBlock2DInfo::dx() const
{
    return ( x1 - x0 ) / static_cast<double>( nx_global );
}


// Same global spacing convention as dx(), independent of MPI decomposition.
//
// Compute y spacing using the global mesh; validation is the solver's responsibility.
double GridBlock2DInfo::dy() const
{
    return ( y1 - y0 ) / static_cast<double>( ny_global );
}


// Return the ghost width required by the configured scheme.
//
// Delegate stencil requirements to the common scheme query.
int ReconstructionOptions::required_ng() const
{
    return required_ghost_cells( scheme );
}
