#include "common_functions.hpp"
#include "thermodynamics.hpp"
#include <stdexcept>
#include <cctype>

// Check whether all values in the array are finite.
// values (in) - array with at least count entries
// count (in) - number of entries to check
bool all_values_finite( const double *values, std::size_t count )
{
    for ( std::size_t i = 0; i < count; ++i )
    {
        if ( !std::isfinite( values[i] ) )
        {
            return false;
        }
    }

    return true;
}


// Check whether all four conservative components are finite.
bool is_finite_state( const Conservative_vector &state )
{
    return all_values_finite( state_components( state ).data(), conservative_variable_count );
}


// Check whether all four flux components are finite.
bool is_finite_flux( const Flux_vector &flux )
{
    return all_values_finite( flux_components( flux ).data(), conservative_variable_count );
}


// Throw with the operation name and numeric return code when a calculation fails.
// code (in) - return code from the calculation
// operation (in) - operation name included in the exception
void require_success( ReturnCodes code, const char *operation )
{
    if ( code != SUCCESS )
    {
        throw std::runtime_error( std::string( operation ) + ": error code " +
                                  std::to_string( static_cast<int>( code ) ) );
    }
}


// Remove leading and trailing whitespace from a string.
std::string trim_string( std::string value )
{
    std::size_t first = 0;

    while ( first < value.size() && std::isspace( static_cast<unsigned char>( value[first] ) ) )
    {
        ++first;
    }

    std::size_t last = value.size();

    while ( last > first && std::isspace( static_cast<unsigned char>( value[last - 1] ) ) )
    {
        --last;
    }

    value = value.substr( first, last - first );
    return value;
}


// Convert a string to lowercase using the current character locale.
std::string to_lowercase( std::string value )
{
    for ( std::size_t i = 0; i < value.size(); ++i )
    {
        value[i] = static_cast<char>( std::tolower( static_cast<unsigned char>( value[i] ) ) );
    }

    return value;
}


// Check finite primitive variables and positive density and pressure.
static ReturnCodes check_primitive( const Primitive_vector &state )
{
    if ( !std::isfinite( state.density ) || !std::isfinite( state.x_velocity ) || !std::isfinite( state.y_velocity ) ||
         !std::isfinite( state.pressure ) )
    {
        return NON_FINITE_STATE;
    }

    if ( state.density <= 0.0 )
    {
        return NEGATIVE_DENSITY;
    }

    if ( state.pressure <= 0.0 )
    {
        return NEGATIVE_PRESSURE;
    }

    return SUCCESS;
}


// Compute density, velocity and pressure from the conserved variables.
// state (in) - conserved variables per unit volume
// gamma (in) - finite heat-capacity ratio greater than one
// output (out) - density, velocity and pressure; assigned only on SUCCESS
ReturnCodes calc_primitive_variables( const Conservative_vector &state, double gamma, Primitive_vector &output )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }

    if ( !all_values_finite( state_components( state ).data(), conservative_variable_count ) )
    {
        return NON_FINITE_STATE;
    }

    if ( state.specific_mass <= 0.0 )
    {
        return NEGATIVE_DENSITY;
    }

    Primitive_vector primitive{};

    primitive.density = state.specific_mass;
    primitive.x_velocity = state.specific_x_momentum / primitive.density;
    primitive.y_velocity = state.specific_y_momentum / primitive.density;

    double kinetic = 0.0;

    kinetic += 0.5 * primitive.density * primitive.x_velocity * primitive.x_velocity;
    kinetic += 0.5 * primitive.density * primitive.y_velocity * primitive.y_velocity;

    // Subtract kinetic energy before recovering ideal-gas pressure.
    const double internal_energy_density = state.specific_energy - kinetic;

    if ( !std::isfinite( internal_energy_density ) || internal_energy_density <= 0.0 )
    {
        return INVALID_INTERNAL_ENERGY;
    }

    try
    {
        primitive.pressure = IdealGas::pressure_from_internal_energy_density( internal_energy_density, gamma );
    }
    catch ( const std::runtime_error & )
    {
        return NEGATIVE_PRESSURE;
    }

    ReturnCodes code = check_primitive( primitive );
    if ( code != SUCCESS )
    {
        return code;
    }

    output = primitive;
    return SUCCESS;
}


// Compute the conserved variables from density, velocity and pressure.
// primitive (in) - density, x/y velocity and pressure
// gamma (in) - finite heat-capacity ratio greater than one
// output (out) - conserved variables per unit volume; assigned only on SUCCESS
ReturnCodes calc_conservative_variables( const Primitive_vector &primitive, double gamma, Conservative_vector &output )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }

    ReturnCodes code = check_primitive( primitive );
    if ( code != SUCCESS )
    {
        return code;
    }

    Conservative_vector state{};

    state.specific_mass = primitive.density;
    state.specific_x_momentum = primitive.density * primitive.x_velocity;
    state.specific_y_momentum = primitive.density * primitive.y_velocity;

    double kinetic = 0.0;

    kinetic += 0.5 * primitive.density * primitive.x_velocity * primitive.x_velocity;
    kinetic += 0.5 * primitive.density * primitive.y_velocity * primitive.y_velocity;

    try
    {
        state.specific_energy = IdealGas::internal_energy_density_from_pressure( primitive.pressure, gamma ) + kinetic;
    }
    catch ( const std::runtime_error & )
    {
        return INVALID_INTERNAL_ENERGY;
    }

    if ( !all_values_finite( state_components( state ).data(), conservative_variable_count ) )
    {
        return NON_FINITE_STATE;
    }

    output = state;
    return SUCCESS;
}


// Compute the ideal-gas speed of sound.
// primitive (in) - finite primitive state with positive density and pressure
// gamma (in) - finite heat-capacity ratio greater than one
// output (out) - speed of sound; assigned only on SUCCESS
ReturnCodes calc_sound_velocity( const Primitive_vector &primitive, double gamma, double &output )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }

    ReturnCodes code = check_primitive( primitive );
    if ( code != SUCCESS )
    {
        return code;
    }

    double sound_speed = 0.0;

    try
    {
        sound_speed = IdealGas::sound_speed( primitive.density, primitive.pressure, gamma );
    }
    catch ( const std::runtime_error & )
    {
        return INVALID_SOUND_VELOCITY;
    }

    output = sound_speed;
    return SUCCESS;
}


// Compute primitive variables, sound speed and specific total enthalpy.
// state (in) - conserved variables per unit volume
// gamma (in) - finite heat-capacity ratio greater than one
// output (out) - primitive variables, sound speed and enthalpy per unit mass; assigned only on SUCCESS
ReturnCodes calc_flow_variables( const Conservative_vector &state, double gamma, EvaluatedState &output )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }

    Primitive_vector primitive{};
    ReturnCodes code = calc_primitive_variables( state, gamma, primitive );
    if ( code != SUCCESS )
    {
        return code;
    }

    double sound_speed = 0.0;

    code = calc_sound_velocity( primitive, gamma, sound_speed );
    if ( code != SUCCESS )
    {
        return code;
    }

    EvaluatedState evaluated{};

    evaluated.density = primitive.density;
    evaluated.x_velocity = primitive.x_velocity;
    evaluated.y_velocity = primitive.y_velocity;
    evaluated.pressure = primitive.pressure;
    evaluated.sound_speed = sound_speed;
    // Total enthalpy per unit mass is (rhoE + p) / rho.
    evaluated.specific_total_enthalpy = ( state.specific_energy + primitive.pressure ) / primitive.density;

    if ( !std::isfinite( evaluated.specific_total_enthalpy ) )
    {
        return INVALID_ENTHALPY;
    }

    output = evaluated;
    return SUCCESS;
}


// Compute the squared velocity magnitude from the conserved variables.
// state (in) - finite conserved variables with positive density
// output (out) - x_velocity squared plus y_velocity squared; assigned only on SUCCESS
ReturnCodes velocity_squared( const Conservative_vector &state, double &output )
{
    if ( !all_values_finite( state_components( state ).data(), conservative_variable_count ) )
    {
        return NON_FINITE_STATE;
    }

    if ( state.specific_mass <= 0.0 )
    {
        return NEGATIVE_DENSITY;
    }

    const double u = state.specific_x_momentum / state.specific_mass;
    const double v = state.specific_y_momentum / state.specific_mass;
    const double speed2 = u * u + v * v;

    if ( !std::isfinite( speed2 ) )
    {
        return NON_FINITE_STATE;
    }

    output = speed2;
    return SUCCESS;
}


// Recover pressure from the conserved variables.
// state (in) - conserved variables per unit volume
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - pressure; assigned only on SUCCESS
ReturnCodes pressure_from_conservative( const Conservative_vector &state, double gamma, double &output )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }

    Primitive_vector primitive{};
    ReturnCodes code = calc_primitive_variables( state, gamma, primitive );
    if ( code != SUCCESS )
    {
        return code;
    }

    output = primitive.pressure;
    return SUCCESS;
}


// Compute internal energy per unit mass by subtracting kinetic energy.
// state (in) - conserved variables per unit volume
// output (out) - positive internal energy per unit mass; assigned only on SUCCESS
ReturnCodes specific_internal_energy_from_conservative( const Conservative_vector &state, double &output )
{
    double speed2 = 0.0;
    ReturnCodes code = velocity_squared( state, speed2 );
    if ( code != SUCCESS )
    {
        return code;
    }

    const double energy = state.specific_energy / state.specific_mass - 0.5 * speed2;

    if ( !std::isfinite( energy ) || energy <= 0.0 )
    {
        return INVALID_INTERNAL_ENERGY;
    }

    output = energy;
    return SUCCESS;
}


// Compute the Euler flux in the selected coordinate direction.
// state, primitive (in) - conserved state and cached flow variables from that same state
// dir (in) - flux direction: 0 for x, 1 for y
// output (out) - physical flux; assigned only on SUCCESS
ReturnCodes calc_physical_flux( const Conservative_vector &state, const EvaluatedState &primitive, int dir,
    Flux_vector &output )
{
    if ( dir != x_direction && dir != y_direction )
    {
        return INVALID_DIRECTION;
    }

    Flux_vector flux{};

    if ( !all_values_finite( state_components( state ).data(), conservative_variable_count ) )
    {
        return NON_FINITE_STATE;
    }

    if ( state.specific_mass <= 0.0 )
    {
        return NEGATIVE_DENSITY;
    }

    if ( !std::isfinite( primitive.density ) || !std::isfinite( primitive.x_velocity ) ||
         !std::isfinite( primitive.y_velocity ) || !std::isfinite( primitive.pressure ) ||
         !std::isfinite( primitive.sound_speed ) || !std::isfinite( primitive.specific_total_enthalpy ) )
    {
        return NON_FINITE_STATE;
    }

    if ( primitive.density <= 0.0 )
    {
        return NEGATIVE_DENSITY;
    }

    if ( primitive.pressure <= 0.0 )
    {
        return NEGATIVE_PRESSURE;
    }

    if ( primitive.sound_speed <= 0.0 )
    {
        return INVALID_SOUND_VELOCITY;
    }

    const double un = ( dir == x_direction ) ? primitive.x_velocity : primitive.y_velocity;

    flux.mass_flux = ( dir == x_direction ? state.specific_x_momentum : state.specific_y_momentum );

    if ( dir == x_direction )
    {
        flux.x_momentum_flux = state.specific_x_momentum * un + primitive.pressure;
        flux.y_momentum_flux = state.specific_y_momentum * un;
    }
    else
    {
        flux.x_momentum_flux = state.specific_x_momentum * un;
        flux.y_momentum_flux = state.specific_y_momentum * un + primitive.pressure;
    }

    // Pressure work contributes to the total-energy flux.
    flux.energy_flux = ( state.specific_energy + primitive.pressure ) * un;

    if ( !all_values_finite( flux_components( flux ).data(), conservative_variable_count ) )
    {
        return NON_FINITE_FLUX;
    }

    output = flux;
    return SUCCESS;
}


// Compute the Euler flux in the selected coordinate direction.
// primitive (in) - density, velocity and pressure
// dir (in) - flux direction: 0 for x, 1 for y
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - physical flux; assigned only on SUCCESS
ReturnCodes calc_physical_flux( const Primitive_vector &primitive, int dir, double gamma, Flux_vector &output )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }

    if ( dir != x_direction && dir != y_direction )
    {
        return INVALID_DIRECTION;
    }

    Flux_vector flux{};

    ReturnCodes code = check_primitive( primitive );
    if ( code != SUCCESS )
    {
        return code;
    }

    const double rho = primitive.density;
    const double u = primitive.x_velocity;
    const double v = primitive.y_velocity;
    const double p = primitive.pressure;
    double e = 0.0;

    try
    {
        e = IdealGas::internal_energy_density_from_pressure( p, gamma ) + 0.5 * rho * ( u * u + v * v );
    }
    catch ( const std::runtime_error & )
    {
        return INVALID_INTERNAL_ENERGY;
    }

    if ( dir == x_direction )
    {
        flux.mass_flux = rho * u;
        flux.x_momentum_flux = rho * u * u + p;
        flux.y_momentum_flux = rho * v * u;
        flux.energy_flux = ( e + p ) * u;
    }
    else
    {
        flux.mass_flux = rho * v;
        flux.x_momentum_flux = rho * u * v;
        flux.y_momentum_flux = rho * v * v + p;
        flux.energy_flux = ( e + p ) * v;
    }

    if ( !all_values_finite( flux_components( flux ).data(), conservative_variable_count ) )
    {
        return NON_FINITE_FLUX;
    }

    output = flux;
    return SUCCESS;
}


// Compute the Euler flux in the selected coordinate direction.
// state (in) - conserved variables per unit volume
// direction (in) - flux direction: 0 for x, 1 for y
// gamma (in) - ideal-gas heat-capacity ratio
// output (out) - physical flux; assigned only on SUCCESS
ReturnCodes calc_physical_flux( const Conservative_vector &state, int direction, double gamma, Flux_vector &output )
{
    if ( !std::isfinite( gamma ) || gamma <= 1.0 )
    {
        return INVALID_GAMMA;
    }

    EvaluatedState evaluated{};
    ReturnCodes code = calc_flow_variables( state, gamma, evaluated );
    if ( code != SUCCESS )
    {
        return code;
    }

    return calc_physical_flux( state, evaluated, direction, output );
}
