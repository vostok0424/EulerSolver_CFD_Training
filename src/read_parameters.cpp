#include "read_parameters.hpp"
#include "common_functions.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>


// File-local parsing data, separate from the caller's typed case parameters.
// Readers retain the first error; subsequent value readers leave their outputs unchanged.
// lines stores the last occurrence of each key so duplicate-key diagnostics match the effective value.
struct ParameterInput
{
    std::unordered_map<std::string, std::string> values;
    std::unordered_map<std::string, int> lines;
    std::string filename;
    std::string error;
    ReturnCodes code = SUCCESS;
};


// Internal helper declarations for the reading entry point below.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// code (in) - failure status to record if no earlier error exists
// key (in) - case-sensitive configuration key
// message (in) - reason included in the validation error
static void fail( ParameterInput &input, ReturnCodes code, const std::string &key, const std::string &message );

// Parse the whole file before typed reads; reject malformed nonempty lines and file I/O errors.
//
// filename (in) - explicit input case file path
// input (in/out) - file-local key/value data, source locations and first-error status
static void load_input( const std::string &filename, ParameterInput &input );

// Basic value readers.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_string( ParameterInput &input, const std::string &key, const std::string &fallback,
    std::string &value );

// Require full decimal integer conversion; malformed and out-of-range values leave the output unchanged.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_int( ParameterInput &input, const std::string &key, int fallback, int &value );

// Require a fully parsed, finite double; NaN, infinity and out-of-range values are errors.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_double( ParameterInput &input, const std::string &key, double fallback, double &value );

// Accept 1/true/yes/on and 0/false/no/off without regard to letter case.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_bool( ParameterInput &input, const std::string &key, bool fallback, bool &value );

// Report a missing required key instead of assigning a default.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_required_double( ParameterInput &input, const std::string &key, double &value );

// Reconstruction, output and initialization options.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (in) - parsed configuration value; unchanged if a previous or current read fails
static void read_scheme( ParameterInput &input, const std::string &key, const std::string &fallback,
    ReconstructionScheme &value );

// Map case-sensitive limiter names; an empty name selects vanLeer for compatibility.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_limiter( ParameterInput &input, const std::string &key, const std::string &fallback,
    SlopeLimiter &value );

// Accept point or cell without regard to letter case; empty values are errors.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_vtk( ParameterInput &input, const std::string &key, const std::string &fallback,
    VTKDataOutputType &value );

// Accept signed x/y directions without regard to letter case; an unsigned axis means positive.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_shock_dir( ParameterInput &input, const std::string &key, const std::string &fallback,
    ShockDir &value );

// Boundary options and required physical values.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_boundary( ParameterInput &input, const std::string &key, const std::string &fallback,
    BoundaryType &value );

// Use the side type over the global type. Any local inlet component selects the complete local set.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// side (in) - physical side name: left, right, bottom or top
// fallback (in) - value used only when the configuration key is absent
// boundary (out) - settings for one physical boundary side
static void read_boundary_side( ParameterInput &input, const std::string &side, const std::string &fallback,
    BoundarySide2D &boundary );


// Read all computational controls directly into the caller's parameter structure.
//
// filename (in) - explicit input case file path
// p (out) - directly filled computational experiment parameters; partial on failure
// error (out) - file, line when available, key and failure reason; empty on success
//
// Read all controls by key, independent of input order, directly into the caller's structure.
// Resolve background values before region defaults; optional region bounds are resolved by setFields.
// Only format and selected option checks occur here; solver/local-grid checks remain with Solver.
// Return immediately between dependent groups on failure; already assigned members are retained.
// Returns SUCCESS on completion; otherwise returns the first parameter error code.
// On failure, parameters may be partially assigned and must not be used; no rollback is performed.
ReturnCodes read_parameters( const std::string &filename, CaseParameters &p, std::string &error )
{
    ParameterInput input;
    load_input( filename, input );
    if ( input.code != SUCCESS )
    {
        error = input.error;
        return input.code;
    }

    // Retain dim only as a compatibility key; the solver is always two-dimensional.
    int dim = 2;
    read_int( input, "dim", 2, dim );
    if ( input.code == SUCCESS && dim != 2 )
    {
        fail( input, INVALID_PARAMETER_VALUE, "dim", "only two-dimensional cases (dim=2) are supported" );
    }
    if ( input.code != SUCCESS )
    {
        error = input.error;
        return input.code;
    }

    // Execution layout, global mesh and solver controls.
    read_int( input, "mpi.px", 1, p.mpi_px );
    read_int( input, "mpi.py", 1, p.mpi_py );
    read_int( input, "nx", 200, p.grid.nx );
    read_int( input, "ny", 50, p.grid.ny );
    read_int( input, "ng", 2, p.grid.ng );
    read_double( input, "x0", 0.0, p.grid.x0 );
    read_double( input, "x1", 1.0, p.grid.x1 );
    read_double( input, "y0", 0.0, p.grid.y0 );
    read_double( input, "y1", 0.2, p.grid.y1 );
    read_double( input, "gamma", 1.4, p.solver.gamma );
    read_double( input, "cfl", 0.5, p.solver.cfl );
    read_double( input, "finalTime", 0.2, p.solver.final_time );
    read_int( input, "outputEvery", 50, p.solver.output_every );
    read_double( input, "outputTimeInterval", 0.0, p.solver.output_time_interval );
    read_bool( input, "writeFinal", true, p.solver.write_final );
    read_string( input, "outPrefix", "run2d", p.solver.out_prefix );
    read_string( input, "flux", "hllc", p.solver.flux );
    read_string( input, "timeIntegrator", "ssprk3", p.solver.time_integrator );
    if ( input.code != SUCCESS )
    {
        error = input.error;
        return input.code;
    }

    // Numerical and output options, including legacy characteristic-only validation.
    read_vtk( input, "output.vtkDataOutputType", "point", p.solver.vtk_data_output_type );
    read_scheme( input, "reconstruction.scheme", "firstOrder", p.reconstruction.scheme );
    read_limiter( input, "reconstruction.limiter", "vanLeer", p.reconstruction.limiter );
    read_double( input, "reconstruction.eps", default_weno_epsilon, p.reconstruction.eps );
    
    if ( input.code != SUCCESS )
    {
        error = input.error;
        return input.code;
    }
    
    if ( p.reconstruction.eps <= 0.0 )
    {
        fail( input, INVALID_PARAMETER_VALUE, "reconstruction.eps", "must be > 0" );
    }
    
    std::string legacy;
    read_string( input, "reconstruction.variables", "", legacy );
    
    if ( !legacy.empty() && legacy != "characteristic" )
    {
        fail( input, INVALID_PARAMETER_VALUE, "reconstruction.variables", "only characteristic is supported" );
    }

    if ( input.code != SUCCESS )
    {
        error = input.error;
        return input.code;
    }

    // Share the gas model with all four physical-boundary descriptions.
    p.boundary.gamma = p.solver.gamma;
    std::string base_boundary;
    
    read_string( input, "bc", "zeroGradient", base_boundary );
    read_boundary_side( input, "left", base_boundary, p.boundary.left );
    read_boundary_side( input, "right", base_boundary, p.boundary.right );
    read_boundary_side( input, "bottom", base_boundary, p.boundary.bottom );
    read_boundary_side( input, "top", base_boundary, p.boundary.top );

    if ( input.code != SUCCESS )
    {
        error = input.error;
        return input.code;
    }

    // Background state supplies defaults for direct regions; shock velocities default to zero.
    auto &bg = p.set_fields.background;
    read_double( input, "setFields.bg.rho", 1.0, bg.density );
    read_double( input, "setFields.bg.u", 0.0, bg.x_velocity );
    read_double( input, "setFields.bg.v", 0.0, bg.y_velocity );
    read_double( input, "setFields.bg.p", 1.0, bg.pressure );
    
    int n_regions = 0;
    read_int( input, "setFields.nRegions", 0, n_regions );
    
    if ( input.code != SUCCESS )
    {
        error = input.error;
        return input.code;
    }
    // Replace regions from an earlier read and fill each new element in place, in index order.
    p.set_fields.regions.clear();
    
    for ( int i = 1; i <= n_regions && input.code == SUCCESS; ++i )
    {
        const std::string base = "setFields.region" + std::to_string( i ) + ".";
        p.set_fields.regions.emplace_back();
        auto &region = p.set_fields.regions.back();
        
        if ( input.values.count( base + "xMin" ) )
        {
            region.x_min.emplace();
            read_double( input, base + "xMin", 0.0, *region.x_min );
        }
        if ( input.values.count( base + "xMax" ) )
        {
            region.x_max.emplace();
            read_double( input, base + "xMax", 0.0, *region.x_max );
        }
        if ( input.values.count( base + "yMin" ) )
        {
            region.y_min.emplace();
            read_double( input, base + "yMin", 0.0, *region.y_min );
        }
        if ( input.values.count( base + "yMax" ) )
        {
            region.y_max.emplace();
            read_double( input, base + "yMax", 0.0, *region.y_max );
        }
        read_double( input, base + "shockMach", -1.0, region.shock_mach );
        read_double( input, base + "rho", bg.density, region.primitive.density );
        read_double( input, base + "p", bg.pressure, region.primitive.pressure );
        read_double( input, base + "u", region.shock_mach > 0.0 ? 0.0 : bg.x_velocity, region.primitive.x_velocity );
        read_double( input, base + "v", region.shock_mach > 0.0 ? 0.0 : bg.y_velocity, region.primitive.y_velocity );
        
        if ( region.shock_mach > 0.0 )
        {
            if ( region.shock_mach <= 1.0 )
            {
                fail( input, INVALID_PARAMETER_VALUE, base + "shockMach", "must be > 1" );
            }
            
            read_shock_dir( input, base + "shockDir", "+x", region.shock_dir );
        }
    }
    error = input.error;
    return input.code;
}


// Internal helper declarations for the reading entry point below.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// code (in) - failure status to record if no earlier error exists
// key (in) - case-sensitive configuration key
// message (in) - reason included in the validation error
//
// Retain the first failure and attach the effective key's source line when it exists.
static void fail( ParameterInput &input, ReturnCodes code, const std::string &key, const std::string &message )
{
    if ( input.code != SUCCESS )
        return;

    input.code = code;
    input.error = input.filename;
    const auto line = input.lines.find( key );
    
    if ( line != input.lines.end() )
    {
        input.error += ":" + std::to_string( line->second );
    }
    
    input.error += ": " + key + ": " + message;
}


// Parse the whole file before typed reads; reject malformed nonempty lines and file I/O errors.
//
// filename (in) - explicit input case file path
// input (in/out) - file-local key/value data, source locations and first-error status
static void load_input( const std::string &filename, ParameterInput &input )
{
    input.filename = filename;
    std::ifstream file( filename );
    if ( !file )
    {
        fail( input, PARAMETER_FILE_ERROR, "file", "cannot open file" );
        return;
    }

    std::string line;
    int number = 0;
    while ( std::getline( file, line ) )
    {
        ++number;
        line = trim_string( line.substr( 0, line.find( '#' ) ) );
        
        if ( line.empty() )
            continue;

        const auto equal = line.find( '=' );
        const std::string key = trim_string( line.substr( 0, equal ) );
        
        if ( equal == std::string::npos || key.empty() )
        {
            fail( input, PARAMETER_PARSE_ERROR, "line " + std::to_string( number ),
                equal == std::string::npos ? "missing '='" : "empty key" );
            return;
        }
        
        input.values[key] = trim_string( line.substr( equal + 1 ) );
        input.lines[key] = number;
    }
    
    if ( file.bad() )
    {
        fail( input, PARAMETER_FILE_ERROR, "file", "failed while reading" );
    }
}


// Basic value readers.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
//
// Use the fallback only for absent keys; an explicitly empty value remains empty.
static void read_string( ParameterInput &input, const std::string &key, const std::string &fallback,
    std::string &value )
{
    if ( input.code != SUCCESS )
        return;
    
    const auto it = input.values.find( key );
    value = it == input.values.end() ? fallback : it->second;
}


// Require full decimal integer conversion; malformed and out-of-range values leave the output unchanged.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_int( ParameterInput &input, const std::string &key, int fallback, int &value )
{
    if ( input.code != SUCCESS )
        return;
    const auto it = input.values.find( key );
    if ( it == input.values.end() )
    {
        value = fallback;
        return;
    }

    try
    {
        std::size_t used = 0;
        const int parsed = std::stoi( it->second, &used );
        if ( used != it->second.size() )
        {
            fail( input, INVALID_PARAMETER_VALUE, key, "invalid integer: " + it->second );
            return;
        }
        value = parsed;
    }
    catch ( const std::invalid_argument & )
    {
        fail( input, INVALID_PARAMETER_VALUE, key, "invalid integer: " + it->second );
    }
    catch ( const std::out_of_range & )
    {
        fail( input, INVALID_PARAMETER_VALUE, key, "out of range: " + it->second );
    }
}


// Require a fully parsed, finite double; NaN, infinity and out-of-range values are errors.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_double( ParameterInput &input, const std::string &key, double fallback, double &value )
{
    if ( input.code != SUCCESS )
        return;
    const auto it = input.values.find( key );
    if ( it == input.values.end() )
    {
        value = fallback;
        return;
    }

    try
    {
        std::size_t used = 0;
        const double parsed = std::stod( it->second, &used );
        
        if ( used != it->second.size() || !std::isfinite( parsed ) )
        {
            fail( input, INVALID_PARAMETER_VALUE, key, "invalid double: " + it->second );
            return;
        }
        value = parsed;
    }
    catch ( const std::invalid_argument & )
    {
        fail( input, INVALID_PARAMETER_VALUE, key, "invalid double: " + it->second );
    }
    catch ( const std::out_of_range & )
    {
        fail( input, INVALID_PARAMETER_VALUE, key, "out of range: " + it->second );
    }
}


// Accept 1/true/yes/on and 0/false/no/off without regard to letter case.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_bool( ParameterInput &input, const std::string &key, bool fallback, bool &value )
{
    if ( input.code != SUCCESS )
        return;
    std::string text;
    read_string( input, key, fallback ? "true" : "false", text );
    text = to_lowercase( text );
    if ( text == "1" || text == "true" || text == "yes" || text == "on" )
        value = true;
    else if ( text == "0" || text == "false" || text == "no" || text == "off" )
        value = false;
    else
        fail( input, INVALID_PARAMETER_VALUE, key, "invalid boolean: " + text );
}


// Report a missing required key instead of assigning a default.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_required_double( ParameterInput &input, const std::string &key, double &value )
{
    if ( input.values.find( key ) == input.values.end() )
    {
        fail( input, MISSING_PARAMETER, key, "required parameter is missing" );
    }
    
    read_double( input, key, 0.0, value );
}


// Reconstruction, output and initialization options.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (in) - parsed configuration value; unchanged if a previous or current read fails
//
// Map case-sensitive scheme names; an empty name selects firstOrder for compatibility.
static void read_scheme( ParameterInput &input, const std::string &key, const std::string &fallback,
    ReconstructionScheme &value )
{
    if ( input.code != SUCCESS )
        return;
    
    std::string text;
    read_string( input, key, fallback, text );
    
    if ( text.empty() )
        text = "firstOrder";
    if ( text == "firstOrder" )
        value = ReconstructionScheme::first_order;
    else if ( text == "muscl" )
        value = ReconstructionScheme::muscl;
    else if ( text == "weno5" )
        value = ReconstructionScheme::weno5;
    else
        fail( input, INVALID_PARAMETER_VALUE, key, "unsupported value: " + text );
}


// Map case-sensitive limiter names; an empty name selects vanLeer for compatibility.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_limiter( ParameterInput &input, const std::string &key, const std::string &fallback,
    SlopeLimiter &value )
{
    if ( input.code != SUCCESS )
        return;
    
    std::string text;
    read_string( input, key, fallback, text );
    
    if ( text.empty() )
        text = "vanLeer";
    if ( text == "vanLeer" )
        value = SlopeLimiter::van_leer;
    else if ( text == "none" )
        value = SlopeLimiter::none;
    else if ( text == "minmod" )
        value = SlopeLimiter::minmod;
    else
        fail( input, INVALID_PARAMETER_VALUE, key, "unsupported value: " + text );
}


// Accept point or cell without regard to letter case; empty values are errors.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_vtk( ParameterInput &input, const std::string &key, const std::string &fallback,
    VTKDataOutputType &value )
{
    if ( input.code != SUCCESS )
        return;
    
    std::string text;
    read_string( input, key, fallback, text );
    text = to_lowercase( text );
    
    if ( text == "point" )
        value = VTKDataOutputType::point;
    else if ( text == "cell" )
        value = VTKDataOutputType::cell;
    else
        fail( input, INVALID_PARAMETER_VALUE, key, "unsupported value: " + text );
}


// Accept signed x/y directions without regard to letter case; an unsigned axis means positive.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
static void read_shock_dir( ParameterInput &input, const std::string &key, const std::string &fallback,
    ShockDir &value )
{
    if ( input.code != SUCCESS )
        return;
    
    std::string text;
    read_string( input, key, fallback, text );
    text = to_lowercase( text );
    
    if ( text == "+x" )
        value = ShockDir::PlusX;
    else if ( text == "x" )
        value = ShockDir::PlusX;
    else if ( text == "-x" )
        value = ShockDir::MinusX;
    else if ( text == "+y" )
        value = ShockDir::PlusY;
    else if ( text == "y" )
        value = ShockDir::PlusY;
    else if ( text == "-y" )
        value = ShockDir::MinusY;
    else
        fail( input, INVALID_PARAMETER_VALUE, key, "unsupported value: " + text );
}


// Boundary options and required physical values.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// key (in) - case-sensitive configuration key
// fallback (in) - value used only when the configuration key is absent
// value (out) - parsed configuration value; unchanged if a previous or current read fails
//
// Map case-sensitive boundary names; empty names select zeroGradient. Periodic names are unsupported.
static void read_boundary( ParameterInput &input, const std::string &key, const std::string &fallback,
    BoundaryType &value )
{
    if ( input.code != SUCCESS )
        return;
    
    std::string text;
    read_string( input, key, fallback, text );
    
    if ( text.empty() )
        text = "zeroGradient";
    if ( text == "inlet" )
        value = BoundaryType::inlet;
    else if ( text == "outlet" )
        value = BoundaryType::outlet;
    else if ( text == "zeroGradient" )
        value = BoundaryType::zero_gradient;
    else if ( text == "slipWall" )
        value = BoundaryType::slip_wall;
    else if ( text == "symmetry" )
        value = BoundaryType::symmetry;
    else if ( text == "internal" )
        value = BoundaryType::internal;
    else
        fail( input, INVALID_PARAMETER_VALUE, key, "unsupported value: " + text );
}


// Use the side type over the global type. Any local inlet component selects the complete local set.
//
// input (in/out) - file-local key/value data, source locations and first-error status
// side (in) - physical side name: left, right, bottom or top
// fallback (in) - value used only when the configuration key is absent
// boundary (out) - settings for one physical boundary side
//
// Outlet pressure uses the side value when present, otherwise the global value, and must be positive.
static void read_boundary_side( ParameterInput &input, const std::string &side, const std::string &fallback,
    BoundarySide2D &boundary )
{
    read_boundary( input, "bc." + side, fallback, boundary.type );
    
    if ( input.code != SUCCESS )
        return;
    if ( boundary.type == BoundaryType::inlet )
    {
        std::string prefix = "bc.inlet.";
        const std::string local = "bc." + side + ".inlet.";
        
        for ( const auto *key : { "rho", "u", "v", "p" } )
        {
            if ( input.values.find( local + key ) != input.values.end() )
            {
                prefix = local;
            }
        }
        
        read_required_double( input, prefix + "rho", boundary.inlet.density );
        read_required_double( input, prefix + "u", boundary.inlet.x_velocity );
        read_required_double( input, prefix + "v", boundary.inlet.y_velocity );
        read_required_double( input, prefix + "p", boundary.inlet.pressure );
    }
    else if ( boundary.type == BoundaryType::outlet )
    {
        const std::string local = "bc." + side + ".outlet.p";
        const std::string key = input.values.find( local ) != input.values.end() ? local : "bc.outlet.p";
        
        read_required_double( input, key, boundary.pout );
        
        if ( boundary.pout <= 0.0 )
        {
            fail( input, INVALID_PARAMETER_VALUE, key, "outlet pressure must be > 0" );
        }
    }
}
