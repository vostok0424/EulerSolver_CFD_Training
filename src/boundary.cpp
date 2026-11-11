#include "common_functions.hpp"

#include "boundary.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>


// Trim a boundary name; use zeroGradient when the name is empty.
static inline std::string normalize_name( const std::string &s )
{
    const std::string v = trim_string( s );

    if ( v.empty() )
    {
        return "zeroGradient";
    }

    return v;
}


// Reject periodic and cyclic boundary names.
static inline void reject_if_periodic( const std::string &n )
{
    const std::string v = normalize_name( n );

    if ( v == "periodic" || v == "cyclic" )
    {
        throw std::runtime_error( "boundary: periodic/cyclic is not supported in this training code." );
    }
}


// Convert a case-sensitive boundary name to its boundary type.
static inline BoundaryType parse_bc_type( const std::string &n, const std::string &where )
{
    const std::string v = normalize_name( n );

    reject_if_periodic( v );

    if ( v == "inlet" )
    {
        return BoundaryType::inlet;
    }

    if ( v == "outlet" )
    {
        return BoundaryType::outlet;
    }

    if ( v == "zeroGradient" )
    {
        return BoundaryType::zero_gradient;
    }

    if ( v == "slipWall" )
    {
        return BoundaryType::slip_wall;
    }

    if ( v == "symmetry" )
    {
        return BoundaryType::symmetry;
    }

    if ( v == "internal" )
    {
        return BoundaryType::internal;
    }

    throw std::runtime_error( "boundary: unsupported BC name in " + where + ": " + n );
}


// Check whether the boundary reflects normal momentum.
static inline bool is_wall_like( BoundaryType t )
{
    return ( t == BoundaryType::slip_wall || t == BoundaryType::symmetry );
}


// Read a side-specific boundary name, or use the supplied fallback.
static inline std::string pick_name( const Cfg &cfg, const std::string &key, const std::string &fallback )
{
    if ( cfg.has( key ) )
    {
        return normalize_name( cfg.get_string( key, "" ) );
    }

    return normalize_name( fallback );
}


// Throw an exception when a required configuration key is absent.
static inline void require_key( const Cfg &cfg, const std::string &key )
{
    if ( !cfg.has( key ) )
    {
        throw std::runtime_error( "boundary: missing required cfg key: " + key );
    }
}


// Read a required floating-point configuration value.
static inline double read_double_strict( const Cfg &cfg, const std::string &key )
{
    require_key( cfg, key );
    return cfg.get_double( key, 0.0 );
}


// Check whether any listed key exists under the supplied prefix.
static inline bool has_any_prefix( const Cfg &cfg, const std::string &prefix,
    const std::initializer_list<std::string> &keys )
{
    for ( const auto &k : keys )
    {
        if ( cfg.has( prefix + k ) )
        {
            return true;
        }
    }

    return false;
}


// Read inlet rho, u, v and p; any side-specific inlet key selects the complete side-specific set.
static Primitive_vector read_inlet_prim2( const Cfg &cfg, const std::string &side )
{
    const std::string p1 = "bc." + side + ".inlet.";
    const std::string p0 = "bc.inlet.";

    const bool side_has = has_any_prefix( cfg, p1, { "rho", "u", "v", "p" } );
    const std::string p = side_has ? p1 : p0;

    Primitive_vector primitive{};

    primitive.density = read_double_strict( cfg, p + "rho" );
    primitive.x_velocity = read_double_strict( cfg, p + "u" );
    primitive.y_velocity = read_double_strict( cfg, p + "v" );
    primitive.pressure = read_double_strict( cfg, p + "p" );
    return primitive;
}


// Read a positive outlet pressure, preferring bc.<side>.outlet.p over bc.outlet.p.
static double read_outlet_p( const Cfg &cfg, const std::string &side )
{
    const std::string k1 = "bc." + side + ".outlet.p";
    const std::string k0 = "bc.outlet.p";
    const double p = read_double_strict( cfg, cfg.has( k1 ) ? k1 : k0 );

    if ( p <= 0.0 )
    {
        throw std::runtime_error( "boundary: outlet pressure must be > 0" );
    }

    return p;
}


// Read boundary types and their required inlet or outlet parameters.
BoundaryConditions2D read_boundary_conditions( const Cfg &cfg )
{
    const std::string base = normalize_name( cfg.get_string( "bc", "zeroGradient" ) );

    BoundaryConditions2D b;

    b.gamma = cfg.get_double( "gamma", 1.4 );

    const std::string left_name = pick_name( cfg, "bc.left", base );
    const std::string right_name = pick_name( cfg, "bc.right", base );
    const std::string bottom_name = pick_name( cfg, "bc.bottom", base );
    const std::string top_name = pick_name( cfg, "bc.top", base );

    b.left.type = parse_bc_type( left_name, "read2D.left" );
    b.right.type = parse_bc_type( right_name, "read2D.right" );
    b.bottom.type = parse_bc_type( bottom_name, "read2D.bottom" );
    b.top.type = parse_bc_type( top_name, "read2D.top" );

    if ( b.left.type == BoundaryType::inlet )
    {
        b.left.inlet = read_inlet_prim2( cfg, "left" );
    }
    else if ( b.left.type == BoundaryType::outlet )
    {
        b.left.pout = read_outlet_p( cfg, "left" );
    }

    if ( b.right.type == BoundaryType::inlet )
    {
        b.right.inlet = read_inlet_prim2( cfg, "right" );
    }
    else if ( b.right.type == BoundaryType::outlet )
    {
        b.right.pout = read_outlet_p( cfg, "right" );
    }

    if ( b.bottom.type == BoundaryType::inlet )
    {
        b.bottom.inlet = read_inlet_prim2( cfg, "bottom" );
    }
    else if ( b.bottom.type == BoundaryType::outlet )
    {
        b.bottom.pout = read_outlet_p( cfg, "bottom" );
    }

    if ( b.top.type == BoundaryType::inlet )
    {
        b.top.inlet = read_inlet_prim2( cfg, "top" );
    }
    else if ( b.top.type == BoundaryType::outlet )
    {
        b.top.pout = read_outlet_p( cfg, "top" );
    }

    return b;
}


// Extrapolate a supersonic state; impose outlet pressure when the normal Mach magnitude is below one.
// uin (in) - adjacent interior state
// pout (in) - prescribed subsonic outlet pressure
// normal_dir (in) - boundary-normal axis: 0 for x, 1 for y
// gamma (in) - ideal-gas heat-capacity ratio
static inline Conservative_vector outlet_state_2d( const Conservative_vector &uin, double pout, int normal_dir,
    double gamma )
{
    Primitive_vector primitive{};

    require_success( calc_primitive_variables( uin, gamma, primitive ), "calc_primitive_variables" );

    double a{};

    require_success( calc_sound_velocity( primitive, gamma, a ), "calc_sound_velocity" );

    const double un = primitive.velocity( normal_dir );
    const double m = std::abs( un ) / a;

    if ( m >= 1.0 )
    {
        return uin;
    }

    Primitive_vector wo = primitive;

    wo.pressure = pout;

    Conservative_vector checked_result{};

    require_success( calc_conservative_variables( wo, gamma, checked_result ), "calc_conservative_variables" );
    return checked_result;
}


// Fill physical-boundary ghost cells; leave interior cells and internal MPI boundaries unchanged.
// state (in/out) - row-major conservative field including ghost cells
// nx, ny, ng (in) - interior sizes and ghost width
// bc (in) - per-side boundary types, inlet states, outlet pressures and gamma
void apply_boundary_conditions( std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    const BoundaryConditions2D &bc )
{
    const double gamma = bc.gamma;
    const int nx_tot = nx + 2 * ng;
    auto idx = [&]( int i, int j )
    {
        return i + nx_tot * j;
    };

    // Fill left/right ghosts for interior rows; walls reverse x-momentum.
    for ( int j = ng; j < ny + ng; ++j )
    {
        if ( bc.left.type == BoundaryType::internal )
        {
        }
        else if ( bc.left.type == BoundaryType::zero_gradient )
        {
            for ( int g = 0; g < ng; ++g )
                state[idx( ng - 1 - g, j )] = state[idx( ng, j )];
        }
        else if ( bc.left.type == BoundaryType::inlet )
        {
            Conservative_vector uin{};

            require_success( calc_conservative_variables( bc.left.inlet, gamma, uin ), "calc_conservative_variables" );

            for ( int g = 0; g < ng; ++g )
                state[idx( ng - 1 - g, j )] = uin;
        }
        else if ( bc.left.type == BoundaryType::outlet )
        {
            const Conservative_vector uadj = state[idx( ng, j )];
            const Conservative_vector uo = outlet_state_2d( uadj, bc.left.pout, x_direction, gamma );

            for ( int g = 0; g < ng; ++g )
                state[idx( ng - 1 - g, j )] = uo;
        }
        else if ( is_wall_like( bc.left.type ) )
        {
            for ( int g = 0; g < ng; ++g )
            {
                Conservative_vector ghost_state = state[idx( ng + g, j )];

                ghost_state.specific_x_momentum = -ghost_state.specific_x_momentum;
                state[idx( ng - 1 - g, j )] = ghost_state;
            }
        }
        else
        {
            throw std::runtime_error( "boundary: unsupported left BC type in apply2D." );
        }

        if ( bc.right.type == BoundaryType::internal )
        {
        }
        else if ( bc.right.type == BoundaryType::zero_gradient )
        {
            for ( int g = 0; g < ng; ++g )
                state[idx( nx + ng + g, j )] = state[idx( nx + ng - 1, j )];
        }
        else if ( bc.right.type == BoundaryType::inlet )
        {
            Conservative_vector uin{};

            require_success( calc_conservative_variables( bc.right.inlet, gamma, uin ), "calc_conservative_variables" );

            for ( int g = 0; g < ng; ++g )
                state[idx( nx + ng + g, j )] = uin;
        }
        else if ( bc.right.type == BoundaryType::outlet )
        {
            const Conservative_vector uadj = state[idx( nx + ng - 1, j )];
            const Conservative_vector uo = outlet_state_2d( uadj, bc.right.pout, x_direction, gamma );

            for ( int g = 0; g < ng; ++g )
                state[idx( nx + ng + g, j )] = uo;
        }
        else if ( is_wall_like( bc.right.type ) )
        {
            for ( int g = 0; g < ng; ++g )
            {
                Conservative_vector ghost_state = state[idx( nx + ng - 1 - g, j )];

                ghost_state.specific_x_momentum = -ghost_state.specific_x_momentum;
                state[idx( nx + ng + g, j )] = ghost_state;
            }
        }
        else
        {
            throw std::runtime_error( "boundary: unsupported right BC type in apply2D." );
        }
    }

    // Fill bottom/top ghosts across the full width, including corners; walls reverse y-momentum.
    for ( int i = 0; i < nx_tot; ++i )
    {
        if ( bc.bottom.type == BoundaryType::internal )
        {
        }
        else if ( bc.bottom.type == BoundaryType::zero_gradient )
        {
            for ( int g = 0; g < ng; ++g )
                state[idx( i, ng - 1 - g )] = state[idx( i, ng )];
        }
        else if ( bc.bottom.type == BoundaryType::inlet )
        {
            Conservative_vector uin{};

            require_success( calc_conservative_variables( bc.bottom.inlet, gamma, uin ),
                "calc_conservative_variables" );
            for ( int g = 0; g < ng; ++g )
                state[idx( i, ng - 1 - g )] = uin;
        }
        else if ( bc.bottom.type == BoundaryType::outlet )
        {
            const Conservative_vector uadj = state[idx( i, ng )];
            const Conservative_vector uo = outlet_state_2d( uadj, bc.bottom.pout, y_direction, gamma );

            for ( int g = 0; g < ng; ++g )
                state[idx( i, ng - 1 - g )] = uo;
        }
        else if ( is_wall_like( bc.bottom.type ) )
        {
            for ( int g = 0; g < ng; ++g )
            {
                Conservative_vector ghost_state = state[idx( i, ng + g )];

                ghost_state.specific_y_momentum = -ghost_state.specific_y_momentum;
                state[idx( i, ng - 1 - g )] = ghost_state;
            }
        }
        else
        {
            throw std::runtime_error( "boundary: unsupported bottom BC type in apply2D." );
        }

        if ( bc.top.type == BoundaryType::internal )
        {
        }
        else if ( bc.top.type == BoundaryType::zero_gradient )
        {
            for ( int g = 0; g < ng; ++g )
                state[idx( i, ny + ng + g )] = state[idx( i, ny + ng - 1 )];
        }
        else if ( bc.top.type == BoundaryType::inlet )
        {
            Conservative_vector uin{};

            require_success( calc_conservative_variables( bc.top.inlet, gamma, uin ), "calc_conservative_variables" );

            for ( int g = 0; g < ng; ++g )
                state[idx( i, ny + ng + g )] = uin;
        }
        else if ( bc.top.type == BoundaryType::outlet )
        {
            const Conservative_vector uadj = state[idx( i, ny + ng - 1 )];
            const Conservative_vector uo = outlet_state_2d( uadj, bc.top.pout, y_direction, gamma );

            for ( int g = 0; g < ng; ++g )
                state[idx( i, ny + ng + g )] = uo;
        }
        else if ( is_wall_like( bc.top.type ) )
        {
            for ( int g = 0; g < ng; ++g )
            {
                Conservative_vector ghost_state = state[idx( i, ny + ng - 1 - g )];

                ghost_state.specific_y_momentum = -ghost_state.specific_y_momentum;
                state[idx( i, ny + ng + g )] = ghost_state;
            }
        }
        else
        {
            throw std::runtime_error( "boundary: unsupported top BC type in apply2D." );
        }
    }
}
