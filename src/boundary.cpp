#include "common_functions.hpp"

#include "boundary.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>


// Check whether the boundary reflects normal momentum.
//
// t (in) - physical boundary type
static bool is_wall_like( BoundaryType t )
{
    return ( t == BoundaryType::slip_wall || t == BoundaryType::symmetry );
}


// Extrapolate a supersonic state; impose outlet pressure when the normal Mach magnitude is below one.
//
// uin (in) - conserved state of the adjacent interior cell
// pout (in) - prescribed positive outlet pressure
// normal_dir (in) - outlet-normal coordinate direction
// gamma (in) - ideal-gas heat-capacity ratio
static Conservative_vector outlet_state_2d( const Conservative_vector &uin, double pout, int normal_dir,
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
//
// state (in/out) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// bc (in) - physical boundary types, inlet states, outlet pressures and gas model
void apply_boundary_conditions( std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    const BoundaryConditions2D &bc )
{
    const double gamma = bc.gamma;
    const int nx_tot = nx + 2 * ng;
    const int ny_tot = ny + 2 * ng;
    auto idx = [&]( int i, int j )
    {
        return i + nx_tot * j;
    };

    // Fill left/right ghosts also in exchanged y halos. Physical y corners are filled below.
    for ( int j = 0; j < ny_tot; ++j )
    {
        if ( ( j < ng && bc.bottom.type != BoundaryType::internal ) ||
             ( j >= ny + ng && bc.top.type != BoundaryType::internal ) )
        {
            continue;
        }

        if ( bc.left.type == BoundaryType::internal )
        {
        }
        else if ( bc.left.type == BoundaryType::zero_gradient )
        {
            for ( int g = 0; g < ng; ++g )
            {
                state[idx( ng - 1 - g, j )] = state[idx( ng, j )];
            }
        }
        else if ( bc.left.type == BoundaryType::inlet )
        {
            Conservative_vector uin{};

            require_success( calc_conservative_variables( bc.left.inlet, gamma, uin ), "calc_conservative_variables" );

            for ( int g = 0; g < ng; ++g )
            {
                state[idx( ng - 1 - g, j )] = uin;
            }
        }
        else if ( bc.left.type == BoundaryType::outlet )
        {
            const Conservative_vector uadj = state[idx( ng, j )];
            const Conservative_vector uo = outlet_state_2d( uadj, bc.left.pout, x_direction, gamma );

            for ( int g = 0; g < ng; ++g )
            {
                state[idx( ng - 1 - g, j )] = uo;
            }
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
            {
                state[idx( nx + ng + g, j )] = state[idx( nx + ng - 1, j )];
            }
        }
        else if ( bc.right.type == BoundaryType::inlet )
        {
            Conservative_vector uin{};

            require_success( calc_conservative_variables( bc.right.inlet, gamma, uin ), "calc_conservative_variables" );

            for ( int g = 0; g < ng; ++g )
            {
                state[idx( nx + ng + g, j )] = uin;
            }
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
            {
                state[idx( i, ng - 1 - g )] = uin;
            }
        }
        else if ( bc.bottom.type == BoundaryType::outlet )
        {
            const Conservative_vector uadj = state[idx( i, ng )];
            const Conservative_vector uo = outlet_state_2d( uadj, bc.bottom.pout, y_direction, gamma );

            for ( int g = 0; g < ng; ++g )
            {
                state[idx( i, ng - 1 - g )] = uo;
            }
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
            {
                state[idx( i, ny + ng + g )] = state[idx( i, ny + ng - 1 )];
            }
        }
        else if ( bc.top.type == BoundaryType::inlet )
        {
            Conservative_vector uin{};

            require_success( calc_conservative_variables( bc.top.inlet, gamma, uin ), "calc_conservative_variables" );

            for ( int g = 0; g < ng; ++g )
            {
                state[idx( i, ny + ng + g )] = uin;
            }
        }
        else if ( bc.top.type == BoundaryType::outlet )
        {
            const Conservative_vector uadj = state[idx( i, ny + ng - 1 )];
            const Conservative_vector uo = outlet_state_2d( uadj, bc.top.pout, y_direction, gamma );

            for ( int g = 0; g < ng; ++g )
            {
                state[idx( i, ny + ng + g )] = uo;
            }
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
