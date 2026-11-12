#include "common_functions.hpp"


#include "setFields.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Compute a region state from direct values or normal-shock jump conditions.
//
// gamma (in) - ideal-gas heat-capacity ratio
// region (in) - initialization region with a direct or upstream primitive state and optional shock controls
Primitive_vector build_region_prim_2d( double gamma, const SetFieldsRegion &region )
{
    Primitive_vector primitive = region.primitive;
    const double shock_mach = region.shock_mach;

    if ( shock_mach > 0.0 )
    {
        if ( shock_mach <= 1.0 )
        {
            throw std::runtime_error( "setFields: shockMach must be > 1" );
        }

        const ShockDir dir = region.shock_dir;

        double nx = 0.0;
        double ny = 0.0;

        switch ( dir )
        {
        case ShockDir::PlusX:
            nx = 1.0;
            ny = 0.0;
            break;
        case ShockDir::MinusX:
            nx = -1.0;
            ny = 0.0;
            break;
        case ShockDir::PlusY:
            nx = 0.0;
            ny = 1.0;
            break;
        case ShockDir::MinusY:
            nx = 0.0;
            ny = -1.0;
            break;
        }

        const double rho1 = primitive.density;
        const double u1 = primitive.x_velocity;
        const double v1 = primitive.y_velocity;
        const double p1 = primitive.pressure;
        const double a1 = std::sqrt( gamma * p1 / rho1 );
        const double un1 = u1 * nx + v1 * ny;
        const double utx = u1 - un1 * nx;
        const double uty = v1 - un1 * ny;
        // shockMach is relative to the upstream gas; preserve its tangential velocity.
        const double shock_velocity = un1 + shock_mach * a1;
        const double w1 = shock_mach * a1;
        // Apply constant-gamma normal-shock density and pressure jump conditions.
        const double rho_ratio =
            ( ( gamma + 1.0 ) * shock_mach * shock_mach ) / ( ( gamma - 1.0 ) * shock_mach * shock_mach + 2.0 );
        const double p_ratio = 1.0 + ( 2.0 * gamma / ( gamma + 1.0 ) ) * ( shock_mach * shock_mach - 1.0 );
        const double rho2 = rho1 * rho_ratio;
        const double p2 = p1 * p_ratio;
        const double w2 = w1 * ( rho1 / rho2 );
        const double un2 = shock_velocity - w2;

        primitive.density = rho2;
        primitive.x_velocity = utx + un2 * nx;
        primitive.y_velocity = uty + un2 * ny;
        primitive.pressure = p2;
        return primitive;
    }

    return primitive;
}


// Fill the interior background, then apply regions in order using the supplied settings.
//
// state (in/out) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
// options (in) - parsed options consumed by this module
//
// Fill the interior with a background state, then apply rectangular regions in index order.
// Later regions overwrite earlier regions where their cell-center selections overlap.
void set_fields_2d( std::vector<Conservative_vector> &state, int nx, int ny, int ng, double x0, double x1, double y0,
    double y1, double gamma, const SetFieldsOptions &options )
{
    const int nx_tot = nx + 2 * ng;
    auto idx = [&]( int i, int j )
    {
        return i + nx_tot * j;
    };

    const double dx = ( x1 - x0 ) / static_cast<double>( nx );
    const double dy = ( y1 - y0 ) / static_cast<double>( ny );

    const Primitive_vector &bg = options.background;

    Conservative_vector ubg{};

    require_success( calc_conservative_variables( bg, gamma, ubg ), "calc_conservative_variables" );

    for ( int j = 0; j < ny; ++j )
        for ( int i = 0; i < nx; ++i )
            state[idx( ng + i, ng + j )] = ubg;

    // Later regions overwrite earlier selections; omitted bounds use the local domain.
    for ( const auto &region : options.regions )
    {
        const double x_min = region.x_min.value_or( x0 );
        const double x_max = region.x_max.value_or( x1 );
        const double y_min = region.y_min.value_or( y0 );
        const double y_max = region.y_max.value_or( y1 );
        const Primitive_vector primitive = build_region_prim_2d( gamma, region );
        Conservative_vector ureg{};

        require_success( calc_conservative_variables( primitive, gamma, ureg ), "calc_conservative_variables" );

        // Select cells by their center coordinates, including centers on the box boundary.
        for ( int j = 0; j < ny; ++j )
        {
            const double y = y0 + ( j + 0.5 ) * dy;

            if ( y < y_min || y > y_max )
            {
                continue;
            }

            for ( int i = 0; i < nx; ++i )
            {
                const double x = x0 + ( i + 0.5 ) * dx;

                if ( x < x_min || x > x_max )
                {
                    continue;
                }

                state[idx( ng + i, ng + j )] = ureg;
            }
        }
    }
}
