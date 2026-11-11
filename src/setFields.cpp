#include "common_functions.hpp"


#include "setFields.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Select the normal-shock propagation direction.
enum class ShockDir
{
    PlusX,
    MinusX,
    PlusY,
    MinusY
};


// Read the shock propagation direction along the x- or y-axis.
ShockDir parse_shock_dir( const std::string &s )
{
    if ( s == "+x" || s == "x" || s == "+X" || s == "X" )
    {
        return ShockDir::PlusX;
    }

    if ( s == "-x" || s == "-X" )
    {
        return ShockDir::MinusX;
    }

    if ( s == "+y" || s == "y" || s == "+Y" || s == "Y" )
    {
        return ShockDir::PlusY;
    }

    if ( s == "-y" || s == "-Y" )
    {
        return ShockDir::MinusY;
    }

    throw std::runtime_error( "setFields: invalid shockDir='" + s + "' (expected +x, -x, +y, or -y)" );
}


// Compute a region state from direct values or normal-shock jump conditions.
Primitive_vector build_region_prim_2d( const Primitive_vector &bg, double gamma, const Cfg &cfg,
    const std::string &base )
{
    Primitive_vector primitive = bg;
    const double shock_mach = cfg.get_double( base + "shockMach", -1.0 );

    if ( shock_mach > 0.0 )
    {
        if ( shock_mach <= 1.0 )
        {
            throw std::runtime_error( "setFields: shockMach must be > 1 for " + base );
        }

        primitive.density = cfg.get_double( base + "rho", bg.density );
        primitive.x_velocity = cfg.get_double( base + "u", 0.0 );
        primitive.y_velocity = cfg.get_double( base + "v", 0.0 );
        primitive.pressure = cfg.get_double( base + "p", bg.pressure );

        const ShockDir dir = parse_shock_dir( cfg.get_string( base + "shockDir", "+x" ) );

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

    primitive.density = cfg.get_double( base + "rho", bg.density );
    primitive.x_velocity = cfg.get_double( base + "u", bg.x_velocity );
    primitive.y_velocity = cfg.get_double( base + "v", bg.y_velocity );
    primitive.pressure = cfg.get_double( base + "p", bg.pressure );
    return primitive;
}


// Fill the interior with a background state, then apply rectangular regions in index order.
// state (in/out) - ghosted conservative field; only interior cells are assigned
// nx, ny, ng (in) - local interior sizes and ghost width
// x0, x1, y0, y1 (in) - local interior-domain bounds
// gamma, cfg (in) - ideal-gas ratio and setFields settings
// Later regions overwrite earlier regions where their cell-center selections overlap.
void set_fields_2d( std::vector<Conservative_vector> &state, int nx, int ny, int ng, double x0, double x1, double y0,
    double y1, double gamma, const Cfg &cfg )
{
    const int nx_tot = nx + 2 * ng;
    auto idx = [&]( int i, int j )
    {
        return i + nx_tot * j;
    };

    const double dx = ( x1 - x0 ) / static_cast<double>( nx );
    const double dy = ( y1 - y0 ) / static_cast<double>( ny );

    Primitive_vector bg{};

    // Read the background primitive state from setFields.bg.rho, u, v and p.
    bg.density = cfg.get_double( "setFields.bg.rho", 1.0 );
    bg.x_velocity = cfg.get_double( "setFields.bg.u", 0.0 );
    bg.y_velocity = cfg.get_double( "setFields.bg.v", 0.0 );
    bg.pressure = cfg.get_double( "setFields.bg.p", 1.0 );

    Conservative_vector ubg{};

    require_success( calc_conservative_variables( bg, gamma, ubg ), "calc_conservative_variables" );

    for ( int j = 0; j < ny; ++j )
        for ( int i = 0; i < nx; ++i )
            state[idx( ng + i, ng + j )] = ubg;

    const int n_regions = cfg.get_int( "setFields.nRegions", 0 );

    // Apply setFields.regionK boxes in ascending K; later boxes overwrite earlier selections.
    for ( int r = 1; r <= n_regions; ++r )
    {
        const std::string base = "setFields.region" + std::to_string( r ) + ".";
        const double x_min = cfg.get_double( base + "xMin", x0 );
        const double x_max = cfg.get_double( base + "xMax", x1 );
        const double y_min = cfg.get_double( base + "yMin", y0 );
        const double y_max = cfg.get_double( base + "yMax", y1 );

        const Primitive_vector primitive = build_region_prim_2d( bg, gamma, cfg, base );
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
