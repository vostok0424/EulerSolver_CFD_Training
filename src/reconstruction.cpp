#include "common_functions.hpp"


#include "reconstruction.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <array>


// Read reconstruction settings.

// Read the reconstruction scheme: firstOrder, muscl or weno5.
static ReconstructionScheme parse_scheme( const std::string &s )
{
    const std::string v = trim_string( s );

    if ( v.empty() || v == "firstOrder" )
    {
        return ReconstructionScheme::first_order;
    }

    if ( v == "muscl" )
    {
        return ReconstructionScheme::muscl;
    }

    if ( v == "weno5" )
    {
        return ReconstructionScheme::weno5;
    }

    throw std::runtime_error( "reconstruction.scheme unsupported: " + v );
}


// Read the MUSCL limiter: none, minmod or vanleer.
static SlopeLimiter parse_limiter( const std::string &s )
{
    const std::string v = trim_string( s );

    if ( v.empty() || v == "vanleer" )
    {
        return SlopeLimiter::van_leer;
    }

    if ( v == "none" )
    {
        return SlopeLimiter::none;
    }

    if ( v == "minmod" )
    {
        return SlopeLimiter::minmod;
    }

    throw std::runtime_error( "reconstruction.limiter unsupported: " + v + " (supported: none|minmod|vanleer)" );
}


// Read the reconstruction scheme, MUSCL limiter and positive WENO epsilon.
ReconstructionOptions read_reconstruction_options( const Cfg &cfg )
{
    ReconstructionOptions opt;

    opt.scheme = parse_scheme( cfg.get_string( "reconstruction.scheme", "firstOrder" ) );
    opt.limiter = parse_limiter( cfg.get_string( "reconstruction.limiter", "vanleer" ) );

    opt.eps = cfg.get_double( "reconstruction.eps", default_weno_epsilon );

    const std::string legacy_vars = trim_string( cfg.get_string( "reconstruction.variables", "" ) );

    if ( !legacy_vars.empty() && legacy_vars != "characteristic" )
    {
        throw std::runtime_error(
            "reconstruction.variables is deprecated; only characteristic reconstruction is supported" );
    }

    if ( !std::isfinite( opt.eps ) || opt.eps <= 0.0 )
    {
        throw std::runtime_error( "reconstruction.eps must be > 0" );
    }

    return opt;
}


// Shared characteristic projection.

// Store the right and left eigenvector matrices of the 2D conservative Euler system.
struct Eigen4
{
    double right_matrix[conservative_variable_count][conservative_variable_count];
    double left_matrix[conservative_variable_count][conservative_variable_count];
};


// Recover the characteristic reference state from the arithmetic mean of adjacent conserved states.
// ulc, urc (in) - valid adjacent conservative states
// gamma (in) - ideal-gas heat-capacity ratio
// rho0, u0, v0, p0, h0, c0 (out) - reference density, velocities, pressure, enthalpy and sound speed
static inline void face_state_2d( const Conservative_vector &ulc, const Conservative_vector &urc, double gamma,
    double &rho0, double &u0, double &v0, double &p0, double &h0, double &c0 )
{
    {
        EvaluatedState checked_state{};

        require_success( calc_flow_variables( ulc, gamma, checked_state ), "calc_flow_variables" );
    }

    {
        EvaluatedState checked_state{};

        require_success( calc_flow_variables( urc, gamma, checked_state ), "calc_flow_variables" );
    }

    // Check both adjacent states before averaging; an average must not hide an invalid input.
    const Conservative_vector uavg = 0.5 * ulc + 0.5 * urc;

    EvaluatedState primitive{};

    require_success( calc_flow_variables( uavg, gamma, primitive ), "calc_flow_variables" );
    rho0 = primitive.density;
    u0 = primitive.x_velocity;
    v0 = primitive.y_velocity;
    p0 = primitive.pressure;
    h0 = primitive.specific_total_enthalpy;
    c0 = primitive.sound_speed;
}


// Compute the left and right conservative eigenvector matrices for an x-normal face.
static inline void build_eigen2_dconservative_x( double u, double v, double specific_total_enthalpy, double c,
    double gamma, Eigen4 &e )
{
    const double beta = ( gamma - 1.0 ) / ( c * c );
    const double q2 = 0.5 * ( u * u + v * v );
    const double alpha = beta * q2;

    // Right-matrix columns: negative acoustic wave, entropy wave, shear wave, positive acoustic wave.
    e.right_matrix[density_component][0] = 1.0;
    e.right_matrix[x_momentum_component][0] = u - c;
    e.right_matrix[y_momentum_component][0] = v;
    e.right_matrix[total_energy_component][0] = specific_total_enthalpy - u * c;

    e.right_matrix[density_component][1] = 1.0;
    e.right_matrix[x_momentum_component][1] = u;
    e.right_matrix[y_momentum_component][1] = v;
    e.right_matrix[total_energy_component][1] = q2;

    e.right_matrix[density_component][2] = 0.0;
    e.right_matrix[x_momentum_component][2] = 0.0;
    e.right_matrix[y_momentum_component][2] = 1.0;
    e.right_matrix[total_energy_component][2] = v;

    e.right_matrix[density_component][3] = 1.0;
    e.right_matrix[x_momentum_component][3] = u + c;
    e.right_matrix[y_momentum_component][3] = v;
    e.right_matrix[total_energy_component][3] = specific_total_enthalpy + u * c;

    // Left-matrix rows project conserved variables onto the same four characteristic waves.
    e.left_matrix[0][density_component] = 0.5 * ( alpha + u / c );
    e.left_matrix[0][x_momentum_component] = -0.5 * ( beta * u + 1.0 / c );
    e.left_matrix[0][y_momentum_component] = -0.5 * beta * v;
    e.left_matrix[0][total_energy_component] = 0.5 * beta;

    e.left_matrix[1][density_component] = 1.0 - alpha;
    e.left_matrix[1][x_momentum_component] = beta * u;
    e.left_matrix[1][y_momentum_component] = beta * v;
    e.left_matrix[1][total_energy_component] = -beta;

    e.left_matrix[2][density_component] = -v;
    e.left_matrix[2][x_momentum_component] = 0.0;
    e.left_matrix[2][y_momentum_component] = 1.0;
    e.left_matrix[2][total_energy_component] = 0.0;

    e.left_matrix[3][density_component] = 0.5 * ( alpha - u / c );
    e.left_matrix[3][x_momentum_component] = -0.5 * ( beta * u - 1.0 / c );
    e.left_matrix[3][y_momentum_component] = -0.5 * beta * v;
    e.left_matrix[3][total_energy_component] = 0.5 * beta;
}


// Compute the left and right conservative eigenvector matrices for a y-normal face.
static inline void build_eigen2_dconservative_y( double u, double v, double specific_total_enthalpy, double c,
    double gamma, Eigen4 &e )
{
    const double beta = ( gamma - 1.0 ) / ( c * c );
    const double q2 = 0.5 * ( u * u + v * v );
    const double alpha = beta * q2;

    // Right-matrix columns: negative acoustic wave, entropy wave, shear wave, positive acoustic wave.
    e.right_matrix[density_component][0] = 1.0;
    e.right_matrix[x_momentum_component][0] = u;
    e.right_matrix[y_momentum_component][0] = v - c;
    e.right_matrix[total_energy_component][0] = specific_total_enthalpy - v * c;

    e.right_matrix[density_component][1] = 1.0;
    e.right_matrix[x_momentum_component][1] = u;
    e.right_matrix[y_momentum_component][1] = v;
    e.right_matrix[total_energy_component][1] = q2;

    e.right_matrix[density_component][2] = 0.0;
    e.right_matrix[x_momentum_component][2] = 1.0;
    e.right_matrix[y_momentum_component][2] = 0.0;
    e.right_matrix[total_energy_component][2] = u;

    e.right_matrix[density_component][3] = 1.0;
    e.right_matrix[x_momentum_component][3] = u;
    e.right_matrix[y_momentum_component][3] = v + c;
    e.right_matrix[total_energy_component][3] = specific_total_enthalpy + v * c;

    // Left-matrix rows project conserved variables onto the same four characteristic waves.
    e.left_matrix[0][density_component] = 0.5 * ( alpha + v / c );
    e.left_matrix[0][x_momentum_component] = -0.5 * beta * u;
    e.left_matrix[0][y_momentum_component] = -0.5 * ( beta * v + 1.0 / c );
    e.left_matrix[0][total_energy_component] = 0.5 * beta;

    e.left_matrix[1][density_component] = 1.0 - alpha;
    e.left_matrix[1][x_momentum_component] = beta * u;
    e.left_matrix[1][y_momentum_component] = beta * v;
    e.left_matrix[1][total_energy_component] = -beta;

    e.left_matrix[2][density_component] = -u;
    e.left_matrix[2][x_momentum_component] = 1.0;
    e.left_matrix[2][y_momentum_component] = 0.0;
    e.left_matrix[2][total_energy_component] = 0.0;

    e.left_matrix[3][density_component] = 0.5 * ( alpha - v / c );
    e.left_matrix[3][x_momentum_component] = -0.5 * beta * u;
    e.left_matrix[3][y_momentum_component] = -0.5 * ( beta * v - 1.0 / c );
    e.left_matrix[3][total_energy_component] = 0.5 * beta;
}


// Map characteristic components back to conserved variables with the right eigenvector matrix.
static inline void reconstruct_conservative_from_char_2d( const Eigen4 &e, const double *w, Conservative_vector &q )
{
    std::array<double, conservative_variable_count> components{};

    for ( std::size_t m = 0; m < conservative_variable_count; ++m )
    {
        components[m] = e.right_matrix[m][0] * w[0] + e.right_matrix[m][1] * w[1] + e.right_matrix[m][2] * w[2] +
                        e.right_matrix[m][3] * w[3];
    }

    q = unpack_state( components.data() );
}


// Project a conservative state onto the four characteristic components.
static inline void project_char4( const Eigen4 &e, const Conservative_vector &q, double w[conservative_variable_count] )
{
    for ( std::size_t s = 0; s < conservative_variable_count; ++s )
    {
        w[s] = e.left_matrix[s][density_component] * q.specific_mass +
               e.left_matrix[s][x_momentum_component] * q.specific_x_momentum +
               e.left_matrix[s][y_momentum_component] * q.specific_y_momentum +
               e.left_matrix[s][total_energy_component] * q.specific_energy;
    }
}


// Compute the row-major index of a cell in the ghosted array.
static inline int idx2( int i, int j, int nx_tot )
{
    return i + nx_tot * j;
}


// 1. First-order reconstruction.

// Copy adjacent cell states to the left and right sides of each x-face.
static void reconstruct_first_order_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    int nx_tot, std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx )
{
    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx + 1; ++i )
        {
            const int face = i + ( nx + 1 ) * j;

            ulx[face] = state[idx2( ng + i - 1, ng + j, nx_tot )];
            urx[face] = state[idx2( ng + i, ng + j, nx_tot )];
        }
    }
}


// Copy adjacent cell states to the left and right sides of each y-face.
static void reconstruct_first_order_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    int nx_tot, std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury )
{
    for ( int j = 0; j < ny + 1; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const int face = i + nx * j;

            uly[face] = state[idx2( ng + i, ng + j - 1, nx_tot )];
            ury[face] = state[idx2( ng + i, ng + j, nx_tot )];
        }
    }
}


// 2. Second-order MUSCL reconstruction.

// Return the sign of x: -1, 0 or 1.
static inline double sgn( double x )
{
    return ( x > 0 ) - ( x < 0 );
}


// Return the smaller same-sign slope, or zero when the slopes have opposite signs or include zero.
static inline double minmod( double a, double b )
{
    if ( a * b <= 0.0 )
    {
        return 0.0;
    }

    return sgn( a ) * std::min( std::abs( a ), std::abs( b ) );
}


// Compute the van Leer limited slope; return zero when the slopes have opposite signs or include zero.
static inline double vanleer( double a, double b )
{
    if ( a * b <= 0.0 )
    {
        return 0.0;
    }

    return ( 2.0 * a * b ) / ( a + b );
}


// Compute the MUSCL slope using the selected limiter.
static inline double apply_limiter( SlopeLimiter lim, double dl, double dr )
{
    switch ( lim )
    {
    case SlopeLimiter::none:
        return 0.5 * ( dl + dr );
    case SlopeLimiter::minmod:
        return minmod( dl, dr );
    case SlopeLimiter::van_leer:
        return vanleer( dl, dr );
    }

    throw std::runtime_error( "Invalid reconstruction limiter enum" );
}


// Reconstruct both face states with second-order MUSCL slopes in characteristic space.
// eig (in) - eigensystem at the conservative arithmetic-mean state
// limiter (in) - slope limiter
// wc (in) - four cells of characteristic components, ordered along the face normal
// ulf, urf (out) - left and right conservative face states
static inline void reconstruct_face_musclfrom_char_cache( const Eigen4 &eig, SlopeLimiter limiter,
    const double ( &wc )[4][conservative_variable_count], Conservative_vector &ulf, Conservative_vector &urf )
{
    double w_lf[conservative_variable_count]{};
    double w_rf[conservative_variable_count]{};

    for ( std::size_t s = 0; s < conservative_variable_count; ++s )
    {
        const double wm_l = wc[0][s];
        const double w0_l = wc[1][s];
        const double wp_l = wc[2][s];
        const double s_l = apply_limiter( limiter, w0_l - wm_l, wp_l - w0_l );

        w_lf[s] = w0_l + 0.5 * s_l;

        const double wm_r = wc[1][s];
        const double w0_r = wc[2][s];
        const double wp_r = wc[3][s];
        const double s_r = apply_limiter( limiter, w0_r - wm_r, wp_r - w0_r );

        w_rf[s] = w0_r - 0.5 * s_r;
    }

    reconstruct_conservative_from_char_2d( eig, w_lf, ulf );
    reconstruct_conservative_from_char_2d( eig, w_rf, urf );
}


// 3. Fifth-order WENO reconstruction.

// Compute the inverse-square factor used in Jiang-Shu WENO weights.
static inline double inv_pow_weno( double x )
{
    return 1.0 / ( x * x );
}


// Compute and normalize the three Jiang-Shu nonlinear weights.
// b0, b1, b2 (in) - smoothness indicators
// eps (in) - positive epsilon in the inverse-square weights
// g0, g1, g2 (in) - linear weights
// w0, w1, w2 (out) - normalized nonlinear weights
static inline void weno5_weights( double b0, double b1, double b2, double eps, double g0, double g1, double g2,
    double &w0, double &w1, double &w2 )
{
    // Each unnormalized weight is its linear weight divided by (epsilon + beta) squared.
    const double a0 = g0 * inv_pow_weno( eps + b0 );
    const double a1 = g1 * inv_pow_weno( eps + b1 );
    const double a2 = g2 * inv_pow_weno( eps + b2 );
    // Normalize the weights so their sum is one.
    const double asum = a0 + a1 + a2;

    w0 = a0 / asum;
    w1 = a1 / asum;
    w2 = a2 / asum;
}


// Compute the left face value from the five-cell stencil centered on the left cell.
static inline double weno5_left( double qim2, double qim1, double qi, double qip1, double qip2, double eps )
{
    // Evaluate the three third-order candidate polynomials at the face.
    const double p0 = ( 1.0 / 3.0 ) * qim2 - ( 7.0 / 6.0 ) * qim1 + ( 11.0 / 6.0 ) * qi;
    const double p1 = ( -1.0 / 6.0 ) * qim1 + ( 5.0 / 6.0 ) * qi + ( 1.0 / 3.0 ) * qip1;
    const double p2 = ( 1.0 / 3.0 ) * qi + ( 5.0 / 6.0 ) * qip1 - ( 1.0 / 6.0 ) * qip2;

    // Compute the Jiang-Shu smoothness indicators for the three substencils.
    const double d0 = qim2 - 2.0 * qim1 + qi;
    const double d1 = qim1 - 2.0 * qi + qip1;
    const double d2 = qi - 2.0 * qip1 + qip2;

    const double b0 =
        ( 13.0 / 12.0 ) * d0 * d0 + 0.25 * ( qim2 - 4.0 * qim1 + 3.0 * qi ) * ( qim2 - 4.0 * qim1 + 3.0 * qi );
    const double b1 = ( 13.0 / 12.0 ) * d1 * d1 + 0.25 * ( qim1 - qip1 ) * ( qim1 - qip1 );
    const double b2 =
        ( 13.0 / 12.0 ) * d2 * d2 + 0.25 * ( 3.0 * qi - 4.0 * qip1 + qip2 ) * ( 3.0 * qi - 4.0 * qip1 + qip2 );

    // Use the fifth-order linear weights before applying smoothness-dependent weighting.
    const double g0 = 0.1;
    const double g1 = 0.6;
    const double g2 = 0.3;

    double w0 = 0.0, w1 = 0.0, w2 = 0.0;

    weno5_weights( b0, b1, b2, eps, g0, g1, g2, w0, w1, w2 );

    return w0 * p0 + w1 * p1 + w2 * p2;
}


// Compute the right face value by reversing the five-cell stencil.
static inline double weno5_right( double qim2, double qim1, double qi, double qip1, double qip2, double eps )
{
    return weno5_left( qip2, qip1, qi, qim1, qim2, eps );
}


// Reconstruct both face states from a six-cell characteristic stencil using WENO5.
// eig (in) - eigensystem at the conservative arithmetic-mean state
// eps (in) - positive epsilon for the Jiang-Shu weights
// wc (in) - six cells of characteristic components, ordered along the face normal
// ulf, urf (out) - left and right conservative face states
static inline void reconstruct_face_weno5_from_char_cache( const Eigen4 &eig, double eps,
    const double ( &wc )[6][conservative_variable_count], Conservative_vector &ulf, Conservative_vector &urf )
{
    double w_lf[conservative_variable_count]{};
    double w_rf[conservative_variable_count]{};

    for ( std::size_t s = 0; s < conservative_variable_count; ++s )
    {
        w_lf[s] = weno5_left( wc[0][s], wc[1][s], wc[2][s], wc[3][s], wc[4][s], eps );
        w_rf[s] = weno5_right( wc[1][s], wc[2][s], wc[3][s], wc[4][s], wc[5][s], eps );
    }

    reconstruct_conservative_from_char_2d( eig, w_lf, ulf );
    reconstruct_conservative_from_char_2d( eig, w_rf, urf );
}


// Store the configured reconstruction scheme and its parameters.
Reconstruction2D::Reconstruction2D( const Cfg &cfg ) : opt_( read_reconstruction_options( cfg ) )
{
}


// Compute left and right conservative states on all (nx + 1) * ny x-faces.
// state (in) - local row-major conservative field with valid ghost cells
// nx, ny, ng (in) - interior cell counts and ghost width
// gamma (in) - ideal-gas heat-capacity ratio
// ulx, urx (out) - left and right conservative face states
void Reconstruction2D::reconstruct_faces_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx ) const
{
    const int nx_tot = nx + 2 * ng;
    const int ny_tot = ny + 2 * ng;

    if ( (int)state.size() != nx_tot * ny_tot )
    {
        throw std::runtime_error( "Reconstruction2D::reconstructFacesX: U size mismatch" );
    }

    if ( ng < required_ghost_cells( opt_.scheme ) )
    {
        throw std::runtime_error(
            "Reconstruction2D::reconstructFacesX: ng too small for selected reconstruction.scheme" );
    }

    const size_t n_face_x = static_cast<size_t>( nx + 1 ) * static_cast<size_t>( ny );

    if ( ulx.size() != n_face_x )
        ulx.resize( n_face_x );
    if ( urx.size() != n_face_x )
        urx.resize( n_face_x );

    // First order uses adjacent cell averages without characteristic projection.
    if ( opt_.scheme == ReconstructionScheme::first_order )
    {
        reconstruct_first_order_x( state, nx, ny, ng, nx_tot, ulx, urx );
        return;
    }

    // Build one eigensystem per face, then project the selected MUSCL or WENO stencil.
    auto reconstruct_face =
        [&]( int i_face, int j_face, ReconstructionScheme scheme, Conservative_vector &ulf, Conservative_vector &urf )
    {
        const int i_r = ng + i_face;
        const int i_l = i_r - 1;
        const int j = ng + j_face;

        const int k_l = idx2( i_l, j, nx_tot );
        const int k_r = idx2( i_r, j, nx_tot );

        double rho0 = 0.0, u0 = 0.0, v0 = 0.0, p0 = 0.0, h0 = 0.0, c0 = 0.0;

        face_state_2d( state[k_l], state[k_r], gamma, rho0, u0, v0, p0, h0, c0 );

        Eigen4 eig{};

        build_eigen2_dconservative_x( u0, v0, h0, c0, gamma, eig );

        if ( scheme == ReconstructionScheme::muscl )
        {
            double wc[4][conservative_variable_count];
            const Conservative_vector *q_base = &state[idx2( i_l - 1, j, nx_tot )];

            project_char4( eig, q_base[0], wc[0] );
            project_char4( eig, q_base[1], wc[1] );
            project_char4( eig, q_base[2], wc[2] );
            project_char4( eig, q_base[3], wc[3] );
            reconstruct_face_musclfrom_char_cache( eig, opt_.limiter, wc, ulf, urf );
        }
        else if ( scheme == ReconstructionScheme::weno5 )
        {
            double wc[6][conservative_variable_count];
            const Conservative_vector *q_base = &state[idx2( i_l - 2, j, nx_tot )];

            project_char4( eig, q_base[0], wc[0] );
            project_char4( eig, q_base[1], wc[1] );
            project_char4( eig, q_base[2], wc[2] );
            project_char4( eig, q_base[3], wc[3] );
            project_char4( eig, q_base[4], wc[4] );
            project_char4( eig, q_base[5], wc[5] );
            reconstruct_face_weno5_from_char_cache( eig, opt_.eps, wc, ulf, urf );
        }
        else
        {
            throw std::runtime_error( "Characteristic reconstruction: unsupported scheme" );
        }
    };

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx + 1; ++i )
        {
            const int f = i + ( nx + 1 ) * j;

            reconstruct_face( i, j, opt_.scheme, ulx[f], urx[f] );
        }
    }
}


// Compute left and right conservative states on all nx * (ny + 1) y-faces.
// state (in) - local row-major conservative field with valid ghost cells
// nx, ny, ng (in) - interior cell counts and ghost width
// gamma (in) - ideal-gas heat-capacity ratio
// uly, ury (out) - left and right conservative face states
void Reconstruction2D::reconstruct_faces_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury ) const
{
    const int nx_tot = nx + 2 * ng;
    const int ny_tot = ny + 2 * ng;

    if ( (int)state.size() != nx_tot * ny_tot )
    {
        throw std::runtime_error( "Reconstruction2D::reconstructFacesY: U size mismatch" );
    }

    if ( ng < required_ghost_cells( opt_.scheme ) )
    {
        throw std::runtime_error(
            "Reconstruction2D::reconstructFacesY: ng too small for selected reconstruction.scheme" );
    }

    const size_t n_face_y = static_cast<size_t>( nx ) * static_cast<size_t>( ny + 1 );

    if ( uly.size() != n_face_y )
        uly.resize( n_face_y );
    if ( ury.size() != n_face_y )
        ury.resize( n_face_y );

    if ( opt_.scheme == ReconstructionScheme::first_order )
    {
        reconstruct_first_order_y( state, nx, ny, ng, nx_tot, uly, ury );
        return;
    }

    auto reconstruct_face =
        [&]( int i_face, int j_face, ReconstructionScheme scheme, Conservative_vector &ulf, Conservative_vector &urf )
    {
        const int i = ng + i_face;
        const int j_t = ng + j_face;
        const int j_b = j_t - 1;

        const int k_b = idx2( i, j_b, nx_tot );
        const int k_t = idx2( i, j_t, nx_tot );

        double rho0 = 0.0, u0 = 0.0, v0 = 0.0, p0 = 0.0, h0 = 0.0, c0 = 0.0;

        face_state_2d( state[k_b], state[k_t], gamma, rho0, u0, v0, p0, h0, c0 );

        Eigen4 eig{};

        build_eigen2_dconservative_y( u0, v0, h0, c0, gamma, eig );

        if ( scheme == ReconstructionScheme::muscl )
        {
            double wc[4][conservative_variable_count];
            const Conservative_vector *q_cache[4] = { &state[idx2( i, j_b - 1, nx_tot )],
                &state[idx2( i, j_b, nx_tot )], &state[idx2( i, j_t, nx_tot )], &state[idx2( i, j_t + 1, nx_tot )] };

            project_char4( eig, *q_cache[0], wc[0] );
            project_char4( eig, *q_cache[1], wc[1] );
            project_char4( eig, *q_cache[2], wc[2] );
            project_char4( eig, *q_cache[3], wc[3] );
            reconstruct_face_musclfrom_char_cache( eig, opt_.limiter, wc, ulf, urf );
        }
        else if ( scheme == ReconstructionScheme::weno5 )
        {
            double wc[6][conservative_variable_count];
            const Conservative_vector *q_cache[6] = { &state[idx2( i, j_b - 2, nx_tot )],
                &state[idx2( i, j_b - 1, nx_tot )], &state[idx2( i, j_b, nx_tot )], &state[idx2( i, j_t, nx_tot )],
                &state[idx2( i, j_t + 1, nx_tot )], &state[idx2( i, j_t + 2, nx_tot )] };

            project_char4( eig, *q_cache[0], wc[0] );
            project_char4( eig, *q_cache[1], wc[1] );
            project_char4( eig, *q_cache[2], wc[2] );
            project_char4( eig, *q_cache[3], wc[3] );
            project_char4( eig, *q_cache[4], wc[4] );
            project_char4( eig, *q_cache[5], wc[5] );
            reconstruct_face_weno5_from_char_cache( eig, opt_.eps, wc, ulf, urf );
        }
        else
        {
            throw std::runtime_error( "Characteristic reconstruction: unsupported scheme" );
        }
    };

    for ( int j = 0; j < ny + 1; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const int f = i + nx * j;

            reconstruct_face( i, j, opt_.scheme, uly[f], ury[f] );
        }
    }
}
