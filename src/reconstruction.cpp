#include "common_functions.hpp"


#include "reconstruction.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>


// Shared characteristic projection.

// Store the right and left eigenvector matrices of the 2D conservative Euler system.
struct Eigen4
{
    double right_matrix[conservative_variable_count][conservative_variable_count];
    double left_matrix[conservative_variable_count][conservative_variable_count];
};


// Recover the characteristic reference state from the arithmetic mean of adjacent conserved states.
//
// ulc (in) - conserved variables in the cell to the left of the interface
// urc (in) - conserved variables in the cell to the right of the interface
// gamma (in) - ideal-gas heat-capacity ratio
// rho0 (out) - density at the characteristic reference state
// u0 (out) - x velocity at the characteristic reference state
// v0 (out) - y velocity at the characteristic reference state
// p0 (out) - pressure at the characteristic reference state
// h0 (out) - specific total enthalpy at the characteristic reference state
// c0 (out) - sound speed at the characteristic reference state
// left_checked, right_checked (in/out) - successful adjacent-cell checks for this reconstruction call
static void face_state_2d( const Conservative_vector &ulc, const Conservative_vector &urc, double gamma, double &rho0,
    double &u0, double &v0, double &p0, double &h0, double &c0,
    unsigned char &left_checked, unsigned char &right_checked )
{
    if ( !left_checked )
    {
        EvaluatedState checked_state{};

        require_success( calc_flow_variables( ulc, gamma, checked_state ), "calc_flow_variables" );
        left_checked = 1;
    }

    if ( !right_checked )
    {
        EvaluatedState checked_state{};

        require_success( calc_flow_variables( urc, gamma, checked_state ), "calc_flow_variables" );
        right_checked = 1;
    }

    // Both adjacent states must pass checks in this call before averaging.
    // Shared cells reuse successful checks; each new face average is still evaluated independently.
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
//
// u (in) - x velocity of the characteristic reference state
// v (in) - y velocity of the characteristic reference state
// specific_total_enthalpy (in) - total enthalpy per unit mass of the reference state
// c (in) - sound speed of the characteristic reference state
// gamma (in) - ideal-gas heat-capacity ratio
// e (out) - left and right characteristic eigenvector matrices
static void build_eigen2_dconservative_x( double u, double v, double specific_total_enthalpy, double c, double gamma, Eigen4 &e )
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
//
// u (in) - x velocity of the characteristic reference state
// v (in) - y velocity of the characteristic reference state
// specific_total_enthalpy (in) - total enthalpy per unit mass of the reference state
// c (in) - sound speed of the characteristic reference state
// gamma (in) - ideal-gas heat-capacity ratio
// e (out) - left and right characteristic eigenvector matrices
static void build_eigen2_dconservative_y( double u, double v, double specific_total_enthalpy, double c, double gamma, Eigen4 &e )
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
//
// e (in) - left and right characteristic eigenvector matrices
// w (in) - characteristic components in conservative-component order
// q (out) - conservative state represented in the characteristic basis
static void reconstruct_conservative_from_char_2d( const Eigen4 &e, const double *w, Conservative_vector &q )
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
//
// e (in) - left and right characteristic eigenvector matrices
// q (in) - conservative state represented in the characteristic basis
// w (out) - characteristic components in conservative-component order
static void project_char4( const Eigen4 &e, const Conservative_vector &q, double w[conservative_variable_count] )
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
//
// i (in) - cell or face index along x
// j (in) - cell or face index along y
// nx_tot (in) - number of cells along x including ghost cells
static int idx2( int i, int j, int nx_tot )
{
    return i + nx_tot * j;
}


// 1. First-order reconstruction.
//
// Implementation: piecewise-constant conservative cell averages; the numerical flux is selected separately.
//
// Reference: S. K. Godunov (1959), "A difference method for numerical calculation of discontinuous
// solutions of the equations of hydrodynamics", Mat. Sb. (N.S.) 47(89)(3), 271-306.
// https://www.mathnet.ru/eng/sm4873

// Copy adjacent cell states to the left and right sides of each x-face.
//
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// nx_tot (in) - number of cells along x including ghost cells
// ulx (out) - left conservative states on x-normal faces
// urx (out) - right conservative states on x-normal faces
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
//
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// nx_tot (in) - number of cells along x including ghost cells
// uly (out) - left conservative states on y-normal faces
// ury (out) - right conservative states on y-normal faces
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
//
// Implementation: characteristic piecewise-linear slopes with none, minmod or van Leer limiting.
// Time integration is handled separately from this spatial reconstruction.
//
// Reference: B. van Leer (1979), "Towards the ultimate conservative difference scheme. V.
// A second-order sequel to Godunov's method", J. Comput. Phys. 32(1), 101-136.
// https://doi.org/10.1016/0021-9991(79)90145-1

// Return the sign of x: -1, 0 or 1.
//
// x (in) - scalar argument of the reconstruction helper
static double sgn( double x )
{
    return ( x > 0 ) - ( x < 0 );
}


// Return the smaller same-sign slope, or zero when the slopes have opposite signs or include zero.
//
// a (in) - first scalar slope candidate
// b (in) - second scalar slope candidate
static double minmod( double a, double b )
{
    if ( a * b <= 0.0 )
    {
        return 0.0;
    }

    return sgn( a ) * std::min( std::abs( a ), std::abs( b ) );
}


// Compute the van Leer limited slope; return zero when the slopes have opposite signs or include zero.
//
// a (in) - first scalar slope candidate
// b (in) - second scalar slope candidate
static double vanLeer( double a, double b )
{
    if ( a * b <= 0.0 )
    {
        return 0.0;
    }

    return ( 2.0 * a * b ) / ( a + b );
}


// Compute the MUSCL slope using the selected limiter.
//
// lim (in) - selected slope limiter
// dl (in) - difference between the center and left stencil values
// dr (in) - difference between the right and center stencil values
static double apply_limiter( SlopeLimiter lim, double dl, double dr )
{
    switch ( lim )
    {
    case SlopeLimiter::none:
        return 0.5 * ( dl + dr );
    case SlopeLimiter::minmod:
        return minmod( dl, dr );
    case SlopeLimiter::van_leer:
        return vanLeer( dl, dr );
    }

    throw std::runtime_error( "Invalid reconstruction limiter enum" );
}


// Reconstruct both face states with second-order MUSCL slopes in characteristic space.
//
// eig (in) - characteristic eigenvectors at the face reference state
// limiter (in) - selected MUSCL slope limiter
// wc (in) - cell stencil of characteristic components ordered along the face normal
// ulf (out) - left conservative state reconstructed at the interface
// urf (out) - right conservative state reconstructed at the interface
static void reconstruct_face_muscl_from_char_cache( const Eigen4 &eig, SlopeLimiter limiter,
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


// Store the supplied reconstruction options.
//
// options (in) - parsed options consumed by this module
Reconstruction2D::Reconstruction2D( const ReconstructionOptions &options ) : opt_( options )
{
}


// Compute left/right face averages.
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// gamma (in) - ideal-gas heat-capacity ratio
// ulx (out) - left conservative states on x-normal faces
// urx (out) - right conservative states on x-normal faces
// Face index: i + (nx + 1) * j.
// Throws on invalid input; output buffers may be partially written on reconstruction failure.
void Reconstruction2D::reconstruct_faces_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx ) const
{
    if ( opt_.scheme != ReconstructionScheme::first_order )
        checked_cells_.assign( state.size(), 0 );
    reconstruct_faces_x_impl( state, nx, ny, ng, gamma, ulx, urx );
}


void Reconstruction2D::reconstruct_faces_x_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
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

    // Build one eigensystem per face, then project the selected MUSCL stencil.
    auto reconstruct_face =
        [&]( int i_face, int j_face, ReconstructionScheme scheme, Conservative_vector &ulf, Conservative_vector &urf )
    {
        const int i_r = ng + i_face;
        const int i_l = i_r - 1;
        const int j = ng + j_face;

        const int k_l = idx2( i_l, j, nx_tot );
        const int k_r = idx2( i_r, j, nx_tot );

        double rho0 = 0.0, u0 = 0.0, v0 = 0.0, p0 = 0.0, h0 = 0.0, c0 = 0.0;

        face_state_2d( state[k_l], state[k_r], gamma, rho0, u0, v0, p0, h0, c0,
            checked_cells_[k_l], checked_cells_[k_r] );

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
            reconstruct_face_muscl_from_char_cache( eig, opt_.limiter, wc, ulf, urf );
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
            const int f = i + ( nx + 1 ) * ( j );

            reconstruct_face( i, j, opt_.scheme, ulx[f], urx[f] );
        }
    }
}


// Compute bottom/top face averages.
//
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// gamma (in) - ideal-gas heat-capacity ratio
// uly (out) - left conservative states on y-normal faces
// ury (out) - right conservative states on y-normal faces
//
// Face index: i + nx * j.
// Throws on invalid input; output buffers may be partially written on reconstruction failure.
void Reconstruction2D::reconstruct_faces_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury ) const
{
    if ( opt_.scheme != ReconstructionScheme::first_order )
        checked_cells_.assign( state.size(), 0 );
    reconstruct_faces_y_impl( state, nx, ny, ng, gamma, uly, ury );
}


void Reconstruction2D::reconstruct_faces_y_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
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

        face_state_2d( state[k_b], state[k_t], gamma, rho0, u0, v0, p0, h0, c0,
            checked_cells_[k_b], checked_cells_[k_t] );

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
            reconstruct_face_muscl_from_char_cache( eig, opt_.limiter, wc, ulf, urf );
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
            const int f = i + ( nx ) * j;

            reconstruct_face( i, j, opt_.scheme, uly[f], ury[f] );
        }
    }
}


void Reconstruction2D::reconstruct_faces( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, int dir, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right )
{
    if ( opt_.scheme != ReconstructionScheme::first_order )
        checked_cells_.assign( state.size(), 0 );
    reconstruct_faces_impl( state, nx, ny, ng, gamma, dir, left, right );
}


// Reuse interior validation and successful ghost checks for an immutable x/y reconstruction pair.
void Reconstruction2D::reconstruct_validated_faces_xy( const std::vector<Conservative_vector> &state,
    int nx, int ny, int ng, double gamma, std::vector<Conservative_vector> &left_x,
    std::vector<Conservative_vector> &right_x, std::vector<Conservative_vector> &left_y,
    std::vector<Conservative_vector> &right_y )
{
    const int nx_tot = nx + 2 * ng;
    if ( nx <= 0 || ny <= 0 || ng < 0 ||
         state.size() != static_cast<std::size_t>( nx_tot ) * static_cast<std::size_t>( ny + 2 * ng ) )
        throw std::runtime_error( "Validated reconstruction: invalid grid or state size" );

    if ( opt_.scheme != ReconstructionScheme::first_order )
    {
        checked_cells_.assign( state.size(), 0 );
        for ( int j = ng; j < ng + ny; ++j )
            std::fill_n( checked_cells_.begin() + ng + nx_tot * j, nx, 1 );
    }

    reconstruct_faces_impl( state, nx, ny, ng, gamma, x_direction, left_x, right_x );
    reconstruct_faces_impl( state, nx, ny, ng, gamma, y_direction, left_y, right_y );
}


void Reconstruction2D::reconstruct_faces_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, int dir, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right )
{
    if ( dir != x_direction && dir != y_direction )
    {
        throw std::runtime_error( "Face reconstruction: invalid direction" );
    }

    if ( dir == x_direction )
    {
        reconstruct_faces_x_impl( state, nx, ny, ng, gamma, left, right );
    }
    else
    {
        reconstruct_faces_y_impl( state, nx, ny, ng, gamma, left, right );
    }
}
