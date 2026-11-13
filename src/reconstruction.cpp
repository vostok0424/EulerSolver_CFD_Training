#include "common_functions.hpp"


#include "reconstruction.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <array>


// Four-point Gauss nodes in the normalized transverse coordinate [-1/2, 1/2].
const int face_quadrature_point_count = 4;
const double face_quadrature_nodes[4] = { -0.43056815579702628761, -0.16999052179242813240, 0.16999052179242813240, 0.43056815579702628761 };


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


// 3. Fifth-order WENO reconstruction.
//
// Implementation: fifth-order JS weights with exponent two; normal reconstruction is characteristic.
// The transverse finite-volume extension evaluates quadratic candidates at four Gauss nodes with
// node-dependent positive optimal weights and the same JS smoothness indicators.
//
// References: G.-S. Jiang and C.-W. Shu (1996), "Efficient Implementation of Weighted ENO Schemes",
// J. Comput. Phys. 126(1), 202-228. https://doi.org/10.1006/jcph.1996.0130
// C.-W. Shu (1997), "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes
// for Hyperbolic Conservation Laws", NASA/CR-97-206253, ICASE Report 97-65 (finite-volume reconstruction).
// https://ntrs.nasa.gov/citations/19980007543

// Compute the inverse-square factor used in Jiang-Shu WENO weights.
//
// x (in) - scalar argument of the reconstruction helper
static double inv_pow_weno( double x )
{
    return 1.0 / ( x * x );
}


// Compute the Jiang-Shu smoothness indicators for three quadratic substencils.
//
// qim2 (in) - scalar cell average two cells to the left
// qim1 (in) - scalar cell average one cell to the left
// qi (in) - central scalar cell average
// qip1 (in) - scalar cell average one cell to the right
// qip2 (in) - scalar cell average two cells to the right
// b0 (out) - smoothness indicator on cells [-2, -1, 0]
// b1 (out) - smoothness indicator on cells [-1, 0, 1]
// b2 (out) - smoothness indicator on cells [0, 1, 2]
static void weno5_indicators( double qim2, double qim1, double qi, double qip1, double qip2, double &b0, double &b1,
    double &b2 )
{
    // Compute the Jiang-Shu smoothness indicators for the three substencils.
    const double d0 = qim2 - 2.0 * qim1 + qi;
    const double d1 = qim1 - 2.0 * qi + qip1;
    const double d2 = qi - 2.0 * qip1 + qip2;

    b0 = ( 13.0 / 12.0 ) * d0 * d0 + 0.25 * ( qim2 - 4.0 * qim1 + 3.0 * qi ) * ( qim2 - 4.0 * qim1 + 3.0 * qi );
    b1 = ( 13.0 / 12.0 ) * d1 * d1 + 0.25 * ( qim1 - qip1 ) * ( qim1 - qip1 );
    b2 = ( 13.0 / 12.0 ) * d2 * d2 + 0.25 * ( 3.0 * qi - 4.0 * qip1 + qip2 ) * ( 3.0 * qi - 4.0 * qip1 + qip2 );
}


// Compute and normalize the three Jiang-Shu nonlinear weights using cached factors.
//
// factor0 (in) - 1 / (eps + b0)^2 for the first substencil
// factor1 (in) - 1 / (eps + b1)^2 for the second substencil
// factor2 (in) - 1 / (eps + b2)^2 for the third substencil
// g0 (in) - first optimal linear weight
// g1 (in) - second optimal linear weight
// g2 (in) - third optimal linear weight
// w0 (out) - first normalized nonlinear weight
// w1 (out) - second normalized nonlinear weight
// w2 (out) - third normalized nonlinear weight
static void weno5_weights( double factor0, double factor1, double factor2, double g0, double g1, double g2,
    double &w0, double &w1, double &w2 )
{
    // Each unnormalized weight is its linear weight divided by (epsilon + beta) squared.
    const double a0 = g0 * factor0;
    const double a1 = g1 * factor1;
    const double a2 = g2 * factor2;
    // Normalize the weights so their sum is one.
    const double asum = a0 + a1 + a2;

    w0 = a0 / asum;
    w1 = a1 / asum;
    w2 = a2 / asum;
}


// Compute the left face value from the five-cell stencil centered on the left cell.
//
// qim2 (in) - scalar stencil value two cells to the left
// qim1 (in) - scalar stencil value one cell to the left
// qi (in) - scalar stencil value at the center
// qip1 (in) - scalar stencil value one cell to the right
// qip2 (in) - scalar stencil value two cells to the right
// eps (in) - positive Jiang-Shu weight regularization parameter
static double weno5_left( double qim2, double qim1, double qi, double qip1, double qip2, double eps )
{
    // Evaluate the three third-order candidate polynomials at the face.
    const double p0 = ( 1.0 / 3.0 ) * qim2 - ( 7.0 / 6.0 ) * qim1 + ( 11.0 / 6.0 ) * qi;
    const double p1 = ( -1.0 / 6.0 ) * qim1 + ( 5.0 / 6.0 ) * qi + ( 1.0 / 3.0 ) * qip1;
    const double p2 = ( 1.0 / 3.0 ) * qi + ( 5.0 / 6.0 ) * qip1 - ( 1.0 / 6.0 ) * qip2;

    double b0 = 0.0, b1 = 0.0, b2 = 0.0;

    weno5_indicators( qim2, qim1, qi, qip1, qip2, b0, b1, b2 );

    // Use the fifth-order linear weights before applying smoothness-dependent weighting.
    const double g0 = 0.1;
    const double g1 = 0.6;
    const double g2 = 0.3;

    double w0 = 0.0, w1 = 0.0, w2 = 0.0;

    const double factor0 = inv_pow_weno( eps + b0 );
    const double factor1 = inv_pow_weno( eps + b1 );
    const double factor2 = inv_pow_weno( eps + b2 );

    weno5_weights( factor0, factor1, factor2, g0, g1, g2, w0, w1, w2 );

    return w0 * p0 + w1 * p1 + w2 * p2;
}


// Compute the right face value by reversing the five-cell stencil.
//
// qim2 (in) - scalar stencil value two cells to the left
// qim1 (in) - scalar stencil value one cell to the left
// qi (in) - scalar stencil value at the center
// qip1 (in) - scalar stencil value one cell to the right
// qip2 (in) - scalar stencil value two cells to the right
// eps (in) - positive Jiang-Shu weight regularization parameter
static double weno5_right( double qim2, double qim1, double qi, double qip1, double qip2, double eps )
{
    return weno5_left( qip2, qip1, qi, qim1, qim2, eps );
}


// Reconstruct both face states from a six-cell characteristic stencil using WENO5.
//
// eig (in) - characteristic eigenvectors at the face reference state
// eps (in) - positive Jiang-Shu weight regularization parameter
// wc (in) - cell stencil of characteristic components ordered along the face normal
// ulf (out) - left conservative state reconstructed at the interface
// urf (out) - right conservative state reconstructed at the interface
static void reconstruct_face_weno5_from_char_cache( const Eigen4 &eig, double eps,
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


// Store the supplied reconstruction options.
//
// options (in) - parsed options consumed by this module
Reconstruction2D::Reconstruction2D( const ReconstructionOptions &options ) : opt_( options )
{
}


// Compute left/right face averages; optional transverse extension adds halo rows.
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// gamma (in) - ideal-gas heat-capacity ratio
// ulx (out) - left conservative states on x-normal faces
// urx (out) - right conservative states on x-normal faces
// transverse_extension (in) - extra transverse layers, from zero to ng; nonzero requires WENO5
// Face index: i + (nx + 1) * (j + transverse_extension).
// Throws on invalid input; output buffers may be partially written on reconstruction failure.
void Reconstruction2D::reconstruct_faces_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx,
    int transverse_extension ) const
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

    if ( transverse_extension < 0 || transverse_extension > ng ||
         ( transverse_extension != 0 && opt_.scheme != ReconstructionScheme::weno5 ) )
    {
        throw std::runtime_error( "Invalid transverse reconstruction extension" );
    }

    const size_t n_face_x = static_cast<size_t>( nx + 1 ) * static_cast<size_t>( ny + 2 * transverse_extension );

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

    checked_cells_.assign( state.size(), 0 );

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

    for ( int j = -transverse_extension; j < ny + transverse_extension; ++j )
    {
        for ( int i = 0; i < nx + 1; ++i )
        {
            const int f = i + ( nx + 1 ) * ( j + transverse_extension );

            reconstruct_face( i, j, opt_.scheme, ulx[f], urx[f] );
        }
    }
}


// Compute bottom/top face averages; optional transverse extension adds halo columns.
//
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// gamma (in) - ideal-gas heat-capacity ratio
// uly (out) - left conservative states on y-normal faces
// ury (out) - right conservative states on y-normal faces
// transverse_extension (in) - extra transverse layers, from zero to ng; nonzero requires WENO5
//
// Face index: i + transverse_extension + (nx + 2 * transverse_extension) * j.
// Throws on invalid input; output buffers may be partially written on reconstruction failure.
void Reconstruction2D::reconstruct_faces_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury,
    int transverse_extension ) const
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

    if ( transverse_extension < 0 || transverse_extension > ng ||
         ( transverse_extension != 0 && opt_.scheme != ReconstructionScheme::weno5 ) )
    {
        throw std::runtime_error( "Invalid transverse reconstruction extension" );
    }

    const size_t n_face_y = static_cast<size_t>( nx + 2 * transverse_extension ) * static_cast<size_t>( ny + 1 );

    if ( uly.size() != n_face_y )
        uly.resize( n_face_y );
    if ( ury.size() != n_face_y )
        ury.resize( n_face_y );

    if ( opt_.scheme == ReconstructionScheme::first_order )
    {
        reconstruct_first_order_y( state, nx, ny, ng, nx_tot, uly, ury );
        return;
    }

    checked_cells_.assign( state.size(), 0 );

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
        for ( int i = -transverse_extension; i < nx + transverse_extension; ++i )
        {
            const int f = i + transverse_extension + ( nx + 2 * transverse_extension ) * j;

            reconstruct_face( i, j, opt_.scheme, uly[f], ury[f] );
        }
    }
}


// Reconstruct five transverse face averages to four Gauss points using WENO-JS.
// Uses the JS weighting and finite-volume reconstruction references listed in Section 3 above.
//
// q (in) - five face averages, with the target face at index two
// eps (in) - positive regularization for the Jiang-Shu nonlinear weights
// output (out) - point values in increasing transverse Gauss-node order
static void transverse_weno5( const std::array<double, 5> &q, double eps, std::array<double, face_quadrature_point_count> &output )
{
    // Optimal weights match the five-cell quartic at each Gauss point.
    // All are positive; negative nodes use the reflected weights of positive nodes.
    const double linear_weights[4][3] =
    {
        { 0.26584209747783194, 0.61125049003224860, 0.12290741248991946 },
        { 0.12816415843550585, 0.52196914985035634, 0.34986669171413781 },
        { 0.34986669171413781, 0.52196914985035634, 0.12816415843550585 },
        { 0.12290741248991946, 0.61125049003224860, 0.26584209747783194 }
    };
    
    const double a = q[0] - q[2];
    const double b = q[1] - q[2];
    const double c = q[3] - q[2];
    const double d = q[4] - q[2];
    double b0 = 0.0, b1 = 0.0, b2 = 0.0;

    weno5_indicators( q[0], q[1], q[2], q[3], q[4], b0, b1, b2 );

    // These factors depend on the stencil, so compute them once for all Gauss nodes.
    const double factor0 = inv_pow_weno( eps + b0 );
    const double factor1 = inv_pow_weno( eps + b1 );
    const double factor2 = inv_pow_weno( eps + b2 );

    for ( int n = 0; n < face_quadrature_point_count; ++n )
    {
        const double x = face_quadrature_nodes[n];
        // Quadratic candidates preserve the central average; subtract it before evaluation.
        const double p0 = -a / 24.0 + b / 12.0 + x * ( a / 2.0 - 2.0 * b + x * ( a / 2.0 - b ) );
        const double p1 = -( b + c ) / 24.0 + x * ( ( c - b ) / 2.0 + x * ( b + c ) / 2.0 );
        const double p2 = c / 12.0 - d / 24.0 + x * ( 2.0 * c - d / 2.0 + x * ( -c + d / 2.0 ) );
        double w0 = 0.0, w1 = 0.0, w2 = 0.0;

        weno5_weights( factor0, factor1, factor2, linear_weights[n][0], linear_weights[n][1], linear_weights[n][2],
            w0, w1, w2 );
        output[n] = q[2] + w0 * p0 + w1 * p1 + w2 * p2;
    }
}


// Reconstruct WENO5 face averages to transverse Gauss points with componentwise WENO-JS.
//
// state (in) - conserved cell averages, including valid diagonal ghost cells
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost layers; WENO5 requires at least three
// gamma (in) - ideal-gas heat-capacity ratio
// dir (in) - face-normal direction: x_direction or y_direction
// left (out) - left/bottom point states, indexed by face then transverse Gauss node
// right (out) - right/top point states, indexed by face then transverse Gauss node
//
// Face index is i + (nx + 1) * j for x and i + nx * j for y.
// WENO5 fills all four slots in increasing transverse coordinate; other schemes fill slot zero.
// Reuses internal face-average buffers between calls.
// Throws on invalid input; output buffers may be partially written on reconstruction failure.
void Reconstruction2D::reconstruct_quadrature( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
    double gamma, int dir, std::vector<FaceQuadratureState> &left, std::vector<FaceQuadratureState> &right )
{
    if ( dir != x_direction && dir != y_direction )
    {
        throw std::runtime_error( "Quadrature reconstruction: invalid direction" );
    }

    const bool high_order = opt_.scheme == ReconstructionScheme::weno5;
    const int extension = high_order ? 2 : 0;

    if ( dir == x_direction )
    {
        reconstruct_faces_x( state, nx, ny, ng, gamma, averages_left_, averages_right_, extension );
    }
    else
    {
        reconstruct_faces_y( state, nx, ny, ng, gamma, averages_left_, averages_right_, extension );
    }

    const int ni = dir == x_direction ? nx + 1 : nx;
    const int nj = dir == x_direction ? ny : ny + 1;

    left.resize( static_cast<std::size_t>( ni ) * nj );
    right.resize( left.size() );

    for ( int side = 0; side < 2; ++side )
    {
        const auto &averages = side == 0 ? averages_left_ : averages_right_;
        auto &points = side == 0 ? left : right;
        
        for ( int j = 0; j < nj; ++j )
        {
            for ( int i = 0; i < ni; ++i )
            {
                const int f = i + ni * j;
                if ( !high_order )
                {
                    points[f][0] = averages[f];
                    continue;
                }
                
                double components[5][conservative_variable_count];

                for ( int k = 0; k < 5; ++k )
                {
                    const int index = dir == x_direction ? i + ni * ( j + k ) : i + k + ( nx + 4 ) * j;
                    pack_state( averages[index], components[k] );
                }

                std::array<std::array<double, conservative_variable_count>, face_quadrature_point_count> values{};

                for ( std::size_t m = 0; m < conservative_variable_count; ++m )
                {
                    std::array<double, 5> stencil{};
                    
                    for ( int k = 0; k < 5; ++k )
                    {
                        stencil[k] = components[k][m];
                    }
                    
                    std::array<double, face_quadrature_point_count> reconstructed{};

                    transverse_weno5( stencil, opt_.eps, reconstructed );
                    
                    for ( int q = 0; q < face_quadrature_point_count; ++q )
                    {
                        values[q][m] = reconstructed[q];
                    }
                }
                
                for ( int q = 0; q < face_quadrature_point_count; ++q )
                {
                    points[f][q] = unpack_state( values[q].data() );
                }
            }
        }
    }
}
