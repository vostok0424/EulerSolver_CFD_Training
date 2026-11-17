#pragma once


#include "structures.hpp"
#include <string>
#include <stdexcept>
#include <vector>


// Build conservative face states with the selected scheme; MUSCL and WENO-Z5 use characteristic projection.
//
// First-order reconstruction.
//
// Implementation: piecewise-constant conservative cell averages; the numerical flux is selected separately.
//
// Reference: S. K. Godunov (1959), "A difference method for numerical calculation of discontinuous
// solutions of the equations of hydrodynamics", Mat. Sb. (N.S.) 47(89)(3), 271-306.
// https://www.mathnet.ru/eng/sm4873
//
// Second-order MUSCL reconstruction.
//
// Implementation: characteristic piecewise-linear slopes with none, minmod or van Leer limiting.
// Time integration is handled separately from this spatial reconstruction.
//
// Reference: B. van Leer (1979), "Towards the ultimate conservative difference scheme. V.
// A second-order sequel to Godunov's method", J. Comput. Phys. 32(1), 101-136.
// https://doi.org/10.1016/0021-9991(79)90145-1
//
// Fifth-order normal finite-volume WENO-Z uses exponent two and three ghost layers.
// Componentwise transverse WENO-Z supplies point states for four-point Gauss face integration.
// Reference: Borges et al. (2008), https://doi.org/10.1016/j.jcp.2007.11.038
//
class Reconstruction2D
{
  public:
    // Store the supplied reconstruction options.
    //
    // options (in) - parsed options consumed by this module
    explicit Reconstruction2D( const ReconstructionOptions &options );

    // Return the stored reconstruction options.
    const ReconstructionOptions &options() const
    {
        return opt_;
    }

    // Reconstruct face point states, including transverse WENO-Z for wenoZ5.
    // state (in) - ghosted cell averages, including valid corner ghosts
    // nx, ny (in) - interior cell counts
    // ng (in) - ghost width, at least three for WENO-Z5
    // gamma (in) - ideal-gas heat-capacity ratio
    // dir (in) - x_direction or y_direction
    // left, right (out) - face-major point states; x face index i+(nx+1)*j, y index i+nx*j
    // WENO-Z5 fills four nodes; firstOrder and MUSCL fill slot zero.
    // Validates grid, gas model and all supplied ghosted cell averages before reconstruction.
    // Throws on invalid input; output buffers may be partially written on reconstruction failure.
    void reconstruct_quadrature( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
        double gamma, int dir, std::vector<FaceQuadratureState> &left, std::vector<FaceQuadratureState> &right );

  private:
    // Solver validates interior states before applying halos/physical ghost boundaries.
    // Keep this trusted entry point private so standalone public calls always validate their inputs.
    // all_cells_validated (in) - also reuse ghost validation performed for the same immutable stage.
    // skip_west, skip_south (in) - omit non-owned faces, later populated by owner-flux exchange.
    // Transverse halo rows/columns for all owned faces remain available.
    friend class Solver;
    void reconstruct_validated_quadrature_xy( const std::vector<Conservative_vector> &state, int nx, int ny,
        int ng, double gamma, std::vector<FaceQuadratureState> &left_x,
        std::vector<FaceQuadratureState> &right_x, std::vector<FaceQuadratureState> &left_y,
        std::vector<FaceQuadratureState> &right_y, bool all_cells_validated = false, bool skip_west = false, bool skip_south = false );

    // Implementation helpers share checks only within one public call or one validated x/y pair.
    void reconstruct_faces_x_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
        double gamma, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right, int extension = 0, int first_face = 0 ) const;
    void reconstruct_faces_y_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
        double gamma, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right, int extension = 0, int first_face = 0 ) const;


    void reconstruct_quadrature_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
        double gamma, int dir, std::vector<FaceQuadratureState> &left, std::vector<FaceQuadratureState> &right, int first_face = 0 );

    // Reused normal face averages, including two transverse halo layers for WENO-Z5.
    std::vector<Conservative_vector> averages_left_;
    std::vector<Conservative_vector> averages_right_;

    ReconstructionOptions opt_;

    // Reset for each standalone call or validated x/y pair; never retain checks across state updates.
    mutable std::vector<unsigned char> checked_cells_;

};


// Return the required ghost width: one for first order, two for MUSCL and three for WENO-Z5.
//
// s (in) - selected reconstruction scheme
int required_ghost_cells( ReconstructionScheme s );
