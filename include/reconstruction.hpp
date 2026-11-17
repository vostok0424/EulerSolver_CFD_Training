#pragma once


#include "structures.hpp"
#include <string>
#include <stdexcept>
#include <vector>


// Build conservative face states with the selected scheme; MUSCL uses characteristic projection.
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

    // Compute left/right face averages.
    //
    // state (in) - conserved variables in the ghosted cell field
    // nx (in) - number of interior cells along x
    // ny (in) - number of interior cells along y
    // ng (in) - number of ghost-cell layers on each side
    // gamma (in) - ideal-gas heat-capacity ratio
    // ulx (out) - left conservative states on x-normal faces
    // urx (out) - right conservative states on x-normal faces
    //
    // Face index: i + (nx + 1) * j.
    // Throws on invalid input; output buffers may be partially written on reconstruction failure.
    void reconstruct_faces_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx ) const;

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
    void reconstruct_faces_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury ) const;

    // Reconstruct left/right face states for the selected normal direction.
    // state (in) - ghosted conservative field
    // nx, ny (in) - interior cell counts
    // ng (in) - ghost width required by the selected scheme
    // gamma (in) - ideal-gas heat-capacity ratio
    // dir (in) - x_direction or y_direction
    // left, right (out) - face states, indexed by i + (nx + 1) * j for x, i + nx * j for y
    // Throws on invalid input or reconstruction failure.
    void reconstruct_faces( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        int dir, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right );

  private:
    // Solver validates interior states before applying halos/physical ghost boundaries.
    // Keep this trusted entry point private so standalone public calls always validate their inputs.
    friend class Solver;
    void reconstruct_validated_faces_xy( const std::vector<Conservative_vector> &state, int nx, int ny,
        int ng, double gamma, std::vector<Conservative_vector> &left_x,
        std::vector<Conservative_vector> &right_x, std::vector<Conservative_vector> &left_y,
        std::vector<Conservative_vector> &right_y );

    // Implementation helpers share checks only within one public call or one validated x/y pair.
    void reconstruct_faces_x_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
        double gamma, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right ) const;
    void reconstruct_faces_y_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
        double gamma, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right ) const;
    void reconstruct_faces_impl( const std::vector<Conservative_vector> &state, int nx, int ny, int ng,
        double gamma, int dir, std::vector<Conservative_vector> &left, std::vector<Conservative_vector> &right );

    ReconstructionOptions opt_;

    // Reset for each standalone call or validated x/y pair; never retain checks across state updates.
    mutable std::vector<unsigned char> checked_cells_;

};
