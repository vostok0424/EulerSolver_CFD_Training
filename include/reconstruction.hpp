#pragma once


#include "structures.hpp"
#include <string>
#include <stdexcept>
#include <vector>


// Build conservative face states with the selected scheme; MUSCL and WENO use characteristic projection.
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

    // Compute left/right face averages; optional transverse extension adds halo rows.
    //
    // state (in) - conserved variables in the ghosted cell field
    // nx (in) - number of interior cells along x
    // ny (in) - number of interior cells along y
    // ng (in) - number of ghost-cell layers on each side
    // gamma (in) - ideal-gas heat-capacity ratio
    // ulx (out) - left conservative states on x-normal faces
    // urx (out) - right conservative states on x-normal faces
    // transverse_extension (in) - extra transverse layers, from zero to ng; nonzero requires WENO5
    //
    // Face index: i + (nx + 1) * (j + transverse_extension).
    // Throws on invalid input; output buffers may be partially written on reconstruction failure.
    void reconstruct_faces_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx,
        int transverse_extension = 0 ) const;

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
    void reconstruct_faces_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury,
        int transverse_extension = 0 ) const;

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
    // Throws on invalid input; output buffers may be partially written on reconstruction failure.
    void reconstruct_quadrature( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        int dir, std::vector<FaceQuadratureState> &left, std::vector<FaceQuadratureState> &right ) const;

  private:
    ReconstructionOptions opt_;
};
