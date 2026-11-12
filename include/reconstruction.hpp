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

    // Compute left and right conservative states on all (nx + 1) * ny x-faces.
    //
    // state (in) - conserved variables in the ghosted cell field
    // nx (in) - number of interior cells along x
    // ny (in) - number of interior cells along y
    // ng (in) - number of ghost-cell layers on each side
    // gamma (in) - ideal-gas heat-capacity ratio
    // ulx (out) - left conservative states on x-normal faces
    // urx (out) - right conservative states on x-normal faces
    void reconstruct_faces_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx ) const;

    // Compute left and right conservative states on all nx * (ny + 1) y-faces.
    //
    // state (in) - conserved variables in the ghosted cell field
    // nx (in) - number of interior cells along x
    // ny (in) - number of interior cells along y
    // ng (in) - number of ghost-cell layers on each side
    // gamma (in) - ideal-gas heat-capacity ratio
    // uly (out) - left conservative states on y-normal faces
    // ury (out) - right conservative states on y-normal faces
    void reconstruct_faces_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury ) const;

  private:
    ReconstructionOptions opt_;
};
