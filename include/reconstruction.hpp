#pragma once


#include "structures.hpp"
#include <string>
#include <stdexcept>
#include <vector>


// Build conservative face states with the selected scheme; MUSCL and WENO use characteristic projection.
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
// Fifth-order WENO-JS reconstruction.
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
    // Reuses internal face-average buffers between calls.
    // Throws on invalid input; output buffers may be partially written on reconstruction failure.
    void reconstruct_quadrature( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        int dir, std::vector<FaceQuadratureState> &left, std::vector<FaceQuadratureState> &right );

  private:
    ReconstructionOptions opt_;

    // Reuse normal face averages across directions and time-integration stages.
    std::vector<Conservative_vector> averages_left_;
    std::vector<Conservative_vector> averages_right_;
};
