#pragma once


#include "cfg.hpp"
#include "structures.hpp"
#include <string>
#include <stdexcept>
#include <vector>


// Default regularization in the Jiang-Shu weights; this is not a physical-state lower bound.
inline constexpr double default_weno_epsilon = 1e-12;


// Select first-order constant, second-order MUSCL or fifth-order Jiang-Shu WENO reconstruction.
enum class ReconstructionScheme
{
    first_order,
    muscl,
    weno5
};


// Select the MUSCL slope: centered (none), minmod or van Leer.
enum class SlopeLimiter
{
    none,
    minmod,
    van_leer
};


// Return the required ghost width: one for first order, two for MUSCL and three for WENO5.
inline int required_ghost_cells( ReconstructionScheme s )
{
    switch ( s )
    {
    case ReconstructionScheme::first_order:
        return 1;
    case ReconstructionScheme::muscl:
        return 2;
    case ReconstructionScheme::weno5:
        return 3;
    }

    throw std::runtime_error( "Invalid reconstruction scheme enum" );
}


// Check whether the selected scheme is MUSCL or WENO5.
inline bool is_high_order( ReconstructionScheme s )
{
    return s == ReconstructionScheme::muscl || s == ReconstructionScheme::weno5;
}


// Store the scheme, MUSCL limiter and positive WENO weight epsilon.
struct ReconstructionOptions
{
    ReconstructionScheme scheme = ReconstructionScheme::first_order;

    // Slope limiter used only by MUSCL.
    SlopeLimiter limiter = SlopeLimiter::van_leer;

    // Positive epsilon in the Jiang-Shu inverse-square weights; used only by WENO5.
    double eps = default_weno_epsilon;

    // Return the ghost width required by the configured scheme.
    int required_ng() const
    {
        return required_ghost_cells( scheme );
    }
};


// Read the reconstruction scheme, MUSCL limiter and positive WENO epsilon.
ReconstructionOptions read_reconstruction_options( const Cfg &cfg );


// Build conservative face states with the selected scheme; MUSCL and WENO use characteristic projection.
class Reconstruction2D
{
  public:
    explicit Reconstruction2D( const Cfg &cfg );

    // Return the stored reconstruction options.
    const ReconstructionOptions &options() const
    {
        return opt_;
    }

    // state (in) - row-major cell field with valid ghosts; gamma is the ideal-gas ratio.
    // ulx, urx (out) - left/right states on (nx + 1)*ny x-faces.
    void reconstruct_faces_x( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &ulx, std::vector<Conservative_vector> &urx ) const;

    // state (in) - row-major cell field with valid ghosts; gamma is the ideal-gas ratio.
    // uly, ury (out) - left/right states on nx*(ny + 1) y-faces.
    void reconstruct_faces_y( const std::vector<Conservative_vector> &state, int nx, int ny, int ng, double gamma,
        std::vector<Conservative_vector> &uly, std::vector<Conservative_vector> &ury ) const;

  private:
    ReconstructionOptions opt_;
};
