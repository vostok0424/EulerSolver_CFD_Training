#pragma once


#include "structures.hpp"
#include "read_parameters.hpp"
#include "flux.hpp"
#include "time_integrator.hpp"
#include "mpi_parallel.hpp"
#include "reconstruction.hpp"
#include "setFields.hpp"
#include "io.hpp"
#include "boundary.hpp"

#include <memory>
#include <string>
#include <vector>


// Store reconstructed left/right states and numerical fluxes for one face-normal direction.
struct DirectionalFaceBuffers2D
{
    std::vector<FaceQuadratureState> left_state;
    std::vector<FaceQuadratureState> right_state;
    std::vector<Flux_vector> flux;

    // Resize the face workspace for the supplied face count or interior mesh dimensions.
    //
    // n_faces (in) - number of faces to allocate
    void resize( std::size_t n_faces )
    {
        left_state.resize( n_faces );
        right_state.resize( n_faces );
        flux.resize( n_faces );
    }
};


// Store (nx + 1)*ny x-faces and nx*(ny + 1) y-faces.
struct FaceBuffers2D
{
    DirectionalFaceBuffers2D x;
    DirectionalFaceBuffers2D y;

    // Resize the face workspace for the supplied face count or interior mesh dimensions.
    //
    // nx (in) - number of interior cells along x
    // ny (in) - number of interior cells along y
    void resize( int nx, int ny )
    {
        x.resize( static_cast<std::size_t>( nx + 1 ) * static_cast<std::size_t>( ny ) );
        y.resize( static_cast<std::size_t>( nx ) * static_cast<std::size_t>( ny + 1 ) );
    }
};


// Own the local ghosted state and advance the 2D Euler equations with MPI halo exchange.
class Solver
{
  public:
    // Construct the local mesh, allocate workspaces and initialize the conservative field.
    //
    // parameters (in) - typed computational experiment parameters
    // mp (in) - MPI topology and communication helper; must outlive the solver
    Solver( const CaseParameters &parameters, const MpiParallel &mp );

    // Advance the initialized solution and process scheduled and final output.
    void run();

  private:
    const MpiParallel &mp_;
    Subdomain2D sub_{};

    GridBlock2DInfo grid_{};

    std::vector<Conservative_vector> conservative_field_;

    FaceBuffers2D faces_{};

    // Current immutable-stage cell data; interiors are filled by validation, ghosts after apply_bc.
    std::vector<EvaluatedState> stage_flow_;
    // Physical fluxes of immutable cell averages; shared by the two neighboring faces.
    std::vector<Flux_vector> stage_physical_x_, stage_physical_y_;
    std::vector<double> flux_send_, flux_receive_;

    BoundaryConditions2D bc_;

    Reconstruction2D recon_;

    FluxScheme flux_scheme_ = FluxScheme::hllc;
    std::string flux_name_;
    std::unique_ptr<TimeIntegrator> ti_;

    bool positivity_preserving_ = true;
    double density_floor_ = POSITIVITY_DENSITY_FLOOR;
    double pressure_floor_ = POSITIVITY_PRESSURE_FLOOR;

    double gamma_{};
    double cfl_{};
    double final_time_{};
    int output_every_{};
    double output_time_interval_{};
    bool write_final_{};
    std::string out_prefix_{};

    VTKDataOutputType vtk_data_output_type_{ VTKDataOutputType::point };

    // Map a ghosted cell index to its row-major storage offset.
    //
    // i (in) - cell or face index along x
    // j (in) - cell or face index along y
    int idx( int i, int j ) const
    {
        return i + grid_.nx_tot * j;
    }

    // Map an x-normal face index to its row-major storage offset.
    //
    // i (in) - cell or face index along x
    // j (in) - cell or face index along y
    int idx_face_x( int i, int j ) const
    {
        return i + ( grid_.nx + 1 ) * j;
    }

    // Map a y-normal face index to its row-major storage offset.
    //
    // i (in) - cell or face index along x
    // j (in) - cell or face index along y
    int idx_face_y( int i, int j ) const
    {
        return i + grid_.nx * j;
    }

    // Exchange MPI halos and apply physical boundary conditions before reconstruction.
    //
    // state (in/out) - conserved variables in the ghosted cell field
    void apply_bc( std::vector<Conservative_vector> &state ) const;

    // Check interior states and throw with cell coordinates on failure.
    //
    // state (in) - conserved variables in the ghosted cell field
    // max_s (out, optional) - local maximum spectral rate for the same state
    void validate_interior( const std::vector<Conservative_vector> &state, double *max_s = nullptr );

    // Compute the CFL time step from the spectral rate collected during state validation.
    //
    // max_s (in) - local maximum spectral rate for the current state
    double compute_dt( double max_s ) const;

    // Compute the flux divergence; the interior must be validated and ghost cells refreshed.
    //
    // state (in) - conserved variables in the ghosted cell field
    // dt (in) - actual forward-Euler substep used by the positivity limiter
    // ax, ay (in) - global wave-speed bounds for this same refreshed stage, ignored when positivity is off
    // rhs (out) - conservative residual on the ghosted cell layout
    void build_rhs( const std::vector<Conservative_vector> &state, std::vector<Conservative_vector> &rhs,
        double dt, double ax, double ay );

    // Obtain global direction wave-speed bounds from the refreshed ghosted stage field.
    // state (in) - finite admissible cell averages including boundary ghosts
    // ax, ay (out) - global maximum |u|+c and |v|+c
    void positivity_wave_speeds( const std::vector<Conservative_vector> &state, double &ax, double &ay );

    // Exchange final interface fluxes: west/south cells own east/north faces.
    void synchronize_face_fluxes();

    // Check whether the step meets the configured output interval.
    //
    // step (in) - current time-step index
    bool should_write_step_output( int step ) const;

    // Process scheduled output using interior cell values only.
    //
    // step (in) - current time-step index
    // t (in) - current simulation time
    // time_output (in) - the clock has reached a scheduled physical output time
    void process_regular_output_phase( int step, double t, bool time_output = false );

    // Write scheduled merged VTK output.
    //
    // t (in) - current simulation time
    void write_output( double t ) const;

    // Write the final merged VTK output when enabled.
    //
    // t (in) - current simulation time
    void write_final_output( double t ) const;
};
