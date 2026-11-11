#pragma once


#include "structures.hpp"
#include "cfg.hpp"
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
    std::vector<Conservative_vector> left_state;
    std::vector<Conservative_vector> right_state;
    std::vector<Flux_vector> flux;

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
    // cfg, mp (in) - case settings and MPI helper; mp must outlive the solver.
    Solver( const Cfg &cfg, const MpiParallel &mp );

    // Advance the initialized solution and process scheduled and final output.
    void run();

  private:
    const MpiParallel &mp_;
    Subdomain2D sub_{};

    GridBlock2DInfo grid_{};

    std::vector<Conservative_vector> conservative_field_;
    std::vector<Conservative_vector> rhs_;

    // Packed communication storage in rho, rho*u, rho*v, rho*E order.
    mutable std::vector<double> halo_buffer_;

    FaceBuffers2D faces_{};

    BoundaryConditions2D bc_;

    Reconstruction2D recon_;

    FluxScheme flux_scheme_ = FluxScheme::hllc;
    std::string flux_name_;
    std::unique_ptr<TimeIntegrator> ti_;

    double gamma_{};
    double cfl_{};
    double final_time_{};
    int output_every_{};
    bool write_final_{};
    std::string out_prefix_{};

    VTKDataOutputType vtk_data_output_type_{ VTKDataOutputType::point };

    // Flatten a ghosted cell index; interior cells start at (grid_.ng, grid_.ng).
    int idx( int i, int j ) const
    {
        return i + grid_.nx_tot * j;
    }

    int idx_face_x( int i, int j ) const
    {
        return i + ( grid_.nx + 1 ) * j;
    }

    int idx_face_y( int i, int j ) const
    {
        return i + grid_.nx * j;
    }

    // Exchange MPI halos and apply physical boundary conditions before reconstruction.
    void apply_bc( std::vector<Conservative_vector> &state ) const;

    // Check interior states and throw with cell coordinates on failure.
    void validate_interior( const std::vector<Conservative_vector> &state ) const;

    // Compute the CFL time step using the global maximum acoustic signal rate.
    double compute_dt( const std::vector<Conservative_vector> &state ) const;

    // Compute the flux divergence; the input ghost cells must already be valid.
    void build_rhs( const std::vector<Conservative_vector> &state, std::vector<Conservative_vector> &rhs );

    // Check whether the step meets the configured output interval.
    bool should_write_step_output( int step ) const;

    // Refresh ghost cells and process scheduled output.
    void process_regular_output_phase( int step, double t );

    // Write scheduled merged VTK output.
    void write_output( int step, double t ) const;

    // Write the final merged VTK output when enabled.
    void write_final_output( double t ) const;
};
