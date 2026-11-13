#include "common_functions.hpp"


#include "solver.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <type_traits>

constexpr const char *k_output_dir_2d = "solution";

// Return the VTK label for the selected data location.
//
// output_type (in) - selected VTK point or cell data location
static const char *vtk_data_output_type_name( VTKDataOutputType output_type )
{
    switch ( output_type )
    {
    case VTKDataOutputType::point:
        return "POINT_DATA";
    case VTKDataOutputType::cell:
        return "CELL_DATA";
    }

    return "UNKNOWN";
}


// Create the solution directory on root, then synchronize all ranks.
//
// mp (in) - MPI topology and communication helper; must outlive the solver
static void ensure_output_directory_exists( const MpiParallel &mp )
{
    if ( mp.is_root() )
    {
        std::filesystem::create_directories( k_output_dir_2d );
    }

    mp.barrier();
}


// Build the scheduled output path from the prefix and physical time.
//
// out_prefix (in) - filename prefix for solution output
// t (in) - current simulation time
static std::string make_step_output_path( const std::string &out_prefix, double t )
{
    std::ostringstream t_ss;

    t_ss << std::fixed << std::setprecision( 6 ) << t;

    const std::filesystem::path out_dir = k_output_dir_2d;
    return ( out_dir / ( out_prefix + "_t=" + t_ss.str() + ".vtk" ) ).string();
}


// Build the final output path from the prefix and time.
//
// out_prefix (in) - filename prefix for solution output
// t (in) - current simulation time
static std::string make_final_output_path( const std::string &out_prefix, double t )
{
    std::ostringstream t_ss;

    t_ss << std::fixed << std::setprecision( 6 ) << t;

    const std::filesystem::path out_dir = k_output_dir_2d;
    return ( out_dir / ( out_prefix + "_final_t=" + t_ss.str() + ".vtk" ) ).string();
}


// Write a merged MPI snapshot using the selected point or cell data location.
//
// mp (in) - MPI topology and communication helper; must outlive the solver
// path (in) - destination VTK file path
// label (in) - output-phase label used in the log
// t (in) - current simulation time
// state (in) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// i_beg (in) - global x index of the first local interior cell
// j_beg (in) - global y index of the first local interior cell
// nx_global (in) - global number of interior cells along x
// ny_global (in) - global number of interior cells along y
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
// data_output_type (in) - selected VTK point or cell data location
static void write_merged_vtk_file_2d( const MpiParallel &mp, const std::string &path, const std::string &label,
    double t, const std::vector<Conservative_vector> &state, int nx, int ny, int ng, int i_beg, int j_beg,
    int nx_global, int ny_global, double x0, double x1, double y0, double y1, double gamma,
    VTKDataOutputType data_output_type )
{
    if ( mp.is_root() )
    {
        std::cout << "[2D] Writing merged " << label << " " << path << " at t=" << t << " as "
                  << vtk_data_output_type_name( data_output_type ) << "\n";
    }

    switch ( data_output_type )
    {
    case VTKDataOutputType::point:
        write_vtk_2d_point_data_gather_mpi( path, state, nx, ny, ng, i_beg, j_beg, nx_global, ny_global, x0, x1, y0, y1,
            gamma, mp.cart_comm() );
        break;

    case VTKDataOutputType::cell:
        write_vtk_2d_cell_data_gather_mpi( path, state, nx, ny, ng, i_beg, j_beg, nx_global, ny_global, x0, x1, y0, y1,
            gamma, mp.cart_comm() );
        break;
    }
}


// Construct the local mesh, allocate workspaces and initialize the conservative field.
//
// parameters (in) - typed computational experiment parameters
// mp (in) - MPI topology and communication helper; must outlive the solver
Solver::Solver( const CaseParameters &parameters, const MpiParallel &mp )
    : mp_( mp ), recon_( parameters.reconstruction )
{
    // 1. Set the global mesh and verify the selected reconstruction stencil fits the ghost width.
    grid_.nx_global = parameters.grid.nx;
    grid_.ny_global = parameters.grid.ny;

    grid_.ng = parameters.grid.ng;

    if ( grid_.ng < 1 )
    {
        throw std::runtime_error( "ng must be >=1 for 2D solver" );
    }

    const int ng_req = required_ghost_cells( recon_.options().scheme );

    if ( grid_.ng < ng_req )
    {
        throw std::runtime_error( "ng is too small for the selected reconstruction.scheme" );
    }

    grid_.x0 = parameters.grid.x0;
    grid_.x1 = parameters.grid.x1;
    grid_.y0 = parameters.grid.y0;
    grid_.y1 = parameters.grid.y1;

    if ( grid_.nx_global <= 0 || grid_.ny_global <= 0 || grid_.x1 <= grid_.x0 || grid_.y1 <= grid_.y0 ||
         !std::isfinite( grid_.dx() ) || !std::isfinite( grid_.dy() ) || grid_.dx() <= 0.0 || grid_.dy() <= 0.0 )
        throw std::runtime_error( "Invalid grid dimensions or physical extents" );

    // 2. Find this rank's global cell range and local physical bounds.
    sub_ = mp_.decompose( grid_.nx_global, grid_.ny_global );
    grid_.i_beg = sub_.i_beg();
    grid_.j_beg = sub_.j_beg();
    grid_.nx = sub_.nx();
    grid_.ny = sub_.ny();

    const double dx_g = grid_.dx();
    const double dy_g = grid_.dy();

    const double x0_loc = grid_.x0 + static_cast<double>( grid_.i_beg ) * dx_g;
    const double x1_loc = grid_.x0 + static_cast<double>( grid_.i_beg + grid_.nx ) * dx_g;
    const double y0_loc = grid_.y0 + static_cast<double>( grid_.j_beg ) * dy_g;
    const double y1_loc = grid_.y0 + static_cast<double>( grid_.j_beg + grid_.ny ) * dy_g;

    // 3. Set the gas model, time controls and output settings.
    gamma_ = parameters.solver.gamma;
    cfl_ = parameters.solver.cfl;
    final_time_ = parameters.solver.final_time;
    output_every_ = parameters.solver.output_every;
    output_time_interval_ = parameters.solver.output_time_interval;
    write_final_ = parameters.solver.write_final;
    out_prefix_ = parameters.solver.out_prefix;

    if ( gamma_ <= 1.0 || cfl_ <= 0.0 || final_time_ < 0.0 )
        throw std::runtime_error( "Require gamma > 1, cfl > 0, finalTime >= 0" );
    if ( !std::isfinite( output_time_interval_ ) || output_time_interval_ < 0.0 )
        throw std::runtime_error( "Require finite outputTimeInterval >= 0" );
    if ( grid_.nx < grid_.ng || grid_.ny < grid_.ng )
        throw std::runtime_error( "Local nx and ny must be >= ng for halo exchange and wall reflection" );

    vtk_data_output_type_ = parameters.solver.vtk_data_output_type;

    // 4. Store the physical boundary conditions.
    bc_ = parameters.boundary;

    // 5. Allocate the ghosted cell fields and reusable face arrays.
    grid_.nx_tot = grid_.nx + 2 * grid_.ng;
    grid_.ny_tot = grid_.ny + 2 * grid_.ng;
    conservative_field_.assign( static_cast<size_t>( grid_.nx_tot ) * static_cast<size_t>( grid_.ny_tot ),
        Conservative_vector{} );
    faces_.resize( grid_.nx, grid_.ny );

    // 6. Select the numerical flux and explicit time integrator.
    flux_name_ = parameters.solver.flux;
    flux_scheme_ = read_flux_scheme( flux_name_ );
    ti_ = make_time_integrator( parameters.solver.time_integrator );

    // 7. Initialize this rank's interior cells; ghost cells are filled before the first residual.
    set_fields_2d( conservative_field_, grid_.nx, grid_.ny, grid_.ng, x0_loc, x1_loc, y0_loc, y1_loc, gamma_, parameters.set_fields );
}


// Exchange MPI halos and apply physical boundary conditions before reconstruction.
//
// state (in/out) - conserved variables in the ghosted cell field
//
// Exchange MPI halos, then fill physical-boundary ghost cells.
void Solver::apply_bc( std::vector<Conservative_vector> &state ) const
{
    mp_.exchange_halos_2d( state, grid_.nx, grid_.ny, grid_.ng );

    // Keep exchanged ghost values at MPI interfaces; apply physical conditions only at global edges.
    const auto nbr = mp_.neighbors();

    BoundaryConditions2D bc_apply = bc_;

    if ( nbr.west != MPI_PROC_NULL )
    {
        bc_apply.left.type = BoundaryType::internal;
    }

    if ( nbr.east != MPI_PROC_NULL )
    {
        bc_apply.right.type = BoundaryType::internal;
    }

    if ( nbr.south != MPI_PROC_NULL )
    {
        bc_apply.bottom.type = BoundaryType::internal;
    }

    if ( nbr.north != MPI_PROC_NULL )
    {
        bc_apply.top.type = BoundaryType::internal;
    }

    apply_boundary_conditions( state, grid_.nx, grid_.ny, grid_.ng, bc_apply );
}


// Check interior states and throw with cell coordinates on failure.
//
// field (in) - ghosted conservative field; only interior cells contribute to this operation
//
// Check every interior state; optionally collect the CFL rate using the same evaluated properties.
// max_s (out, optional) - local maximum spectral rate; only requested when another step follows.
void Solver::validate_interior( const std::vector<Conservative_vector> &field, double *max_s ) const
{
    const double dx = grid_.dx();
    const double dy = grid_.dy();

    if ( max_s != nullptr )
        *max_s = 0.0;

    for ( int j = 0; j < grid_.ny; ++j )
    {
        for ( int i = 0; i < grid_.nx; ++i )
        {
            const auto &state = field[idx( grid_.ng + i, grid_.ng + j )];

            try
            {
                EvaluatedState checked_state{};

                require_success( calc_flow_variables( state, gamma_, checked_state ), "calc_flow_variables" );

                if ( max_s != nullptr )
                {
                    const double spectral_rate = ( std::abs( checked_state.x_velocity ) + checked_state.sound_speed ) / dx +
                                                 ( std::abs( checked_state.y_velocity ) + checked_state.sound_speed ) / dy;
                    if ( !std::isfinite( spectral_rate ) || spectral_rate <= 0.0 )
                        throw std::runtime_error( "CFL: invalid spectral radius" );
                    *max_s = std::max( *max_s, spectral_rate );
                }
            }
            catch ( const std::exception &e )
            {
                std::ostringstream msg;

                msg << "cell (" << grid_.i_beg + i << "," << grid_.j_beg + j << "), U=[" << std::setprecision( 17 )
                    << state.specific_mass << "," << state.specific_x_momentum << "," << state.specific_y_momentum
                    << "," << state.specific_energy << "]: " << e.what();
                throw std::runtime_error( msg.str() );
            }
        }
    }
}


// Compute a common MPI time step from the spectral rate collected during state validation.
//
// max_s (in) - local maximum for the current conservative field
//
// Compute dt = CFL / global max((abs(u) + c)/dx + (abs(v) + c)/dy).
double Solver::compute_dt( double max_s ) const
{
    // Use the global maximum so every rank advances with the same time step.
    const double max_sg = mp_.allreduce_max( max_s );
    const double dt = cfl_ / max_sg;

    if ( !std::isfinite( dt ) || dt <= 0.0 )
    {
        throw std::runtime_error( "CFL: invalid time step" );
    }

    return dt;
}


// Compute the flux divergence; the input ghost cells must already be valid.
//
// state (in) - conserved variables in the ghosted cell field
// output_rhs (out) - conservative residual on the ghosted cell layout
//
// Compute the finite-volume residual: RHS = -(F_right - F_left)/dx - (G_top - G_bottom)/dy.
void Solver::build_rhs( const std::vector<Conservative_vector> &state, std::vector<Conservative_vector> &output_rhs )
{
    std::fill( output_rhs.begin(), output_rhs.end(), Conservative_vector{} );

    // 1. Reconstruct face states from the field with valid ghost cells.
    recon_.reconstruct_quadrature( state, grid_.nx, grid_.ny, grid_.ng, gamma_, x_direction, faces_.x.left_state, faces_.x.right_state );
    recon_.reconstruct_quadrature( state, grid_.nx, grid_.ny, grid_.ng, gamma_, y_direction, faces_.y.left_state, faces_.y.right_state );

    const double dx = grid_.dx();
    const double dy = grid_.dy();

    // 2. Let the numerical flux validate face states; attach global face coordinates to failures.
    auto face_flux =
        [&]( const Conservative_vector &left, const Conservative_vector &right, int dir, int i, int j, int q )
    {
        try
        {
            Flux_vector flux{};
            const ReturnCodes code = calc_numerical_flux( flux_scheme_, left, right, dir, gamma_, flux );

            if ( code != SUCCESS )
            {
                throw std::runtime_error( "numerical flux: error code " + std::to_string( static_cast<int>( code ) ) );
            }

            if ( !is_finite_flux( flux ) )
            {
                throw std::runtime_error( "non-finite numerical flux" );
            }

            return flux;
        }
        catch ( const std::exception &e )
        {
            throw std::runtime_error( flux_name_ + " " + std::string( dir == x_direction ? "x" : "y" ) + "-face (" +
                                      std::to_string( grid_.i_beg + i ) + "," + std::to_string( grid_.j_beg + j ) +
                                      ") quadrature=" + std::to_string( q ) + ": " + e.what() );
        }
    };

    // Four-point Gauss weights for a face average.
    const double quadrature_weights[4] = {
        0.17392742256872692869, 0.32607257743127307131, 0.32607257743127307131, 0.17392742256872692869 };
    const int count = recon_.options().scheme == ReconstructionScheme::weno5 ? 4 : 1;

    for ( int j = 0; j < grid_.ny; ++j )
    {
        for ( int i = 0; i < grid_.nx + 1; ++i )
        {
            const int f = idx_face_x( i, j );

            Flux_vector average{};
            for ( int q = 0; q < count; ++q )
            {
                const auto flux = face_flux( faces_.x.left_state[f][q], faces_.x.right_state[f][q], x_direction, i, j, q );
                const double weight = count == 1 ? 1.0 : quadrature_weights[q];
                
                average.mass_flux += weight * flux.mass_flux;
                average.x_momentum_flux += weight * flux.x_momentum_flux;
                average.y_momentum_flux += weight * flux.y_momentum_flux;
                average.energy_flux += weight * flux.energy_flux;
            }
            faces_.x.flux[f] = average;
        }
    }

    for ( int j = 0; j < grid_.ny + 1; ++j )
    {
        for ( int i = 0; i < grid_.nx; ++i )
        {
            const int f = idx_face_y( i, j );

            Flux_vector average{};
            for ( int q = 0; q < count; ++q )
            {
                const auto flux = face_flux( faces_.y.left_state[f][q], faces_.y.right_state[f][q], y_direction, i, j, q );
                const double weight = count == 1 ? 1.0 : quadrature_weights[q];
                
                average.mass_flux += weight * flux.mass_flux;
                average.x_momentum_flux += weight * flux.x_momentum_flux;
                average.y_momentum_flux += weight * flux.y_momentum_flux;
                average.energy_flux += weight * flux.energy_flux;
            }
            faces_.y.flux[f] = average;
        }
    }

    for ( int j = 0; j < grid_.ny; ++j )
    {
        for ( int i = 0; i < grid_.nx; ++i )
        {
            const Flux_vector &fx_l = faces_.x.flux[idx_face_x( i, j )];
            const Flux_vector &fx_r = faces_.x.flux[idx_face_x( i + 1, j )];
            const Flux_vector &gy_b = faces_.y.flux[idx_face_y( i, j )];
            const Flux_vector &gy_t = faces_.y.flux[idx_face_y( i, j + 1 )];

            // 3. Subtract outgoing fluxes from incoming fluxes to form the cell residual.
            Conservative_vector rhs{};

            rhs.specific_mass = ( -1.0 * ( fx_r.mass_flux - fx_l.mass_flux ) ) / dx - ( gy_t.mass_flux - gy_b.mass_flux ) / dy;
            rhs.specific_x_momentum = ( -1.0 * ( fx_r.x_momentum_flux - fx_l.x_momentum_flux ) ) / dx - ( gy_t.x_momentum_flux - gy_b.x_momentum_flux ) / dy;
            rhs.specific_y_momentum = ( -1.0 * ( fx_r.y_momentum_flux - fx_l.y_momentum_flux ) ) / dx - ( gy_t.y_momentum_flux - gy_b.y_momentum_flux ) / dy;
            rhs.specific_energy = ( -1.0 * ( fx_r.energy_flux - fx_l.energy_flux ) ) / dx - ( gy_t.energy_flux - gy_b.energy_flux ) / dy;

            if ( !is_finite_state( rhs ) )
            {
                throw std::runtime_error( "Non-finite flux divergence" );
            }

            output_rhs[idx( grid_.ng + i, grid_.ng + j )] = rhs;
        }
    }
}


// Check whether the step meets the configured output interval.
//
// step (in) - current time-step index
//
// Check the step against the configured output interval; non-positive intervals disable step output.
bool Solver::should_write_step_output( const int step ) const
{
    return output_every_ > 0 && ( step % output_every_ == 0 );
}


// Write scheduled merged VTK output.
//
// t (in) - current simulation time
//
// Write a merged snapshot after the caller has selected an output event.
void Solver::write_output( double t ) const
{
    ensure_output_directory_exists( mp_ );

    const std::string fname = make_step_output_path( out_prefix_, t );

    write_merged_vtk_file_2d( mp_, fname, "step", t, conservative_field_, grid_.nx, grid_.ny, grid_.ng, grid_.i_beg,
        grid_.j_beg, grid_.nx_global, grid_.ny_global, grid_.x0, grid_.x1, grid_.y0, grid_.y1, gamma_,
        vtk_data_output_type_ );
}


// Write the final merged VTK output when enabled.
//
// t (in) - current simulation time
//
// Write the final merged snapshot when final output is enabled.
void Solver::write_final_output( const double t ) const
{
    if ( !write_final_ )
    {
        return;
    }

    ensure_output_directory_exists( mp_ );

    const std::string fname = make_final_output_path( out_prefix_, t );

    write_merged_vtk_file_2d( mp_, fname, "final", t, conservative_field_, grid_.nx, grid_.ny, grid_.ng, grid_.i_beg,
        grid_.j_beg, grid_.nx_global, grid_.ny_global, grid_.x0, grid_.x1, grid_.y0, grid_.y1, gamma_,
        vtk_data_output_type_ );
}


// Process scheduled output using interior cell values only.
//
// step (in) - current time-step index
// t (in) - current simulation time
//
// Output gathers interior cells and does not require valid ghost cells.
void Solver::process_regular_output_phase( const int step, const double t, const bool time_output )
{
    if ( output_time_interval_ > 0.0 ? time_output : should_write_step_output( step ) )
        write_output( t );
}


// Advance the initialized solution and process scheduled and final output.
//
// Advance the solution to final_time_, checking each stage and completed step.
void Solver::run()
{
    double t = 0.0;
    int step = 0;
    std::size_t output_index = 1;
    double next_output_time = output_time_interval_;
    std::cout << "[2D][r" << mp_.rank() << "] Starting run, finalTime=" << final_time_ << ", global=" << grid_.nx_global
              << "x" << grid_.ny_global << ", local=" << grid_.nx << "x" << grid_.ny << ", begin=(" << grid_.i_beg
              << "," << grid_.j_beg << ")\n";

    double max_s = 0.0;
    validate_interior( conservative_field_, final_time_ > 0.0 ? &max_s : nullptr );

    process_regular_output_phase( step, t, true );

    int stage = 0;
    // Each integrator stage must validate its interior and refresh ghosts before reconstruction.
    auto rhs_fun = [this, &stage]( std::vector<Conservative_vector> &uin, std::vector<Conservative_vector> &rout )
    {
        ++stage;
        validate_interior( uin );
        apply_bc( uin );
        build_rhs( uin, rout );
    };

    while ( t < final_time_ )
    {
        double dt = compute_dt( max_s );

        // Land on the next scheduled time or final time without exceeding the CFL step.
        const bool time_mode = output_time_interval_ > 0.0;
        const double target_time = time_mode ? std::min( next_output_time, final_time_ ) : final_time_;
        const double remaining = target_time - t;
        const bool reaches_target = dt >= remaining;
        if ( reaches_target )
            dt = remaining;

        if ( !std::isfinite( dt ) || dt <= 0.0 || t + dt <= t )
            throw std::runtime_error( "Time step does not advance the physical clock" );
        stage = 0;

        try
        {
            ti_->step( conservative_field_, dt, rhs_fun );

            // Reuse this validation's flow properties for the next step's CFL calculation.
            const double next_time = reaches_target ? target_time : t + dt;
            validate_interior( conservative_field_, next_time < final_time_ ? &max_s : nullptr );
        }
        catch ( const std::exception &e )
        {
            throw std::runtime_error( "step=" + std::to_string( step ) + ", t=" + std::to_string( t ) +
                                      ", RHS stage=" + std::to_string( stage ) + ": " + e.what() );
        }

        // Advance the clock only after the full step passes the state checks.
        t = reaches_target ? target_time : t + dt;
        ++step;

        // Treat a rounded interval multiple at finalTime as the same output event.
        // Cap the tolerance below the interval so distinct events cannot be merged.
        const double output_tolerance = std::min( output_time_interval_ * 0.25,
            8.0 * std::numeric_limits<double>::epsilon() * std::max( std::abs( t ), std::abs( next_output_time ) ) );
        const bool time_output = time_mode && ( t == next_output_time || ( t == final_time_ && std::abs( t - next_output_time ) <= output_tolerance ) );
        
        process_regular_output_phase( step, t, time_output );
        
        if ( time_output && t < final_time_ )
        {
            // Multiplication avoids drift from repeatedly adding the interval.
            if ( output_index == std::numeric_limits<std::size_t>::max() )
            {
                throw std::runtime_error( "Output time index overflow" );
            }
            
            ++output_index;
            next_output_time = static_cast<double>( output_index ) * output_time_interval_;
            
            if ( next_output_time <= t )
            {
                throw std::runtime_error( "outputTimeInterval does not advance the output clock" );
            }
        }
    }

    if ( write_final_ )
    {
        write_final_output( t );
    }
}
