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

// Read the VTK data location: point or cell, ignoring letter case.
static VTKDataOutputType parse_vtkdata_output_type( const std::string &value )
{
    std::string s = to_lowercase( value );

    if ( s == "point" )
    {
        return VTKDataOutputType::point;
    }

    if ( s == "cell" )
    {
        return VTKDataOutputType::cell;
    }

    throw std::runtime_error( "Unknown output.vtkDataOutputType: " + value + " (expected point or cell)" );
}


// Return the VTK label for the selected data location.
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
static void ensure_output_directory_exists( const MpiParallel &mp )
{
    if ( mp.is_root() )
    {
        std::filesystem::create_directories( k_output_dir_2d );
    }

    mp.barrier();
}


// Build the scheduled output path from the prefix, step and time.
static std::string make_step_output_path( const std::string &out_prefix, int step, double t )
{
    std::ostringstream step_ss;

    step_ss << std::setw( 4 ) << std::setfill( '0' ) << step;

    std::ostringstream t_ss;

    t_ss << std::fixed << std::setprecision( 6 ) << t;

    const std::filesystem::path out_dir = k_output_dir_2d;
    return ( out_dir / ( out_prefix + "_step" + step_ss.str() + "_t=" + t_ss.str() + ".vtk" ) ).string();
}


// Build the final output path from the prefix and time.
static std::string make_final_output_path( const std::string &out_prefix, double t )
{
    std::ostringstream t_ss;

    t_ss << std::fixed << std::setprecision( 6 ) << t;

    const std::filesystem::path out_dir = k_output_dir_2d;
    return ( out_dir / ( out_prefix + "_final_t=" + t_ss.str() + ".vtk" ) ).string();
}


// Write a merged MPI snapshot using the selected point or cell data location.
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


// Read the case, decompose the mesh, allocate fields and initialize the interior state.
// cfg (in) - case configuration
// mp (in) - MPI helper; must outlive the solver
Solver::Solver( const Cfg &cfg, const MpiParallel &mp ) : mp_( mp ), recon_( cfg )
{
    // 1. Read the global mesh and verify the selected reconstruction stencil fits the ghost width.
    grid_.nx_global = cfg.get_int( "nx", 200 );
    grid_.ny_global = cfg.get_int( "ny", 50 );

    grid_.ng = cfg.get_int( "ng", 2 );

    if ( grid_.ng < 1 )
    {
        throw std::runtime_error( "ng must be >=1 for 2D solver" );
    }

    const int ng_req = required_ghost_cells( recon_.options().scheme );

    if ( grid_.ng < ng_req )
    {
        throw std::runtime_error( "ng is too small for the selected reconstruction.scheme" );
    }

    grid_.x0 = cfg.get_double( "x0", 0.0 );
    grid_.x1 = cfg.get_double( "x1", 1.0 );
    grid_.y0 = cfg.get_double( "y0", 0.0 );
    grid_.y1 = cfg.get_double( "y1", 0.2 );

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

    // 3. Read the gas model, time controls and output settings.
    gamma_ = cfg.get_double( "gamma", 1.4 );
    cfl_ = cfg.get_double( "cfl", 0.5 );
    final_time_ = cfg.get_double( "finalTime", 0.2 );
    output_every_ = cfg.get_int( "outputEvery", 50 );
    write_final_ = cfg.get_bool( "writeFinal", true );
    out_prefix_ = cfg.get_string( "outPrefix", "run2d" );

    if ( gamma_ <= 1.0 || cfl_ <= 0.0 || final_time_ < 0.0 )
        throw std::runtime_error( "Require gamma > 1, cfl > 0, finalTime >= 0" );
    if ( grid_.nx < grid_.ng || grid_.ny < grid_.ng )
        throw std::runtime_error( "Local nx and ny must be >= ng for halo exchange and wall reflection" );

    vtk_data_output_type_ = parse_vtkdata_output_type( cfg.get_string( "output.vtkDataOutputType", "point" ) );

    // 4. Read the physical boundary conditions.
    bc_ = read_boundary_conditions( cfg );

    // 5. Allocate the ghosted cell fields and reusable face arrays.
    grid_.nx_tot = grid_.nx + 2 * grid_.ng;
    grid_.ny_tot = grid_.ny + 2 * grid_.ng;
    conservative_field_.assign( static_cast<size_t>( grid_.nx_tot ) * static_cast<size_t>( grid_.ny_tot ),
        Conservative_vector{} );
    rhs_.assign( static_cast<size_t>( grid_.nx_tot ) * static_cast<size_t>( grid_.ny_tot ), Conservative_vector{} );
    faces_.resize( grid_.nx, grid_.ny );

    // 6. Select the numerical flux and explicit time integrator.
    flux_name_ = cfg.get_string( "flux", "hllc" );
    flux_scheme_ = read_flux_scheme( flux_name_ );
    ti_ = make_time_integrator( cfg.get_string( "timeIntegrator", "ssprk3" ) );

    // 7. Initialize this rank's interior cells; ghost cells are filled before the first residual.
    set_fields_2d( conservative_field_, grid_.nx, grid_.ny, grid_.ng, x0_loc, x1_loc, y0_loc, y1_loc, gamma_, cfg );
}


// Exchange MPI halos, then fill physical-boundary ghost cells.
// state (in/out) - local conservative field including ghost cells
void Solver::apply_bc( std::vector<Conservative_vector> &state ) const
{
    // Pack named state members explicitly; MPI does not depend on the structure memory layout.
    halo_buffer_.resize( state.size() * conservative_variable_count );

    for ( std::size_t cell = 0; cell < state.size(); ++cell )
        pack_state( state[cell], halo_buffer_.data() + cell * conservative_variable_count );
    mp_.exchange_halos_2d( halo_buffer_.data(), grid_.nx, grid_.ny, grid_.ng, conservative_variable_count );

    for ( std::size_t cell = 0; cell < state.size(); ++cell )
        state[cell] = unpack_state( halo_buffer_.data() + cell * conservative_variable_count );

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


// Check every interior state; report global cell indices and conserved values on failure.
void Solver::validate_interior( const std::vector<Conservative_vector> &field ) const
{
    for ( int j = 0; j < grid_.ny; ++j )
    {
        for ( int i = 0; i < grid_.nx; ++i )
        {
            const auto &state = field[idx( grid_.ng + i, grid_.ng + j )];

            try
            {
                {
                    EvaluatedState checked_state{};

                    require_success( calc_flow_variables( state, gamma_, checked_state ), "calc_flow_variables" );
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


// Compute a common MPI time step: dt = CFL / max((abs(u) + c)/dx + (abs(v) + c)/dy).
double Solver::compute_dt( const std::vector<Conservative_vector> &field ) const
{
    const double dx = grid_.dx();
    const double dy = grid_.dy();
    double max_s = 0.0;

    for ( int j = 0; j < grid_.ny; ++j )
    {
        for ( int i = 0; i < grid_.nx; ++i )
        {
            EvaluatedState primitive{};

            require_success( calc_flow_variables( field[idx( grid_.ng + i, grid_.ng + j )], gamma_, primitive ),
                "calc_flow_variables" );
            const double s = ( std::abs( primitive.x_velocity ) + primitive.sound_speed ) / dx +
                             ( std::abs( primitive.y_velocity ) + primitive.sound_speed ) / dy;
            if ( !std::isfinite( s ) || s <= 0.0 )
                throw std::runtime_error( "CFL: invalid spectral radius at cell (" + std::to_string( grid_.i_beg + i ) +
                                          "," + std::to_string( grid_.j_beg + j ) + ")" );
            max_s = std::max( max_s, s );
        }
    }

    // Use the global maximum so every rank advances with the same time step.
    const double max_sg = mp_.allreduce_max( max_s );
    const double dt = cfl_ / max_sg;

    if ( !std::isfinite( dt ) || dt <= 0.0 )
    {
        throw std::runtime_error( "CFL: invalid time step" );
    }

    return dt;
}


// Compute the finite-volume residual: RHS = -(F_right - F_left)/dx - (G_top - G_bottom)/dy.
// state (in) - local conservative field with valid ghost cells
// output_rhs (out) - preallocated residual array; ghost entries are set to zero
void Solver::build_rhs( const std::vector<Conservative_vector> &state, std::vector<Conservative_vector> &output_rhs )
{
    std::fill( output_rhs.begin(), output_rhs.end(), Conservative_vector{} );

    // 1. Reconstruct face states from the field with valid ghost cells.
    recon_.reconstruct_faces_x( state, grid_.nx, grid_.ny, grid_.ng, gamma_, faces_.x.left_state,
        faces_.x.right_state );
    recon_.reconstruct_faces_y( state, grid_.nx, grid_.ny, grid_.ng, gamma_, faces_.y.left_state,
        faces_.y.right_state );

    const double dx = grid_.dx();
    const double dy = grid_.dy();

    // 2. Check face states and attach global face coordinates to any flux failure.
    auto face_flux = [&]( const Conservative_vector &left, const Conservative_vector &right, int dir, int i, int j )
    {
        try
        {
            {
                EvaluatedState checked_state{};

                require_success( calc_flow_variables( left, gamma_, checked_state ), "calc_flow_variables" );
            }

            {
                EvaluatedState checked_state{};

                require_success( calc_flow_variables( right, gamma_, checked_state ), "calc_flow_variables" );
            }

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
                                      "): " + e.what() );
        }
    };

    for ( int j = 0; j < grid_.ny; ++j )
    {
        for ( int i = 0; i < grid_.nx + 1; ++i )
        {
            const int f = idx_face_x( i, j );

            faces_.x.flux[f] = face_flux( faces_.x.left_state[f], faces_.x.right_state[f], x_direction, i, j );
        }
    }

    for ( int j = 0; j < grid_.ny + 1; ++j )
    {
        for ( int i = 0; i < grid_.nx; ++i )
        {
            const int f = idx_face_y( i, j );

            faces_.y.flux[f] = face_flux( faces_.y.left_state[f], faces_.y.right_state[f], y_direction, i, j );
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

            rhs.specific_mass =
                ( -1.0 * ( fx_r.mass_flux - fx_l.mass_flux ) ) / dx - ( gy_t.mass_flux - gy_b.mass_flux ) / dy;
            rhs.specific_x_momentum = ( -1.0 * ( fx_r.x_momentum_flux - fx_l.x_momentum_flux ) ) / dx -
                                      ( gy_t.x_momentum_flux - gy_b.x_momentum_flux ) / dy;
            rhs.specific_y_momentum = ( -1.0 * ( fx_r.y_momentum_flux - fx_l.y_momentum_flux ) ) / dx -
                                      ( gy_t.y_momentum_flux - gy_b.y_momentum_flux ) / dy;
            rhs.specific_energy =
                ( -1.0 * ( fx_r.energy_flux - fx_l.energy_flux ) ) / dx - ( gy_t.energy_flux - gy_b.energy_flux ) / dy;

            if ( !is_finite_state( rhs ) )
            {
                throw std::runtime_error( "Non-finite flux divergence" );
            }

            output_rhs[idx( grid_.ng + i, grid_.ng + j )] = rhs;
        }
    }
}


// Check the step against the configured output interval; non-positive intervals disable step output.
bool Solver::should_write_step_output( const int step ) const
{
    return output_every_ > 0 && ( step % output_every_ == 0 );
}


// Write a merged snapshot when the step meets the output interval.
void Solver::write_output( int step, double t ) const
{
    if ( !should_write_step_output( step ) )
        return;

    ensure_output_directory_exists( mp_ );

    const std::string fname = make_step_output_path( out_prefix_, step, t );

    write_merged_vtk_file_2d( mp_, fname, "step", t, conservative_field_, grid_.nx, grid_.ny, grid_.ng, grid_.i_beg,
        grid_.j_beg, grid_.nx_global, grid_.ny_global, grid_.x0, grid_.x1, grid_.y0, grid_.y1, gamma_,
        vtk_data_output_type_ );
}


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


// Refresh ghost cells and process scheduled output for the current step.
void Solver::process_regular_output_phase( const int step, const double t )
{
    apply_bc( conservative_field_ );
    write_output( step, t );
}


// Advance the solution to final_time_, checking each stage and completed step.
void Solver::run()
{
    double t = 0.0;
    int step = 0;
    std::cout << "[2D][r" << mp_.rank() << "] Starting run, finalTime=" << final_time_ << ", global=" << grid_.nx_global
              << "x" << grid_.ny_global << ", local=" << grid_.nx << "x" << grid_.ny << ", begin=(" << grid_.i_beg
              << "," << grid_.j_beg << ")\n";

    validate_interior( conservative_field_ );

    apply_bc( conservative_field_ );
    build_rhs( conservative_field_, rhs_ );
    process_regular_output_phase( step, t );

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
        double dt = compute_dt( conservative_field_ );

        // Shorten the final step to reach the requested terminal time.
        if ( t + dt > final_time_ )
            dt = final_time_ - t;

        if ( !std::isfinite( dt ) || dt <= 0.0 || t + dt <= t )
            throw std::runtime_error( "Time step does not advance the physical clock" );
        stage = 0;

        try
        {
            ti_->step( conservative_field_, dt, rhs_fun );
            validate_interior( conservative_field_ );
        }
        catch ( const std::exception &e )
        {
            throw std::runtime_error( "step=" + std::to_string( step ) + ", t=" + std::to_string( t ) +
                                      ", RHS stage=" + std::to_string( stage ) + ": " + e.what() );
        }

        // Advance the clock only after the full step passes the state checks.
        t += dt;
        ++step;

        process_regular_output_phase( step, t );
    }

    if ( write_final_ )
    {
        apply_bc( conservative_field_ );
        write_final_output( t );
    }
}
