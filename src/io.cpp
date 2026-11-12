#include "common_functions.hpp"
#include "io.hpp"
#include "structures.hpp"
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <iomanip>
#include <limits>
#include <locale>
#include <cstdio>


// Write interior-cell fields to a VTK rectilinear grid after averaging them onto grid points.
//
// filename (in) - destination VTK file path
// field (in) - ghosted conservative field; only interior cells contribute to this operation
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
void write_vtk_2d_point_data( const std::string &filename, const std::vector<Conservative_vector> &field, int nx,
    int ny, int ng, double x0, double x1, double y0, double y1, double gamma )
{
    if ( nx <= 0 || ny <= 0 )
    {
        throw std::runtime_error( "writeVTK2D_PointData: nx, ny must be > 0" );
    }

    if ( ng < 0 )
    {
        throw std::runtime_error( "writeVTK2D_PointData: ng must be >= 0" );
    }

    const int nx_tot = nx + 2 * ng;
    auto idx = [&]( int i, int j )
    {
        return i + nx_tot * j;
    };

    const int nx_pts = nx + 1;
    const int ny_pts = ny + 1;
    auto idx_p = [&]( int i, int j )
    {
        return i + nx_pts * j;
    };

    const double dx = ( x1 - x0 ) / static_cast<double>( nx );
    const double dy = ( y1 - y0 ) / static_cast<double>( ny );

    // Write a temporary file and rename it after the stream is successfully closed.
    const std::string tmp_file = filename + ".tmp";
    std::ofstream out( tmp_file );

    if ( !out )
    {
        throw std::runtime_error( "writeVTK2D_PointData: cannot open file: " + tmp_file );
    }

    out.imbue( std::locale::classic() );
    out.exceptions( std::ofstream::failbit | std::ofstream::badbit );
    out.setf( std::ios::scientific );
    out << std::setprecision( std::numeric_limits<double>::max_digits10 );

    out << "# vtk DataFile Version 3.0\n";
    out << "2D Euler solution\n";
    out << "ASCII\n";
    out << "DATASET RECTILINEAR_GRID\n";
    out << "DIMENSIONS " << nx_pts << ' ' << ny_pts << " 1\n";

    out << "X_COORDINATES " << nx_pts << " double\n";

    for ( int i = 0; i < nx_pts; ++i )
    {
        const double x = x0 + static_cast<double>( i ) * dx;

        out << x << "\n";
    }

    out << "Y_COORDINATES " << ny_pts << " double\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        const double y = y0 + static_cast<double>( j ) * dy;

        out << y << "\n";
    }

    out << "Z_COORDINATES 1 double\n";
    out << 0.0 << "\n";

    const int n_cells = nx * ny;
    const int n_pts = nx_pts * ny_pts;
    auto idx_c = [&]( int i, int j )
    {
        return i + nx * j;
    };

    // Recover primitive fields at cell centers before averaging each field onto grid points.
    std::vector<double> rho_c( static_cast<std::size_t>( n_cells ) );
    std::vector<double> u_c( static_cast<std::size_t>( n_cells ) );
    std::vector<double> v_c( static_cast<std::size_t>( n_cells ) );
    std::vector<double> p_c( static_cast<std::size_t>( n_cells ) );
    std::vector<double> rho_uc( static_cast<std::size_t>( n_cells ) );
    std::vector<double> rho_vc( static_cast<std::size_t>( n_cells ) );
    std::vector<double> ec( static_cast<std::size_t>( n_cells ) );

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const Conservative_vector &state = field[idx( ng + i, ng + j )];
            Primitive_vector primitive{};

            require_success( calc_primitive_variables( state, gamma, primitive ), "calc_primitive_variables" );

            const std::size_t kc = static_cast<std::size_t>( idx_c( i, j ) );

            rho_c[kc] = primitive.density;
            u_c[kc] = primitive.x_velocity;
            v_c[kc] = primitive.y_velocity;
            p_c[kc] = primitive.pressure;
            rho_uc[kc] = state.specific_x_momentum;
            rho_vc[kc] = state.specific_y_momentum;
            ec[kc] = state.specific_energy;
        }
    }

    // Average only adjacent interior cells: four at interior points, two at edges and one at corners.
    std::vector<double> rho_p( static_cast<std::size_t>( n_pts ) );
    std::vector<double> u_p( static_cast<std::size_t>( n_pts ) );
    std::vector<double> v_p( static_cast<std::size_t>( n_pts ) );
    std::vector<double> p_p( static_cast<std::size_t>( n_pts ) );
    std::vector<double> rho_up( static_cast<std::size_t>( n_pts ) );
    std::vector<double> rho_vp( static_cast<std::size_t>( n_pts ) );
    std::vector<double> ep( static_cast<std::size_t>( n_pts ) );

    for ( int j = 0; j < ny_pts; ++j )
    {
        for ( int i = 0; i < nx_pts; ++i )
        {
            double rho_sum = 0.0;
            double u_sum = 0.0;
            double v_sum = 0.0;
            double p_sum = 0.0;
            double rho_usum = 0.0;
            double rho_vsum = 0.0;
            double esum = 0.0;
            int cnt = 0;

            for ( int dj = -1; dj <= 0; ++dj )
            {
                for ( int di = -1; di <= 0; ++di )
                {
                    const int ic = i + di;
                    const int jc = j + dj;

                    if ( ic >= 0 && ic < nx && jc >= 0 && jc < ny )
                    {
                        const std::size_t kc = static_cast<std::size_t>( idx_c( ic, jc ) );

                        rho_sum += rho_c[kc];
                        u_sum += u_c[kc];
                        v_sum += v_c[kc];
                        p_sum += p_c[kc];
                        rho_usum += rho_uc[kc];
                        rho_vsum += rho_vc[kc];
                        esum += ec[kc];
                        ++cnt;
                    }
                }
            }

            const std::size_t kp = static_cast<std::size_t>( idx_p( i, j ) );

            rho_p[kp] = rho_sum / static_cast<double>( cnt );
            u_p[kp] = u_sum / static_cast<double>( cnt );
            v_p[kp] = v_sum / static_cast<double>( cnt );
            p_p[kp] = p_sum / static_cast<double>( cnt );
            rho_up[kp] = rho_usum / static_cast<double>( cnt );
            rho_vp[kp] = rho_vsum / static_cast<double>( cnt );
            ep[kp] = esum / static_cast<double>( cnt );
        }
    }

    out << "POINT_DATA " << n_pts << "\n";

    out << "SCALARS rho double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        for ( int i = 0; i < nx_pts; ++i )
        {
            out << rho_p[static_cast<std::size_t>( idx_p( i, j ) )] << "\n";
        }
    }

    out << "VECTORS velocity double\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        for ( int i = 0; i < nx_pts; ++i )
        {
            const std::size_t kp = static_cast<std::size_t>( idx_p( i, j ) );

            out << u_p[kp] << ' ' << v_p[kp] << ' ' << 0.0 << "\n";
        }
    }

    out << "SCALARS p double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        for ( int i = 0; i < nx_pts; ++i )
        {
            out << p_p[static_cast<std::size_t>( idx_p( i, j ) )] << "\n";
        }
    }

    out << "SCALARS rho_u double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        for ( int i = 0; i < nx_pts; ++i )
        {
            out << rho_up[static_cast<std::size_t>( idx_p( i, j ) )] << "\n";
        }
    }

    out << "SCALARS rho_v double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        for ( int i = 0; i < nx_pts; ++i )
        {
            out << rho_vp[static_cast<std::size_t>( idx_p( i, j ) )] << "\n";
        }
    }

    out << "SCALARS E double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        for ( int i = 0; i < nx_pts; ++i )
        {
            out << ep[static_cast<std::size_t>( idx_p( i, j ) )] << "\n";
        }
    }

    out.flush();
    out.close();

    if ( std::rename( tmp_file.c_str(), filename.c_str() ) != 0 )
    {
        throw std::runtime_error( "writeVTK2D_PointData: failed to rename tmp file to final output: " + filename );
    }
}


// Write one set of field values per interior cell to a VTK rectilinear grid.
//
// filename (in) - destination VTK file path
// field (in) - ghosted conservative field; only interior cells contribute to this operation
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
void write_vtk_2d_cell_data( const std::string &filename, const std::vector<Conservative_vector> &field, int nx, int ny,
    int ng, double x0, double x1, double y0, double y1, double gamma )
{
    if ( nx <= 0 || ny <= 0 )
    {
        throw std::runtime_error( "writeVTK2D_CellData: nx, ny must be > 0" );
    }

    if ( ng < 0 )
    {
        throw std::runtime_error( "writeVTK2D_CellData: ng must be >= 0" );
    }

    const int nx_tot = nx + 2 * ng;
    auto idx = [&]( int i, int j )
    {
        return i + nx_tot * j;
    };

    const int nx_pts = nx + 1;
    const int ny_pts = ny + 1;
    const double dx = ( x1 - x0 ) / static_cast<double>( nx );
    const double dy = ( y1 - y0 ) / static_cast<double>( ny );

    const std::string tmp_file = filename + ".tmp";
    std::ofstream out( tmp_file );

    if ( !out )
    {
        throw std::runtime_error( "writeVTK2D_CellData: cannot open file: " + tmp_file );
    }

    out.imbue( std::locale::classic() );
    out.exceptions( std::ofstream::failbit | std::ofstream::badbit );
    out.setf( std::ios::scientific );
    out << std::setprecision( std::numeric_limits<double>::max_digits10 );

    out << "# vtk DataFile Version 3.0\n";
    out << "2D Euler solution\n";
    out << "ASCII\n";
    out << "DATASET RECTILINEAR_GRID\n";
    out << "DIMENSIONS " << nx_pts << ' ' << ny_pts << " 1\n";

    out << "X_COORDINATES " << nx_pts << " double\n";

    for ( int i = 0; i < nx_pts; ++i )
    {
        const double x = x0 + static_cast<double>( i ) * dx;

        out << x << "\n";
    }

    out << "Y_COORDINATES " << ny_pts << " double\n";

    for ( int j = 0; j < ny_pts; ++j )
    {
        const double y = y0 + static_cast<double>( j ) * dy;

        out << y << "\n";
    }

    out << "Z_COORDINATES 1 double\n";
    out << 0.0 << "\n";

    const int n_cells = nx * ny;

    out << "CELL_DATA " << n_cells << "\n";

    out << "SCALARS rho double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const Conservative_vector &state = field[idx( ng + i, ng + j )];
            Primitive_vector primitive{};

            require_success( calc_primitive_variables( state, gamma, primitive ), "calc_primitive_variables" );
            out << primitive.density << "\n";
        }
    }

    out << "VECTORS velocity double\n";

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const Conservative_vector &state = field[idx( ng + i, ng + j )];
            Primitive_vector primitive{};

            require_success( calc_primitive_variables( state, gamma, primitive ), "calc_primitive_variables" );
            out << primitive.x_velocity << ' ' << primitive.y_velocity << ' ' << 0.0 << "\n";
        }
    }

    out << "SCALARS p double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const Conservative_vector &state = field[idx( ng + i, ng + j )];
            Primitive_vector primitive{};

            require_success( calc_primitive_variables( state, gamma, primitive ), "calc_primitive_variables" );
            out << primitive.pressure << "\n";
        }
    }

    out << "SCALARS rho_u double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const Conservative_vector &state = field[idx( ng + i, ng + j )];

            out << state.specific_x_momentum << "\n";
        }
    }

    out << "SCALARS rho_v double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const Conservative_vector &state = field[idx( ng + i, ng + j )];

            out << state.specific_y_momentum << "\n";
        }
    }

    out << "SCALARS E double 1\n";
    out << "LOOKUP_TABLE default\n";

    for ( int j = 0; j < ny; ++j )
    {
        for ( int i = 0; i < nx; ++i )
        {
            const Conservative_vector &state = field[idx( ng + i, ng + j )];

            out << state.specific_energy << "\n";
        }
    }

    out.flush();
    out.close();

    if ( std::rename( tmp_file.c_str(), filename.c_str() ) != 0 )
    {
        throw std::runtime_error( "writeVTK2D_CellData: failed to rename tmp file to final output: " + filename );
    }
}


// Gather interior states on communicator rank 0 and write a single POINT_DATA file.
//
// filename (in) - destination VTK file path
// ulocal (in) - ghosted conservative field; only interior cells contribute to this operation
// nx_local (in) - local number of interior cells along x
// ny_local (in) - local number of interior cells along y
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
// comm (in) - communicator containing all ranks participating in the collective write
void write_vtk_2d_point_data_gather_mpi( const std::string &filename, const std::vector<Conservative_vector> &ulocal,
    int nx_local, int ny_local, int ng, int i_beg, int j_beg, int nx_global, int ny_global, double x0, double x1,
    double y0, double y1, double gamma, MPI_Comm comm )
{
    if ( nx_local <= 0 || ny_local <= 0 )
    {
        throw std::runtime_error( "writeVTK2D_PointData_GatherMPI: nxLocal/nyLocal must be > 0" );
    }

    if ( nx_global <= 0 || ny_global <= 0 )
    {
        throw std::runtime_error( "writeVTK2D_PointData_GatherMPI: nxGlobal/nyGlobal must be > 0" );
    }

    if ( ng < 1 )
    {
        throw std::runtime_error( "writeVTK2D_PointData_GatherMPI: ng must be >= 1" );
    }

    int rank = 0, size = 1;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );

    const int nx_tot_loc = nx_local + 2 * ng;
    const int ny_tot_loc = ny_local + 2 * ng;

    (void)ny_tot_loc;

    auto idx_loc = [&]( int i, int j )
    {
        return i + nx_tot_loc * j;
    };

    const int ncomp = conservative_variable_count;
    const int local_cells = nx_local * ny_local;
    // Pack only interior cells; MPI counts and displacements are measured in doubles.
    const int local_count = local_cells * ncomp;

    std::vector<double> send( local_count );
    int t = 0;

    for ( int j = 0; j < ny_local; ++j )
    {
        for ( int i = 0; i < nx_local; ++i )
        {
            const Conservative_vector &state = ulocal[idx_loc( ng + i, ng + j )];

            pack_state( state, send.data() + t );
            t += ncomp;
        }
    }

    // Gather each block's global start indices and interior dimensions before gathering its state data.
    int meta[4] = { i_beg, j_beg, nx_local, ny_local };
    std::vector<int> all_meta;
    if ( rank == 0 )
        all_meta.resize( static_cast<size_t>( size ) * 4u );
    MPI_Gather( meta, 4, MPI_INT, rank == 0 ? all_meta.data() : nullptr, 4, MPI_INT, 0, comm );

    std::vector<int> counts;
    std::vector<int> displs;
    if ( rank == 0 )
    {
        counts.resize( size );
        displs.resize( size );
    }

    MPI_Gather( &local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm );

    std::vector<double> recv;

    if ( rank == 0 )
    {
        int total = 0;

        for ( int r = 0; r < size; ++r )
        {
            displs[r] = total;
            total += counts[r];
        }

        recv.resize( static_cast<size_t>( total ) );
    }

    MPI_Gatherv( send.data(), local_count, MPI_DOUBLE, rank == 0 ? recv.data() : nullptr,
        rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, comm );

    if ( rank != 0 )
        return;

    // Use one unused ghost layer to match the serial writer interface; output reads only interior cells.
    const int ng_out = 1;
    const int nx_tot_g = nx_global + 2 * ng_out;
    const int ny_tot_g = ny_global + 2 * ng_out;
    auto idx_g = [&]( int i, int j )
    {
        return i + nx_tot_g * j;
    };

    std::vector<Conservative_vector> uglobal( static_cast<size_t>( nx_tot_g ) * static_cast<size_t>( ny_tot_g ),
        Conservative_vector{} );

    // Place each rank's packed interior states at their global cell indices.
    for ( int r = 0; r < size; ++r )
    {
        const int ib = all_meta[4 * r + 0];
        const int jb = all_meta[4 * r + 1];
        const int nxr = all_meta[4 * r + 2];
        const int nyr = all_meta[4 * r + 3];

        const int base = displs[r];
        const int expected = nxr * nyr * ncomp;

        if ( counts[r] != expected )
        {
            throw std::runtime_error( "writeVTK2D_PointData_GatherMPI: inconsistent counts vs meta for rank " +
                                      std::to_string( r ) );
        }

        for ( int j = 0; j < nyr; ++j )
        {
            for ( int i = 0; i < nxr; ++i )
            {
                const int gi = ib + i;
                const int gj = jb + j;

                if ( gi < 0 || gi >= nx_global || gj < 0 || gj >= ny_global )
                {
                    throw std::runtime_error( "writeVTK2D_PointData_GatherMPI: rank block out of global bounds" );
                }

                const int loc_cell = ( i + nxr * j ) * ncomp;
                const auto state = unpack_state( recv.data() + base + loc_cell );

                uglobal[idx_g( ng_out + gi, ng_out + gj )] = state;
            }
        }
    }

    write_vtk_2d_point_data( filename, uglobal, nx_global, ny_global, ng_out, x0, x1, y0, y1, gamma );
}


// Gather interior states on communicator rank 0 and write a single CELL_DATA file.
//
// filename (in) - destination VTK file path
// ulocal (in) - ghosted conservative field; only interior cells contribute to this operation
// nx_local (in) - local number of interior cells along x
// ny_local (in) - local number of interior cells along y
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
// comm (in) - communicator containing all ranks participating in the collective write
void write_vtk_2d_cell_data_gather_mpi( const std::string &filename, const std::vector<Conservative_vector> &ulocal,
    int nx_local, int ny_local, int ng, int i_beg, int j_beg, int nx_global, int ny_global, double x0, double x1,
    double y0, double y1, double gamma, MPI_Comm comm )
{
    if ( nx_local <= 0 || ny_local <= 0 )
    {
        throw std::runtime_error( "writeVTK2D_CellData_GatherMPI: nxLocal/nyLocal must be > 0" );
    }

    if ( nx_global <= 0 || ny_global <= 0 )
    {
        throw std::runtime_error( "writeVTK2D_CellData_GatherMPI: nxGlobal/nyGlobal must be > 0" );
    }

    if ( ng < 1 )
    {
        throw std::runtime_error( "writeVTK2D_CellData_GatherMPI: ng must be >= 1" );
    }

    int rank = 0, size = 1;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );

    const int nx_tot_loc = nx_local + 2 * ng;
    auto idx_loc = [&]( int i, int j )
    {
        return i + nx_tot_loc * j;
    };

    const int ncomp = conservative_variable_count;
    const int local_cells = nx_local * ny_local;
    const int local_count = local_cells * ncomp;

    std::vector<double> send( local_count );
    int t = 0;

    for ( int j = 0; j < ny_local; ++j )
    {
        for ( int i = 0; i < nx_local; ++i )
        {
            const Conservative_vector &state = ulocal[idx_loc( ng + i, ng + j )];

            pack_state( state, send.data() + t );
            t += ncomp;
        }
    }

    int meta[4] = { i_beg, j_beg, nx_local, ny_local };
    std::vector<int> all_meta;

    if ( rank == 0 )
        all_meta.resize( static_cast<size_t>( size ) * 4u );
    MPI_Gather( meta, 4, MPI_INT, rank == 0 ? all_meta.data() : nullptr, 4, MPI_INT, 0, comm );

    std::vector<int> counts;
    std::vector<int> displs;

    if ( rank == 0 )
    {
        counts.resize( size );
        displs.resize( size );
    }

    MPI_Gather( &local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm );

    std::vector<double> recv;

    if ( rank == 0 )
    {
        int total = 0;

        for ( int r = 0; r < size; ++r )
        {
            displs[r] = total;
            total += counts[r];
        }

        recv.resize( static_cast<size_t>( total ) );
    }

    MPI_Gatherv( send.data(), local_count, MPI_DOUBLE, rank == 0 ? recv.data() : nullptr,
        rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, comm );

    if ( rank != 0 )
        return;

    const int ng_out = 1;
    const int nx_tot_g = nx_global + 2 * ng_out;
    const int ny_tot_g = ny_global + 2 * ng_out;
    auto idx_g = [&]( int i, int j )
    {
        return i + nx_tot_g * j;
    };

    std::vector<Conservative_vector> uglobal( static_cast<size_t>( nx_tot_g ) * static_cast<size_t>( ny_tot_g ),
        Conservative_vector{} );

    for ( int r = 0; r < size; ++r )
    {
        const int ib = all_meta[4 * r + 0];
        const int jb = all_meta[4 * r + 1];
        const int nxr = all_meta[4 * r + 2];
        const int nyr = all_meta[4 * r + 3];

        const int base = displs[r];
        const int expected = nxr * nyr * ncomp;

        if ( counts[r] != expected )
        {
            throw std::runtime_error( "writeVTK2D_CellData_GatherMPI: inconsistent counts vs meta for rank " +
                                      std::to_string( r ) );
        }

        for ( int j = 0; j < nyr; ++j )
        {
            for ( int i = 0; i < nxr; ++i )
            {
                const int gi = ib + i;
                const int gj = jb + j;

                if ( gi < 0 || gi >= nx_global || gj < 0 || gj >= ny_global )
                {
                    throw std::runtime_error( "writeVTK2D_CellData_GatherMPI: rank block out of global bounds" );
                }

                const int loc_cell = ( i + nxr * j ) * ncomp;
                const auto state = unpack_state( recv.data() + base + loc_cell );

                uglobal[idx_g( ng_out + gi, ng_out + gj )] = state;
            }
        }
    }

    write_vtk_2d_cell_data( filename, uglobal, nx_global, ny_global, ng_out, x0, x1, y0, y1, gamma );
}
