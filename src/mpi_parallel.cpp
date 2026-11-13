#include "mpi_parallel.hpp"

#include <algorithm>
#include <exception>
#include <string>


// Throw an exception with the operation name when an MPI call reports failure.
//
// err (in) - MPI return status to check
// where (in) - MPI operation name included in an error report
static void mpi_check( int err, const char *where )
{
    if ( err == MPI_SUCCESS )
        return;
    throw std::runtime_error( std::string( "MPI error at: " ) + where );
}


// Split a global axis into contiguous blocks; distribute remainder cells to the first blocks.
//
// global_n (in) - global number of cells along the split axis
// p (in) - number of process blocks along the split axis
// coord (in) - process coordinate along the split axis
//
// Split a global axis into contiguous blocks; the first remainder blocks get one extra cell.
Range1D MpiParallel::split1_d( int global_n, int p, int coord )
{
    if ( p <= 0 )
    {
        throw std::runtime_error( "MpiParallel::split1D: P must be > 0" );
    }

    if ( global_n <= 0 )
    {
        throw std::runtime_error( "MpiParallel::split1D: globalN must be > 0" );
    }

    if ( coord < 0 || coord >= p )
    {
        throw std::runtime_error( "MpiParallel::split1D: coord out of range" );
    }

    const int base = global_n / p;
    const int rem = global_n % p;

    Range1D r;

    r.count = base + ( coord < rem ? 1 : 0 );
    r.begin = coord * base + std::min( coord, rem );
    return r;
}


// Create the Cartesian MPI topology and identify neighboring ranks.
//
// px (in) - number of process columns along x
// py (in) - number of process rows along y
// parent (in) - parent MPI communicator
// periodic_x (in) - whether the Cartesian topology is periodic along x
// periodic_y (in) - whether the Cartesian topology is periodic along y
// reorder (in) - whether MPI may reorder Cartesian ranks
MpiParallel::MpiParallel( int px, int py, MPI_Comm parent, bool periodic_x, bool periodic_y, bool reorder )
    : parent_( parent ), px_( px ), py_( py )
{
    if ( px_ <= 0 || py_ <= 0 )
    {
        throw std::runtime_error( "MpiParallel: px and py must be > 0" );
    }

    mpi_check( MPI_Comm_rank( parent_, &rank_ ), "MPI_Comm_rank" );
    mpi_check( MPI_Comm_size( parent_, &size_ ), "MPI_Comm_size" );

    if ( px_ * py_ != size_ )
    {
        throw std::runtime_error( "MpiParallel: px*py must equal MPI_Comm_size" );
    }

    init_cart_( periodic_x, periodic_y, reorder );
}


// Create the Cartesian topology with MPI dimensions ordered as (y, x).
//
// periodic_x (in) - whether the Cartesian topology is periodic along x
// periodic_y (in) - whether the Cartesian topology is periodic along y
// reorder (in) - whether MPI may reorder Cartesian ranks
void MpiParallel::init_cart_( bool periodic_x, bool periodic_y, bool reorder )
{
    // MPI axis 0 is y and axis 1 is x; the stored cell index varies fastest along x.
    int dims[2] = { py_, px_ };
    int periods[2] = { periodic_y ? 1 : 0, periodic_x ? 1 : 0 };
    int reo = reorder ? 1 : 0;

    mpi_check( MPI_Cart_create( parent_, 2, dims, periods, reo, &cart_ ), "MPI_Cart_create" );

    if ( cart_ == MPI_COMM_NULL )
    {
        throw std::runtime_error( "MpiParallel: MPI_Cart_create returned MPI_COMM_NULL" );
    }

    mpi_check( MPI_Comm_rank( cart_, &rank_ ), "MPI_Comm_rank(cart)" );
    mpi_check( MPI_Comm_size( cart_, &size_ ), "MPI_Comm_size(cart)" );

    int coords[2] = { 0, 0 };

    mpi_check( MPI_Cart_coords( cart_, rank_, 2, coords ), "MPI_Cart_coords" );
    cy_ = coords[0];
    cx_ = coords[1];

    mpi_check( MPI_Cart_shift( cart_, 0, 1, &nbr_.south, &nbr_.north ), "MPI_Cart_shift(y)" );

    mpi_check( MPI_Cart_shift( cart_, 1, 1, &nbr_.west, &nbr_.east ), "MPI_Cart_shift(x)" );
}


// Transfer MPI communicator and buffer ownership from the source helper.
//
// o (in/out) - MPI helper whose communicator and buffers are transferred
//
// Transfer communicator ownership from another MPI helper.
MpiParallel::MpiParallel( MpiParallel &&o ) noexcept
{
    *this = std::move( o );
}


// Transfer MPI communicator and buffer ownership from the source helper.
//
// o (in/out) - MPI helper whose communicator and buffers are transferred
MpiParallel &MpiParallel::operator=( MpiParallel &&o ) noexcept
{
    if ( this == &o )
    {
        return *this;
    }

    parent_ = o.parent_;
    cart_ = o.cart_;

    rank_ = o.rank_;
    size_ = o.size_;

    px_ = o.px_;
    py_ = o.py_;

    cx_ = o.cx_;
    cy_ = o.cy_;

    nbr_ = o.nbr_;

    send_w_ = std::move( o.send_w_ );
    recv_w_ = std::move( o.recv_w_ );
    send_e_ = std::move( o.send_e_ );
    recv_e_ = std::move( o.recv_e_ );
    send_s_ = std::move( o.send_s_ );
    recv_s_ = std::move( o.recv_s_ );
    send_n_ = std::move( o.send_n_ );
    recv_n_ = std::move( o.recv_n_ );

    o.parent_ = MPI_COMM_NULL;
    o.cart_ = MPI_COMM_NULL;
    return *this;
}


// Release owned resources during object destruction.
MpiParallel::~MpiParallel()
{
    // Skip collective cleanup during exception unwinding; main reports the failure and aborts the MPI job.
    if ( std::uncaught_exceptions() > 0 )
        return;
    if ( cart_ != MPI_COMM_NULL )
    {
        MPI_Comm_free( &cart_ );

        cart_ = MPI_COMM_NULL;
    }
}


// Return this rank's global interior-cell ranges.
//
// global_nx (in) - global number of interior cells along x
// global_ny (in) - global number of interior cells along y
//
// Compute this rank's global interior-cell ranges.
Subdomain2D MpiParallel::decompose( int global_nx, int global_ny ) const
{
    if ( global_nx <= 0 || global_ny <= 0 )
    {
        throw std::runtime_error( "MpiParallel::decompose: globalNx/globalNy must be > 0" );
    }

    Subdomain2D sub;

    sub.x = split1_d( global_nx, px_, cx_ );
    sub.y = split1_d( global_ny, py_, cy_ );

    if ( sub.nx() <= 0 || sub.ny() <= 0 )
    {
        throw std::runtime_error( "MpiParallel::decompose: local nx/ny <= 0 (grid too small for px/py)" );
    }

    return sub;
}


// Synchronize all Cartesian ranks.
//
// Synchronize all ranks in the Cartesian communicator.
void MpiParallel::barrier() const
{
    mpi_check( MPI_Barrier( cart_ ), "MPI_Barrier" );
}


// Return the maximum local_val over all Cartesian ranks.
//
// local_val (in) - this rank's contribution to the collective reduction
//
// Compute the maximum value over all Cartesian ranks.
double MpiParallel::allreduce_max( double local_val ) const
{
    double out = 0.0;

    mpi_check( MPI_Allreduce( &local_val, &out, 1, MPI_DOUBLE, MPI_MAX, cart_ ), "MPI_Allreduce(MAX)" );
    return out;
}


// Return the minimum local_val over all Cartesian ranks.
//
// local_val (in) - this rank's contribution to the collective reduction
//
// Compute the minimum value over all Cartesian ranks.
double MpiParallel::allreduce_min( double local_val ) const
{
    double out = 0.0;

    mpi_check( MPI_Allreduce( &local_val, &out, 1, MPI_DOUBLE, MPI_MIN, cart_ ), "MPI_Allreduce(MIN)" );
    return out;
}


// Return the sum of local_val over all Cartesian ranks.
//
// local_val (in) - this rank's contribution to the collective reduction
//
// Compute the sum over all Cartesian ranks.
double MpiParallel::allreduce_sum( double local_val ) const
{
    double out = 0.0;

    mpi_check( MPI_Allreduce( &local_val, &out, 1, MPI_DOUBLE, MPI_SUM, cart_ ), "MPI_Allreduce(SUM)" );
    return out;
}


// Pack boundary state strips directly and exchange them with Cartesian neighbors.
//
// state (in/out) - ghosted conservative field; only received ghost strips are modified
// nx_local (in) - local number of interior cells along x
// ny_local (in) - local number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
//
// Exchange cell values with the four Cartesian neighbors and update ghost strips.
// MPI_PROC_NULL sides are left for the physical boundary conditions.
void MpiParallel::exchange_halos_2d( std::vector<Conservative_vector> &state, int nx_local, int ny_local, int ng ) const
{
    if ( nx_local <= 0 || ny_local <= 0 )
    {
        throw std::runtime_error( "exchangeHalos2D: nxLocal/nyLocal must be > 0" );
    }

    if ( ng < 0 )
    {
        throw std::runtime_error( "exchangeHalos2D: ng must be >= 0" );
    }

    if ( nx_local < ng || ny_local < ng )
        throw std::runtime_error( "exchangeHalos2D: local dimensions must be >= ng" );

    const int nx_tot = nx_local + 2 * ng;
    const int ny_tot = ny_local + 2 * ng;
    if ( state.size() != static_cast<std::size_t>( nx_tot ) * static_cast<std::size_t>( ny_tot ) )
        throw std::runtime_error( "exchangeHalos2D: state size mismatch" );

    if ( ng == 0 || ( nbr_.west == MPI_PROC_NULL && nbr_.east == MPI_PROC_NULL &&
                       nbr_.south == MPI_PROC_NULL && nbr_.north == MPI_PROC_NULL ) )
        return;

    const int ncomp = conservative_variable_count;

    // Include orthogonal ghost cells in each exchanged strip; counts below are in doubles.
    const int strip_x_cells = ng * ny_tot;
    const int strip_x_dbl = strip_x_cells * ncomp;

    const int strip_y_cells = nx_tot * ng;
    const int strip_y_dbl = strip_y_cells * ncomp;

    // Pack ng columns for the west/east exchange.
    auto pack_cols = [&]( int i0, std::vector<double> &buf )
    {
        buf.resize( strip_x_dbl );

        int t = 0;

        for ( int j = 0; j < ny_tot; ++j )
        {
            for ( int di = 0; di < ng; ++di )
            {
                pack_state( state[i0 + di + nx_tot * j], &buf[t] );

                t += ncomp;
            }
        }
    };

    auto unpack_cols = [&]( int i0, const std::vector<double> &buf )
    {
        int t = 0;

        for ( int j = 0; j < ny_tot; ++j )
        {
            for ( int di = 0; di < ng; ++di )
            {
                state[i0 + di + nx_tot * j] = unpack_state( &buf[t] );

                t += ncomp;
            }
        }
    };

    // Pack ng rows for the south/north exchange.
    auto pack_rows = [&]( int j0, std::vector<double> &buf )
    {
        buf.resize( strip_y_dbl );

        int t = 0;

        for ( int dj = 0; dj < ng; ++dj )
        {
            for ( int i = 0; i < nx_tot; ++i )
            {
                pack_state( state[i + nx_tot * ( j0 + dj )], &buf[t] );

                t += ncomp;
            }
        }
    };

    auto unpack_rows = [&]( int j0, const std::vector<double> &buf )
    {
        int t = 0;

        for ( int dj = 0; dj < ng; ++dj )
        {
            for ( int i = 0; i < nx_tot; ++i )
            {
                state[i + nx_tot * ( j0 + dj )] = unpack_state( &buf[t] );

                t += ncomp;
            }
        }
    };

    // Use a separate message tag for each travel direction.
    constexpr int tag_x_w2e = 200;
    constexpr int tag_x_e2w = 201;
    constexpr int tag_y_s2n = 210;
    constexpr int tag_y_n2s = 211;

    MPI_Request req[8];
    int nreq = 0;

    // 1. Post receives for all existing neighbors.
    if ( nbr_.west != MPI_PROC_NULL )
    {
        recv_w_.resize( strip_x_dbl );
        mpi_check( MPI_Irecv( recv_w_.data(), strip_x_dbl, MPI_DOUBLE, nbr_.west, tag_x_w2e, cart_, &req[nreq++] ),
            "MPI_Irecv(W)" );
    }

    if ( nbr_.east != MPI_PROC_NULL )
    {
        recv_e_.resize( strip_x_dbl );
        mpi_check( MPI_Irecv( recv_e_.data(), strip_x_dbl, MPI_DOUBLE, nbr_.east, tag_x_e2w, cart_, &req[nreq++] ),
            "MPI_Irecv(E)" );
    }

    if ( nbr_.west != MPI_PROC_NULL )
    {
        // 2. Pack interior strips and start nonblocking sends.
        pack_cols( ng, send_w_ );
        mpi_check( MPI_Isend( send_w_.data(), strip_x_dbl, MPI_DOUBLE, nbr_.west, tag_x_e2w, cart_, &req[nreq++] ),
            "MPI_Isend(W)" );
    }

    if ( nbr_.east != MPI_PROC_NULL )
    {
        pack_cols( ng + nx_local - ng, send_e_ );
        mpi_check( MPI_Isend( send_e_.data(), strip_x_dbl, MPI_DOUBLE, nbr_.east, tag_x_w2e, cart_, &req[nreq++] ),
            "MPI_Isend(E)" );
    }

    // 3. Complete all receives and sends before reusing the buffers.
    if ( nreq > 0 )
    {
        mpi_check( MPI_Waitall( nreq, req, MPI_STATUSES_IGNORE ), "MPI_Waitall(exchangeHalos2D)" );
    }

    if ( nbr_.west != MPI_PROC_NULL )
    {
        // 4. Copy received strips into ghost cells.
        unpack_cols( 0, recv_w_ );
    }

    if ( nbr_.east != MPI_PROC_NULL )
    {
        unpack_cols( ng + nx_local, recv_e_ );
    }

    // Finish x first: y strips must contain the newly received x halos to carry diagonal cells.
    nreq = 0;
    if ( nbr_.south != MPI_PROC_NULL )
    {
        recv_s_.resize( strip_y_dbl );
        mpi_check( MPI_Irecv( recv_s_.data(), strip_y_dbl, MPI_DOUBLE, nbr_.south, tag_y_s2n, cart_, &req[nreq++] ),
            "MPI_Irecv(S)" );
    }

    if ( nbr_.north != MPI_PROC_NULL )
    {
        recv_n_.resize( strip_y_dbl );
        mpi_check( MPI_Irecv( recv_n_.data(), strip_y_dbl, MPI_DOUBLE, nbr_.north, tag_y_n2s, cart_, &req[nreq++] ),
            "MPI_Irecv(N)" );
    }

    if ( nbr_.south != MPI_PROC_NULL )
    {
        pack_rows( ng, send_s_ );
        mpi_check( MPI_Isend( send_s_.data(), strip_y_dbl, MPI_DOUBLE, nbr_.south, tag_y_n2s, cart_, &req[nreq++] ),
            "MPI_Isend(S)" );
    }

    if ( nbr_.north != MPI_PROC_NULL )
    {
        pack_rows( ng + ny_local - ng, send_n_ );
        mpi_check( MPI_Isend( send_n_.data(), strip_y_dbl, MPI_DOUBLE, nbr_.north, tag_y_s2n, cart_, &req[nreq++] ),
            "MPI_Isend(N)" );
    }

    // 3. Complete all receives and sends before reusing the buffers.
    if ( nreq > 0 )
    {
        mpi_check( MPI_Waitall( nreq, req, MPI_STATUSES_IGNORE ), "MPI_Waitall(exchangeHalos2D)" );
    }

    if ( nbr_.south != MPI_PROC_NULL )
    {
        unpack_rows( 0, recv_s_ );
    }

    if ( nbr_.north != MPI_PROC_NULL )
    {
        unpack_rows( ng + ny_local, recv_n_ );
    }
}
