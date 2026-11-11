#pragma once

#include <mpi.h>

#include <vector>
#include <stdexcept>
#include <cstddef>


// Store a global interior-cell range [begin, begin + count).
struct Range1D
{
    int begin = 0;
    int count = 0;
};


// Store this rank's global x/y interior-cell ranges; ghost cells are excluded.
struct Subdomain2D
{
    Range1D x;
    Range1D y;

    int nx() const
    {
        return x.count;
    }

    int ny() const
    {
        return y.count;
    }

    int i_beg() const
    {
        return x.begin;
    }

    int j_beg() const
    {
        return y.begin;
    }
};


// Own the Cartesian communicator and reusable buffers for decomposition, halo exchange and reductions.
class MpiParallel
{
  public:
    // Store Cartesian neighbor ranks; MPI_PROC_NULL marks a non-periodic domain edge.
    struct Neighbors
    {
        int west = MPI_PROC_NULL;
        int east = MPI_PROC_NULL;
        int south = MPI_PROC_NULL;
        int north = MPI_PROC_NULL;
    };

    // px, py (in) - process-grid dimensions; px*py must equal the parent communicator size.
    // parent (in) - communicator used to create the Cartesian topology.
    MpiParallel( int px, int py, MPI_Comm parent = MPI_COMM_WORLD, bool periodic_x = false, bool periodic_y = false,
        bool reorder = false );

    MpiParallel( const MpiParallel & ) = delete;

    MpiParallel &operator=( const MpiParallel & ) = delete;

    MpiParallel( MpiParallel && ) noexcept;

    MpiParallel &operator=( MpiParallel && ) noexcept;

    ~MpiParallel();

    MPI_Comm parent_comm() const
    {
        return parent_;
    }

    // Return the Cartesian communicator used by exchanges and reductions.
    MPI_Comm cart_comm() const
    {
        return cart_;
    }

    int rank() const
    {
        return rank_;
    }

    int size() const
    {
        return size_;
    }

    int px() const
    {
        return px_;
    }

    int py() const
    {
        return py_;
    }

    int coord_x() const
    {
        return cx_;
    }

    int coord_y() const
    {
        return cy_;
    }

    Neighbors neighbors() const
    {
        return nbr_;
    }

    bool is_root() const
    {
        return rank_ == 0;
    }

    // Return this rank's global interior-cell ranges.
    Subdomain2D decompose( int global_nx, int global_ny ) const;

    // Synchronize all Cartesian ranks.
    void barrier() const;

    // Return the maximum local_val over all Cartesian ranks.
    double allreduce_max( double local_val ) const;

    // Return the minimum local_val over all Cartesian ranks.
    double allreduce_min( double local_val ) const;

    // Return the sum of local_val over all Cartesian ranks.
    double allreduce_sum( double local_val ) const;

    // Exchange packed cell components with Cartesian neighbors and fill ghost strips.
    // data (in/out) - packed cell components; cell (i,j) begins at (i + (nx_local + 2*ng)*j)*ncomp.
    // Interior values are sent to neighbors; received values replace ghost strips.
    void exchange_halos_2d( double *data, int nx_local, int ny_local, int ng, int ncomp ) const;

  private:
    // Split a global axis into contiguous blocks; distribute remainder cells to the first blocks.
    static Range1D split1_d( int global_n, int p, int coord );

    void init_cart_( bool periodic_x, bool periodic_y, bool reorder );

  private:
    MPI_Comm parent_ = MPI_COMM_NULL;
    MPI_Comm cart_ = MPI_COMM_NULL;

    int rank_ = 0;
    int size_ = 1;

    int px_ = 1;
    int py_ = 1;

    int cx_ = 0;
    int cy_ = 0;

    Neighbors nbr_{};

    // Reuse separate send/receive buffers for each coordinate direction.
    mutable std::vector<double> send_w_, recv_w_;
    mutable std::vector<double> send_e_, recv_e_;
    mutable std::vector<double> send_s_, recv_s_;
    mutable std::vector<double> send_n_, recv_n_;
};
