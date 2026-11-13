#pragma once

#include <mpi.h>

#include "structures.hpp"

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

    // Return the local number of interior cells along x.
    int nx() const
    {
        return x.count;
    }

    // Return the local number of interior cells along y.
    int ny() const
    {
        return y.count;
    }

    // Return the global x index of the first local interior cell.
    int i_beg() const
    {
        return x.begin;
    }

    // Return the global y index of the first local interior cell.
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

    // Create the Cartesian MPI topology and identify neighboring ranks.
    //
    // px (in) - number of process columns along x
    // py (in) - number of process rows along y
    // parent (in) - parent MPI communicator
    // periodic_x (in) - whether the Cartesian topology is periodic along x
    // periodic_y (in) - whether the Cartesian topology is periodic along y
    // reorder (in) - whether MPI may reorder Cartesian ranks
    MpiParallel( int px, int py, MPI_Comm parent = MPI_COMM_WORLD, bool periodic_x = false, bool periodic_y = false,
        bool reorder = false );

    // Disable copying of the MPI communicator owner.
    //
    // source (in) - source MPI helper; copying is disabled and moving transfers ownership
    MpiParallel( const MpiParallel & ) = delete;

    // Disable copying of the MPI communicator owner.
    //
    // source (in) - source MPI helper; copying is disabled and moving transfers ownership
    MpiParallel &operator=( const MpiParallel & ) = delete;

    // Transfer MPI communicator and buffer ownership from the source helper.
    //
    // source (in/out) - source MPI helper; copying is disabled and moving transfers ownership
    MpiParallel( MpiParallel && ) noexcept;

    // Transfer MPI communicator and buffer ownership from the source helper.
    //
    // source (in/out) - source MPI helper; copying is disabled and moving transfers ownership
    MpiParallel &operator=( MpiParallel && ) noexcept;

    // Release owned resources during object destruction.
    ~MpiParallel();

    // Return the parent communicator without transferring ownership.
    MPI_Comm parent_comm() const
    {
        return parent_;
    }

    // Return the Cartesian communicator used by exchanges and reductions.
    MPI_Comm cart_comm() const
    {
        return cart_;
    }

    // Return this process's rank in the Cartesian communicator.
    int rank() const
    {
        return rank_;
    }

    // Return the number of ranks in the Cartesian communicator.
    int size() const
    {
        return size_;
    }

    // Return the number of Cartesian process columns.
    int px() const
    {
        return px_;
    }

    // Return the number of Cartesian process rows.
    int py() const
    {
        return py_;
    }

    // Return this process's Cartesian x coordinate.
    int coord_x() const
    {
        return cx_;
    }

    // Return this process's Cartesian y coordinate.
    int coord_y() const
    {
        return cy_;
    }

    // Return the four Cartesian neighbor ranks; MPI_PROC_NULL marks a physical edge.
    Neighbors neighbors() const
    {
        return nbr_;
    }

    // Check whether this process is Cartesian rank zero.
    bool is_root() const
    {
        return rank_ == 0;
    }

    // Return this rank's global interior-cell ranges.
    //
    // global_nx (in) - global number of interior cells along x
    // global_ny (in) - global number of interior cells along y
    Subdomain2D decompose( int global_nx, int global_ny ) const;

    // Synchronize all Cartesian ranks.
    void barrier() const;

    // Return the maximum local_val over all Cartesian ranks.
    //
    // local_val (in) - this rank's contribution to the collective reduction
    double allreduce_max( double local_val ) const;

    // Return the minimum local_val over all Cartesian ranks.
    //
    // local_val (in) - this rank's contribution to the collective reduction
    double allreduce_min( double local_val ) const;

    // Return the sum of local_val over all Cartesian ranks.
    //
    // local_val (in) - this rank's contribution to the collective reduction
    double allreduce_sum( double local_val ) const;

    // Pack boundary state strips directly and exchange them with Cartesian neighbors.
    //
    // state (in/out) - ghosted conservative field; only received ghost strips are modified
    // nx_local (in) - local number of interior cells along x
    // ny_local (in) - local number of interior cells along y
    // ng (in) - number of ghost-cell layers on each side
    //
    // Interior values are sent to neighbors; received values replace ghost strips.
    void exchange_halos_2d( std::vector<Conservative_vector> &state, int nx_local, int ny_local, int ng ) const;

  private:
    // Split a global axis into contiguous blocks; distribute remainder cells to the first blocks.
    //
    // global_n (in) - global number of cells along the split axis
    // p (in) - number of process blocks along the split axis
    // coord (in) - process coordinate along the split axis
    static Range1D split1_d( int global_n, int p, int coord );

    // Create the Cartesian topology with MPI dimensions ordered as (y, x).
    //
    // periodic_x (in) - whether the Cartesian topology is periodic along x
    // periodic_y (in) - whether the Cartesian topology is periodic along y
    // reorder (in) - whether MPI may reorder Cartesian ranks
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
