// mpi_parallel.hpp
// MPI helper utilities for simple Cartesian domain decomposition.
//
// This module is intentionally small: it does NOT implement a full parallel mesh.
// Instead, it provides:
//   - A 2D Cartesian communicator (px-by-py rank layout)
//   - Convenience functions to compute each rank's global index range
//   - Halo exchange for cell-centered arrays with ghost layers
//   - A few common collectives (barrier, allreduce)
//
// The solvers keep their own arrays (including ghosts) and call exchangeHalos*()
// before reconstruction/flux evaluation so that stencils see consistent neighbor data.

#pragma once

#include <mpi.h>

#include <vector>
#include <stdexcept>
#include <cstddef>

namespace mpi_parallel {

// Range1D
// -------
// A contiguous 0-based index range on one axis: [begin, begin+count).
// Used to describe the global ownership of interior cells for each MPI rank.
struct Range1D {
    int begin = 0;
    int count = 0;
};

// Subdomain1D
// -----------
// This-rank 1D subdomain (interior only, no ghosts).
// The solver allocates ghosts itself; this struct only reports the global start
// index and the number of interior cells owned by this rank.
struct Subdomain1D {
    Range1D x;

    int nx() const { return x.count; }
    int iBeg() const { return x.begin; }
};

// Subdomain2D
// -----------
// This-rank 2D subdomain (interior only, no ghosts).
// x and y ranges describe the global ownership of interior cells.
struct Subdomain2D {
    Range1D x;
    Range1D y;

    int nx() const { return x.count; }
    int ny() const { return y.count; }
    int iBeg() const { return x.begin; }
    int jBeg() const { return y.begin; }
};

// MpiParallel
// -----------
// Lightweight wrapper around an MPI Cartesian communicator.
//
// Typical workflow:
//   1) Create MpiParallel(px, py)
//   2) Ask for this rank's subdomain: decompose(...) or decompose1D(...)
//   3) During time stepping, call exchangeHalos*() to update ghost layers
//
// Important:
// - The runtime must launch with: mpirun -np (px*py)
// - For 1D runs, require py==1 (split only along x).
class MpiParallel {
public:
    struct Neighbors {
        int west  = MPI_PROC_NULL; // -x
        int east  = MPI_PROC_NULL; // +x
        int south = MPI_PROC_NULL; // -y
        int north = MPI_PROC_NULL; // +y
    };

    // Construct a px-by-py Cartesian communicator on top of `parent`.
    // The communicator is freed in the destructor.
    // Manual 2D block decomposition:
    // - px blocks along x, py blocks along y
    // - requires px*py == MPI_Comm_size(parent)
    // - default: non-periodic, no reorder
    MpiParallel(int px, int py,
                MPI_Comm parent = MPI_COMM_WORLD,
                bool periodicX = false,
                bool periodicY = false,
                bool reorder   = false);

    MpiParallel(const MpiParallel&) = delete;
    MpiParallel& operator=(const MpiParallel&) = delete;

    MpiParallel(MpiParallel&&) noexcept;
    MpiParallel& operator=(MpiParallel&&) noexcept;

    ~MpiParallel();

    // The communicator passed in at construction (usually MPI_COMM_WORLD).
    MPI_Comm parentComm() const { return parent_; }
    // The created 2D Cartesian communicator used for neighbor exchanges/collectives.
    MPI_Comm cartComm()   const { return cart_;   }

    int rank() const { return rank_; }
    int size() const { return size_; }

    int px() const { return px_; }
    int py() const { return py_; }

    // Cartesian coords (0..px-1, 0..py-1)
    int coordX() const { return cx_; }
    int coordY() const { return cy_; }

    // Neighbor ranks in the Cartesian grid (MPI_PROC_NULL at physical boundaries).
    Neighbors neighbors() const { return nbr_; }

    bool isRoot() const { return rank_ == 0; }

    // Compute this-rank subdomain for global grid (globalNx, globalNy).
    // (interior only; ghosts are managed by solver arrays, not by this struct)
    Subdomain2D decompose(int globalNx, int globalNy) const;

    // Compute this-rank subdomain for a 1D global grid (globalNx) using px() along x.
    // Requirement: py() == 1 for 1D runs.
    Subdomain1D decompose1D(int globalNx) const;

    // Collectives on cart communicator
    void barrier() const;

    double allreduceMax(double localVal) const;
    double allreduceMin(double localVal) const;
    double allreduceSum(double localVal) const;

    // Halo exchange for cell-centered arrays stored as contiguous doubles.
    //
    // Required memory layout:
    // - local interior: nxLocal * nyLocal cells
    // - array storage includes ghosts: nxTot = nxLocal + 2*ng, nyTot = nyLocal + 2*ng
    // - row-major cells: cellIndex = i + nxTot*j, where i in [0..nxTot-1], j in [0..nyTot-1]
    // - AoS components: each cell has ncomp doubles contiguous:
    //   (e.g., ncomp=3 for 1D conservative, ncomp=4 for 2D conservative)
    //      cellPtr = data + (cellIndex * ncomp)
    //
    // What it does:
    // - exchanges ng columns with west/east and ng rows with south/north
    // - fills ghost layers (i<ng, i>=ng+nxLocal, j<ng, j>=ng+nyLocal)
    //
    void exchangeHalos2D(double* data,
                         int nxLocal, int nyLocal,
                         int ng, int ncomp) const;

    // Halo exchange for 1D cell-centered arrays stored as contiguous doubles.
    //
    // Required memory layout:
    // - local interior: nxLocal cells
    // - array storage includes ghosts: nxTot = nxLocal + 2*ng
    // - cells in 1D: cellIndex = i, where i in [0..nxTot-1]
    // - AoS components: each cell has ncomp doubles contiguous:
    //   (e.g., ncomp=3 for (rho, rho*u, rho*E))
    //      cellPtr = data + (cellIndex * ncomp)
    //
    // What it does:
    // - exchanges ng cells with west/east neighbors (along x)
    // - fills ghost layers (i<ng, i>=ng+nxLocal)
    //
    void exchangeHalos1D(double* data,
                         int nxLocal,
                         int ng, int ncomp) const;

private:
    // Split a global 1D range into P blocks and return the block for a given coord.
    // The first (globalN % P) blocks get one extra cell.
    static Range1D split1D(int globalN, int P, int coord);

    void initCart_(bool periodicX, bool periodicY, bool reorder);

private:
    MPI_Comm parent_ = MPI_COMM_NULL;
    MPI_Comm cart_   = MPI_COMM_NULL;

    int rank_ = 0;
    int size_ = 1;

    int px_ = 1;
    int py_ = 1;

    int cx_ = 0;
    int cy_ = 0;

    Neighbors nbr_{};

    // Internal reusable buffers (separated per direction to avoid aliasing)
    mutable std::vector<double> sendW_, recvW_;
    mutable std::vector<double> sendE_, recvE_;
    mutable std::vector<double> sendS_, recvS_;
    mutable std::vector<double> sendN_, recvN_;
};

} // namespace mpi_parallel
