// mpi_parallel.cpp
// ----------------
// Minimal MPI helper for Cartesian domain decomposition.
//
// Responsibilities:
// - Build a 2D Cartesian communicator (px-by-py) using MPI_Cart_create.
// - Provide deterministic global-to-local decomposition for 2D grids.
// - Expose neighbor ranks (west/east/south/north).
// - Provide halo exchange helpers for ghost cells.
// - Provide a few common collectives (barrier, allreduce max/min/sum).
//
// Data layout assumptions for halo exchange:
// - Cell data is stored as an array of doubles with AoS layout per cell:
//     data[(cellIndex * ncomp) + c]
// - For 2D, the ghosted array is flattened row-major:
//     cellIndex = I + (nxLocal + 2*ng) * J
//
#include "mpi_parallel.hpp"

#include <algorithm>
#include <cstring>
#include <string>

namespace mpi_parallel {

// Small helper: convert MPI error codes into C++ exceptions with context.
static inline void mpiCheck(int err, const char* where) {
    if (err == MPI_SUCCESS) return;
    throw std::runtime_error(std::string("MPI error at: ") + where);
}

// Split a global index range [0, globalN) into P contiguous blocks.
//
// Returns {begin,count} for the block owned by process coordinate `coord`.
// The first `rem = globalN % P` blocks get one extra element.
// This is the standard "block" decomposition used by many MPI codes.
Range1D MpiParallel::split1D(int globalN, int P, int coord) {
    if (P <= 0) throw std::runtime_error("MpiParallel::split1D: P must be > 0");
    if (globalN <= 0) throw std::runtime_error("MpiParallel::split1D: globalN must be > 0");
    if (coord < 0 || coord >= P) throw std::runtime_error("MpiParallel::split1D: coord out of range");

    const int base = globalN / P;
    const int rem  = globalN % P;

    Range1D r;
    r.count = base + (coord < rem ? 1 : 0);
    r.begin = coord * base + std::min(coord, rem);
    return r;
}

// Build the MPI Cartesian topology.
// - px, py: number of ranks in x and y direction
// - parent: usually MPI_COMM_WORLD
// - periodicX/Y: whether the topology wraps around (not used for physical BCs)
// - reorder: allow MPI to reorder ranks for performance
MpiParallel::MpiParallel(int px, int py,
                         MPI_Comm parent,
                         bool periodicX,
                         bool periodicY,
                         bool reorder)
: parent_(parent), px_(px), py_(py) {
    if (px_ <= 0 || py_ <= 0) {
        throw std::runtime_error("MpiParallel: px and py must be > 0");
    }

    mpiCheck(MPI_Comm_rank(parent_, &rank_), "MPI_Comm_rank");
    mpiCheck(MPI_Comm_size(parent_, &size_), "MPI_Comm_size");

    if (px_ * py_ != size_) {
        throw std::runtime_error("MpiParallel: px*py must equal MPI_Comm_size");
    }

    initCart_(periodicX, periodicY, reorder);
}

void MpiParallel::initCart_(bool periodicX, bool periodicY, bool reorder) {
    // Cartesian topology dimensions are ordered as (y, x) so that:
    //   coords[0] = cy (y-coordinate), coords[1] = cx (x-coordinate)
    // This matches our use of x as the fast-varying index in flattened arrays.
    int dims[2]    = { py_, px_ };
    int periods[2] = { periodicY ? 1 : 0, periodicX ? 1 : 0 };
    int reo        = reorder ? 1 : 0;

    mpiCheck(MPI_Cart_create(parent_, 2, dims, periods, reo, &cart_), "MPI_Cart_create");
    if (cart_ == MPI_COMM_NULL) {
        throw std::runtime_error("MpiParallel: MPI_Cart_create returned MPI_COMM_NULL");
    }

    // update rank/size in cart communicator (in case reorder=true)
    mpiCheck(MPI_Comm_rank(cart_, &rank_), "MPI_Comm_rank(cart)");
    mpiCheck(MPI_Comm_size(cart_, &size_), "MPI_Comm_size(cart)");

    int coords[2] = {0, 0};
    mpiCheck(MPI_Cart_coords(cart_, rank_, 2, coords), "MPI_Cart_coords");
    cy_ = coords[0];
    cx_ = coords[1];

    // Neighbor ranks in the Cartesian grid.
    // MPI_Cart_shift returns MPI_PROC_NULL on physical edges when not periodic.
    // neighbors
    // axis 0: y
    mpiCheck(MPI_Cart_shift(cart_, 0, 1, &nbr_.south, &nbr_.north), "MPI_Cart_shift(y)");
    // axis 1: x
    mpiCheck(MPI_Cart_shift(cart_, 1, 1, &nbr_.west,  &nbr_.east ), "MPI_Cart_shift(x)");
}

// Move support: allows storing MpiParallel in containers or returning it by value.
MpiParallel::MpiParallel(MpiParallel&& o) noexcept {
    *this = std::move(o);
}

// Move support: allows storing MpiParallel in containers or returning it by value.
MpiParallel& MpiParallel::operator=(MpiParallel&& o) noexcept {
    if (this == &o) return *this;

    parent_ = o.parent_;
    cart_   = o.cart_;

    rank_ = o.rank_;
    size_ = o.size_;

    px_ = o.px_;
    py_ = o.py_;

    cx_ = o.cx_;
    cy_ = o.cy_;

    nbr_ = o.nbr_;

    sendW_ = std::move(o.sendW_); recvW_ = std::move(o.recvW_);
    sendE_ = std::move(o.sendE_); recvE_ = std::move(o.recvE_);
    sendS_ = std::move(o.sendS_); recvS_ = std::move(o.recvS_);
    sendN_ = std::move(o.sendN_); recvN_ = std::move(o.recvN_);

    o.parent_ = MPI_COMM_NULL;
    o.cart_   = MPI_COMM_NULL;
    return *this;
}

// Free the Cartesian communicator (if created).
MpiParallel::~MpiParallel() {
    if (cart_ != MPI_COMM_NULL) {
        MPI_Comm_free(&cart_);
        cart_ = MPI_COMM_NULL;
    }
}

// Compute this-rank 2D subdomain for a global (globalNx x globalNy) interior grid.
// The returned begin/count are global interior indices (0-based).
Subdomain2D MpiParallel::decompose(int globalNx, int globalNy) const {
    if (globalNx <= 0 || globalNy <= 0) {
        throw std::runtime_error("MpiParallel::decompose: globalNx/globalNy must be > 0");
    }

    Subdomain2D sub;
    sub.x = split1D(globalNx, px_, cx_);
    sub.y = split1D(globalNy, py_, cy_);

    if (sub.nx() <= 0 || sub.ny() <= 0) {
        throw std::runtime_error("MpiParallel::decompose: local nx/ny <= 0 (grid too small for px/py)");
    }
    return sub;
}


// Synchronize all ranks in the Cartesian communicator.
void MpiParallel::barrier() const {
    mpiCheck(MPI_Barrier(cart_), "MPI_Barrier");
}

// Global maximum across ranks (cart communicator).
double MpiParallel::allreduceMax(double localVal) const {
    double out = 0.0;
    mpiCheck(MPI_Allreduce(&localVal, &out, 1, MPI_DOUBLE, MPI_MAX, cart_), "MPI_Allreduce(MAX)");
    return out;
}

// Global minimum across ranks (cart communicator).
double MpiParallel::allreduceMin(double localVal) const {
    double out = 0.0;
    mpiCheck(MPI_Allreduce(&localVal, &out, 1, MPI_DOUBLE, MPI_MIN, cart_), "MPI_Allreduce(MIN)");
    return out;
}

// Global sum across ranks (cart communicator).
double MpiParallel::allreduceSum(double localVal) const {
    double out = 0.0;
    mpiCheck(MPI_Allreduce(&localVal, &out, 1, MPI_DOUBLE, MPI_SUM, cart_), "MPI_Allreduce(SUM)");
    return out;
}

// Exchange 2D ghost layers with Cartesian neighbors.
//
// Arguments:
// - data: pointer to the ghosted array, size (nxLocal+2*ng)*(nyLocal+2*ng)*ncomp
// - nxLocal, nyLocal: interior sizes owned by this rank
// - ng: number of ghost layers
// - ncomp: number of double components per cell (e.g., 4 for 2D conservative state)
//
// Communication pattern:
//   1) Post Irecv for each existing neighbor
//   2) Pack interior strips into contiguous buffers and Isend
//   3) Waitall
//   4) Unpack received buffers into ghost layers
void MpiParallel::exchangeHalos2D(double* data,
                                 int nxLocal, int nyLocal,
                                 int ng, int ncomp) const
{
    if (!data) throw std::runtime_error("exchangeHalos2D: data is null");
    if (nxLocal <= 0 || nyLocal <= 0) throw std::runtime_error("exchangeHalos2D: nxLocal/nyLocal must be > 0");
    if (ng < 0) throw std::runtime_error("exchangeHalos2D: ng must be >= 0");
    if (ncomp <= 0) throw std::runtime_error("exchangeHalos2D: ncomp must be > 0");
    if (ng == 0) return;

    const int nxTot = nxLocal + 2 * ng;
    const int nyTot = nyLocal + 2 * ng;

    // Addressing helpers: return a pointer to the first component of cell (I,J).
    // Layout is AoS: [cell0 comp0..compN-1][cell1 comp0..]...
    auto cellPtr = [&](int I, int J) -> double* {
        const int k = I + nxTot * J;
        return data + static_cast<std::size_t>(k) * static_cast<std::size_t>(ncomp);
    };
    auto cellPtrC = [&](int I, int J) -> const double* {
        const int k = I + nxTot * J;
        return data + static_cast<std::size_t>(k) * static_cast<std::size_t>(ncomp);
    };

    // Buffer sizes (in doubles).
    // We send/recv full strips that include ghost rows/cols in the orthogonal direction.
    const int stripX_cells = ng * nyTot;            // vertical strip: ng columns x nyTot rows
    const int stripX_dbl   = stripX_cells * ncomp;

    const int stripY_cells = nxTot * ng;            // horizontal strip: nxTot cols x ng rows
    const int stripY_dbl   = stripY_cells * ncomp;

    // Pack/unpack vertical strips of width ng (columns). Used for west/east exchange.
    auto packCols = [&](int I0, std::vector<double>& buf) {
        buf.resize(stripX_dbl);
        int t = 0;
        for (int J = 0; J < nyTot; ++J) {
            for (int di = 0; di < ng; ++di) {
                const double* src = cellPtrC(I0 + di, J);
                std::memcpy(&buf[t], src, sizeof(double) * ncomp);
                t += ncomp;
            }
        }
    };

    auto unpackCols = [&](int I0, const std::vector<double>& buf) {
        int t = 0;
        for (int J = 0; J < nyTot; ++J) {
            for (int di = 0; di < ng; ++di) {
                double* dst = cellPtr(I0 + di, J);
                std::memcpy(dst, &buf[t], sizeof(double) * ncomp);
                t += ncomp;
            }
        }
    };

    // Pack/unpack horizontal strips of height ng (rows). Used for south/north exchange.
    auto packRows = [&](int J0, std::vector<double>& buf) {
        buf.resize(stripY_dbl);
        int t = 0;
        for (int dj = 0; dj < ng; ++dj) {
            for (int I = 0; I < nxTot; ++I) {
                const double* src = cellPtrC(I, J0 + dj);
                std::memcpy(&buf[t], src, sizeof(double) * ncomp);
                t += ncomp;
            }
        }
    };

    auto unpackRows = [&](int J0, const std::vector<double>& buf) {
        int t = 0;
        for (int dj = 0; dj < ng; ++dj) {
            for (int I = 0; I < nxTot; ++I) {
                double* dst = cellPtr(I, J0 + dj);
                std::memcpy(dst, &buf[t], sizeof(double) * ncomp);
                t += ncomp;
            }
        }
    };

    // Message tags: distinct per direction.
    // Using different tags avoids accidental cross-matching between sends/recvs.
    constexpr int TAG_X_W2E = 200; // data travelling west -> east
    constexpr int TAG_X_E2W = 201; // data travelling east -> west
    constexpr int TAG_Y_S2N = 210; // data travelling south -> north
    constexpr int TAG_Y_N2S = 211; // data travelling north -> south

    MPI_Request req[8];
    int nreq = 0;

    // --- 1) Post receives first (avoids deadlocks with blocking sends) ---
    if (nbr_.west != MPI_PROC_NULL) {
        recvW_.resize(stripX_dbl);
        mpiCheck(MPI_Irecv(recvW_.data(), stripX_dbl, MPI_DOUBLE, nbr_.west, TAG_X_W2E, cart_, &req[nreq++]),
                 "MPI_Irecv(W)");
    }
    if (nbr_.east != MPI_PROC_NULL) {
        recvE_.resize(stripX_dbl);
        mpiCheck(MPI_Irecv(recvE_.data(), stripX_dbl, MPI_DOUBLE, nbr_.east, TAG_X_E2W, cart_, &req[nreq++]),
                 "MPI_Irecv(E)");
    }
    if (nbr_.south != MPI_PROC_NULL) {
        recvS_.resize(stripY_dbl);
        mpiCheck(MPI_Irecv(recvS_.data(), stripY_dbl, MPI_DOUBLE, nbr_.south, TAG_Y_S2N, cart_, &req[nreq++]),
                 "MPI_Irecv(S)");
    }
    if (nbr_.north != MPI_PROC_NULL) {
        recvN_.resize(stripY_dbl);
        mpiCheck(MPI_Irecv(recvN_.data(), stripY_dbl, MPI_DOUBLE, nbr_.north, TAG_Y_N2S, cart_, &req[nreq++]),
                 "MPI_Irecv(N)");
    }

    // --- 2) Pack interior strips and post sends ---
    // west: send our left interior strip (I0=ng) to west (as east->west)
    if (nbr_.west != MPI_PROC_NULL) {
        packCols(ng, sendW_);
        mpiCheck(MPI_Isend(sendW_.data(), stripX_dbl, MPI_DOUBLE, nbr_.west, TAG_X_E2W, cart_, &req[nreq++]),
                 "MPI_Isend(W)");
    }
    // east: send our right interior strip (I0=ng+nxLocal-ng) to east (as west->east)
    if (nbr_.east != MPI_PROC_NULL) {
        packCols(ng + nxLocal - ng, sendE_);
        mpiCheck(MPI_Isend(sendE_.data(), stripX_dbl, MPI_DOUBLE, nbr_.east, TAG_X_W2E, cart_, &req[nreq++]),
                 "MPI_Isend(E)");
    }
    // south: send our bottom interior strip (J0=ng) to south (as north->south)
    if (nbr_.south != MPI_PROC_NULL) {
        packRows(ng, sendS_);
        mpiCheck(MPI_Isend(sendS_.data(), stripY_dbl, MPI_DOUBLE, nbr_.south, TAG_Y_N2S, cart_, &req[nreq++]),
                 "MPI_Isend(S)");
    }
    // north: send our top interior strip (J0=ng+nyLocal-ng) to north (as south->north)
    if (nbr_.north != MPI_PROC_NULL) {
        packRows(ng + nyLocal - ng, sendN_);
        mpiCheck(MPI_Isend(sendN_.data(), stripY_dbl, MPI_DOUBLE, nbr_.north, TAG_Y_S2N, cart_, &req[nreq++]),
                 "MPI_Isend(N)");
    }

    // --- 3) Wait for all communications to complete ---
    if (nreq > 0) {
        mpiCheck(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE), "MPI_Waitall(exchangeHalos2D)");
    }

    // --- 4) Unpack into ghost layers (write ghost cells only) ---
    // left ghost: I0=0
    if (nbr_.west != MPI_PROC_NULL) {
        unpackCols(0, recvW_);
    }
    // right ghost: I0=ng+nxLocal
    if (nbr_.east != MPI_PROC_NULL) {
        unpackCols(ng + nxLocal, recvE_);
    }
    // bottom ghost: J0=0
    if (nbr_.south != MPI_PROC_NULL) {
        unpackRows(0, recvS_);
    }
    // top ghost: J0=ng+nyLocal
    if (nbr_.north != MPI_PROC_NULL) {
        unpackRows(ng + nyLocal, recvN_);
    }
}


} // namespace mpi_parallel
