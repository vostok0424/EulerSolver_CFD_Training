#include "io2d.hpp"
#include "state.hpp"
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <iomanip>
#include <limits>
#include <locale>
#include <cstdio>

// io2d.cpp
// --------
// Legacy VTK output for 2D solutions.
//
// This file writes a single ASCII legacy VTK rectilinear-grid file (.vtk)
// for post-processing in ParaView or other VTK-capable tools.
//
// The solution is cell-centered and stored as CELL_DATA on a 2D
// RECTILINEAR_GRID. The x- and y-coordinates written to the VTK file are the
// cell-edge coordinates spanning the physical domain [x0, x1] x [y0, y1].
//
// Output fields:
//   rho      : density
//   velocity : vector field (u, v, 0)
//   p        : pressure
//   rho_u    : x-momentum
//   rho_v    : y-momentum
//   E        : total energy from the conservative state
//
// MPI:
// - writeVTK2D_GatherMPI gathers each rank's *interior* cell data to rank 0,
//   assembles a global field, then calls writeVTK2D to write a single merged VTK file.

// MPI gather/assembly packs Vec4 into raw doubles. Ensure it is tightly-packed.
static_assert(std::is_trivially_copyable_v<Vec4>, "Vec4 must be trivially copyable for MPI I/O packing");
static_assert(sizeof(Vec4) == 4 * sizeof(double), "Vec4 must be exactly 4 doubles (no padding)");

void writeVTK2D(const std::string& filename,
                const std::vector<Vec4>& U,
                int nx, int ny, int ng,
                double x0, double x1,
                double y0, double y1,
                double gamma)
{
    if (nx <= 0 || ny <= 0) throw std::runtime_error("writeVTK2D: nx, ny must be > 0");
    if (ng < 0) throw std::runtime_error("writeVTK2D: ng must be >= 0");

    const int nxTot = nx + 2 * ng;
    auto idx = [&](int i, int j) { return i + nxTot * j; };

    const int nxPts = nx + 1;
    const int nyPts = ny + 1;
    const double dx = (x1 - x0) / static_cast<double>(nx);
    const double dy = (y1 - y0) / static_cast<double>(ny);

    const std::string tmpFile = filename + ".tmp";
    std::ofstream out(tmpFile);
    if (!out) throw std::runtime_error("writeVTK2D: cannot open file: " + tmpFile);

    out.imbue(std::locale::classic());
    out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    out.setf(std::ios::scientific);
    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    // Legacy VTK header.
    out << "# vtk DataFile Version 3.0\n";
    out << "2D Euler solution\n";
    out << "ASCII\n";
    out << "DATASET RECTILINEAR_GRID\n";
    out << "DIMENSIONS " << nxPts << ' ' << nyPts << " 1\n";

    out << "X_COORDINATES " << nxPts << " double\n";
    for (int i = 0; i < nxPts; ++i) {
        const double x = x0 + static_cast<double>(i) * dx;
        out << x << "\n";
    }

    out << "Y_COORDINATES " << nyPts << " double\n";
    for (int j = 0; j < nyPts; ++j) {
        const double y = y0 + static_cast<double>(j) * dy;
        out << y << "\n";
    }

    out << "Z_COORDINATES 1 double\n";
    out << 0.0 << "\n";

    out << "CELL_DATA " << (nx * ny) << "\n";

    out << "SCALARS rho double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Vec4& Q = U[idx(ng + i, ng + j)];
            const auto W = EosIdealGas<2>::consToPrim(Q, gamma);
            out << W.rho << "\n";
        }
    }

    out << "VECTORS velocity double\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Vec4& Q = U[idx(ng + i, ng + j)];
            const auto W = EosIdealGas<2>::consToPrim(Q, gamma);
            out << W.u[0] << ' ' << W.u[1] << ' ' << 0.0 << "\n";
        }
    }

    out << "SCALARS p double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Vec4& Q = U[idx(ng + i, ng + j)];
            const auto W = EosIdealGas<2>::consToPrim(Q, gamma);
            out << W.p << "\n";
        }
    }

    out << "SCALARS rho_u double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Vec4& Q = U[idx(ng + i, ng + j)];
            out << Q[1] << "\n";
        }
    }

    out << "SCALARS rho_v double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Vec4& Q = U[idx(ng + i, ng + j)];
            out << Q[2] << "\n";
        }
    }

    out << "SCALARS E double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Vec4& Q = U[idx(ng + i, ng + j)];
            out << Q[3] << "\n";
        }
    }

    out.flush();
    out.close();

    if (std::rename(tmpFile.c_str(), filename.c_str()) != 0) {
        throw std::runtime_error("writeVTK2D: failed to rename tmp file to final output: " + filename);
    }
}

// Gather local interior cell data to rank 0 and write a single merged VTK file.
// Each rank provides ONLY its interior cells (ghost cells are excluded).
void writeVTK2D_GatherMPI(const std::string& filename,
                          const std::vector<Vec4>& Ulocal,
                          int nxLocal, int nyLocal, int ng,
                          int iBeg, int jBeg,
                          int nxGlobal, int nyGlobal,
                          double x0, double x1,
                          double y0, double y1,
                          double gamma,
                          MPI_Comm comm)
{
    if (nxLocal <= 0 || nyLocal <= 0) throw std::runtime_error("writeVTK2D_GatherMPI: nxLocal/nyLocal must be > 0");
    if (nxGlobal <= 0 || nyGlobal <= 0) throw std::runtime_error("writeVTK2D_GatherMPI: nxGlobal/nyGlobal must be > 0");
    if (ng < 1) throw std::runtime_error("writeVTK2D_GatherMPI: ng must be >= 1");

    // Rank information for collective operations.
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Pack this rank's interior conservative variables (no ghosts) into a raw double buffer.
    // This keeps MPI_Gatherv simple and portable.
    const int nxTotLoc = nxLocal + 2 * ng;
    const int nyTotLoc = nyLocal + 2 * ng;
    (void)nyTotLoc;

    // Local ghosted indexing for this rank.
    auto idxLoc = [&](int i, int j) { return i + nxTotLoc * j; };

    // Payload size in doubles: (nxLocal*nyLocal) cells * 4 components.
    const int ncomp = 4;
    const int localCells = nxLocal * nyLocal;
    const int localCount = localCells * ncomp; // in doubles

    std::vector<double> send(localCount);
    int t = 0;
    for (int j = 0; j < nyLocal; ++j) {
        for (int i = 0; i < nxLocal; ++i) {
            const Vec4& Q = Ulocal[idxLoc(ng + i, ng + j)];
            send[t + 0] = Q[0];
            send[t + 1] = Q[1];
            send[t + 2] = Q[2];
            send[t + 3] = Q[3];
            t += ncomp;
        }
    }

    // Send metadata so root knows where this rank's block sits in the global array.
    int meta[4] = { iBeg, jBeg, nxLocal, nyLocal };
    std::vector<int> allMeta; // size*4 on root
    if (rank == 0) allMeta.resize(static_cast<size_t>(size) * 4u);
    MPI_Gather(meta, 4, MPI_INT,
               rank == 0 ? allMeta.data() : nullptr, 4, MPI_INT,
               0, comm);

    // Gather payload sizes, then gather the packed buffers using MPI_Gatherv.
    std::vector<int> counts; // in doubles
    std::vector<int> displs; // in doubles
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
    }
    MPI_Gather(&localCount, 1, MPI_INT,
               rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
               0, comm);

    std::vector<double> recv;
    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = total;
            total += counts[r];
        }
        recv.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(send.data(), localCount, MPI_DOUBLE,
                rank == 0 ? recv.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE,
                0, comm);

    // Only rank 0 assembles and writes the merged file.
    if (rank != 0) return;

    // Assemble a global field with a single ghost layer (ngOut=1) to match writeVTK2D's interface.
    // The ghost values themselves are not important for output; writeVTK2D reads only interior.
    const int ngOut = 1;
    const int nxTotG = nxGlobal + 2 * ngOut;
    const int nyTotG = nyGlobal + 2 * ngOut;
    auto idxG = [&](int i, int j) { return i + nxTotG * j; };

    std::vector<Vec4> Uglobal(static_cast<size_t>(nxTotG) * static_cast<size_t>(nyTotG), Vec4{});

    // Unpack each rank's block into its correct global location.
    for (int r = 0; r < size; ++r) {
        const int ib = allMeta[4 * r + 0];
        const int jb = allMeta[4 * r + 1];
        const int nxr = allMeta[4 * r + 2];
        const int nyr = allMeta[4 * r + 3];

        const int base = displs[r];
        const int expected = nxr * nyr * ncomp;
        if (counts[r] != expected) {
            throw std::runtime_error("writeVTK2D_GatherMPI: inconsistent counts vs meta for rank " + std::to_string(r));
        }

        for (int j = 0; j < nyr; ++j) {
            for (int i = 0; i < nxr; ++i) {
                const int gi = ib + i;
                const int gj = jb + j;
                if (gi < 0 || gi >= nxGlobal || gj < 0 || gj >= nyGlobal) {
                    throw std::runtime_error("writeVTK2D_GatherMPI: rank block out of global bounds");
                }

                const int locCell = (i + nxr * j) * ncomp;
                Vec4 Q{};
                Q[0] = recv[base + locCell + 0];
                Q[1] = recv[base + locCell + 1];
                Q[2] = recv[base + locCell + 2];
                Q[3] = recv[base + locCell + 3];

                Uglobal[idxG(ngOut + gi, ngOut + gj)] = Q;
            }
        }
    }

    // Write a single merged VTK containing cell-centered data.
    writeVTK2D(filename, Uglobal, nxGlobal, nyGlobal, ngOut, x0, x1, y0, y1, gamma);
}
