#include "io.hpp"
#include "state.hpp"
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <iomanip>
#include <limits>
#include <locale>
#include <cstdio>

// io.cpp
// --------
// Legacy VTK output for 2D solutions.
//
// This file writes a single ASCII legacy VTK rectilinear-grid file (.vtk)
// for post-processing in ParaView or other VTK-capable tools.
//
// The solution is written as POINT_DATA on a 2D RECTILINEAR_GRID.
// The x- and y-coordinates written to the VTK file are the cell-edge
// coordinates spanning the physical domain [x0, x1] x [y0, y1]. Node values
// are obtained from the cell-centered finite-volume solution by averaging the
// surrounding cells.
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
//   The merged output is converted to POINT_DATA inside writeVTK2D.

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
    auto idxP = [&](int i, int j) { return i + nxPts * j; };
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

    const int nCells = nx * ny;
    const int nPts = nxPts * nyPts;
    auto idxC = [&](int i, int j) { return i + nx * j; };

    std::vector<double> rhoC(static_cast<std::size_t>(nCells));
    std::vector<double> uC(static_cast<std::size_t>(nCells));
    std::vector<double> vC(static_cast<std::size_t>(nCells));
    std::vector<double> pC(static_cast<std::size_t>(nCells));
    std::vector<double> rhoUC(static_cast<std::size_t>(nCells));
    std::vector<double> rhoVC(static_cast<std::size_t>(nCells));
    std::vector<double> EC(static_cast<std::size_t>(nCells));

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Vec4& Q = U[idx(ng + i, ng + j)];
            const auto W = EosIdealGas<2>::consToPrim(Q, gamma);
            const std::size_t kc = static_cast<std::size_t>(idxC(i, j));
            rhoC[kc] = W.rho;
            uC[kc] = W.u[0];
            vC[kc] = W.u[1];
            pC[kc] = W.p;
            rhoUC[kc] = Q[1];
            rhoVC[kc] = Q[2];
            EC[kc] = Q[3];
        }
    }

    std::vector<double> rhoP(static_cast<std::size_t>(nPts));
    std::vector<double> uP(static_cast<std::size_t>(nPts));
    std::vector<double> vP(static_cast<std::size_t>(nPts));
    std::vector<double> pP(static_cast<std::size_t>(nPts));
    std::vector<double> rhoUP(static_cast<std::size_t>(nPts));
    std::vector<double> rhoVP(static_cast<std::size_t>(nPts));
    std::vector<double> EP(static_cast<std::size_t>(nPts));

    for (int j = 0; j < nyPts; ++j) {
        for (int i = 0; i < nxPts; ++i) {
            double rhoSum = 0.0;
            double uSum = 0.0;
            double vSum = 0.0;
            double pSum = 0.0;
            double rhoUSum = 0.0;
            double rhoVSum = 0.0;
            double ESum = 0.0;
            int cnt = 0;

            for (int dj = -1; dj <= 0; ++dj) {
                for (int di = -1; di <= 0; ++di) {
                    const int ic = i + di;
                    const int jc = j + dj;
                    if (ic >= 0 && ic < nx && jc >= 0 && jc < ny) {
                        const std::size_t kc = static_cast<std::size_t>(idxC(ic, jc));
                        rhoSum += rhoC[kc];
                        uSum += uC[kc];
                        vSum += vC[kc];
                        pSum += pC[kc];
                        rhoUSum += rhoUC[kc];
                        rhoVSum += rhoVC[kc];
                        ESum += EC[kc];
                        ++cnt;
                    }
                }
            }

            const std::size_t kp = static_cast<std::size_t>(idxP(i, j));
            rhoP[kp] = rhoSum / static_cast<double>(cnt);
            uP[kp] = uSum / static_cast<double>(cnt);
            vP[kp] = vSum / static_cast<double>(cnt);
            pP[kp] = pSum / static_cast<double>(cnt);
            rhoUP[kp] = rhoUSum / static_cast<double>(cnt);
            rhoVP[kp] = rhoVSum / static_cast<double>(cnt);
            EP[kp] = ESum / static_cast<double>(cnt);
        }
    }

    out << "POINT_DATA " << nPts << "\n";

    out << "SCALARS rho double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < nyPts; ++j) {
        for (int i = 0; i < nxPts; ++i) {
            out << rhoP[static_cast<std::size_t>(idxP(i, j))] << "\n";
        }
    }

    out << "VECTORS velocity double\n";
    for (int j = 0; j < nyPts; ++j) {
        for (int i = 0; i < nxPts; ++i) {
            const std::size_t kp = static_cast<std::size_t>(idxP(i, j));
            out << uP[kp] << ' ' << vP[kp] << ' ' << 0.0 << "\n";
        }
    }

    out << "SCALARS p double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < nyPts; ++j) {
        for (int i = 0; i < nxPts; ++i) {
            out << pP[static_cast<std::size_t>(idxP(i, j))] << "\n";
        }
    }

    out << "SCALARS rho_u double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < nyPts; ++j) {
        for (int i = 0; i < nxPts; ++i) {
            out << rhoUP[static_cast<std::size_t>(idxP(i, j))] << "\n";
        }
    }

    out << "SCALARS rho_v double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < nyPts; ++j) {
        for (int i = 0; i < nxPts; ++i) {
            out << rhoVP[static_cast<std::size_t>(idxP(i, j))] << "\n";
        }
    }

    out << "SCALARS E double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 0; j < nyPts; ++j) {
        for (int i = 0; i < nxPts; ++i) {
            out << EP[static_cast<std::size_t>(idxP(i, j))] << "\n";
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
// The merged output is converted to POINT_DATA inside writeVTK2D.
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

    // Write a single merged VTK as point data derived from the cell-centered solution.
    writeVTK2D(filename, Uglobal, nxGlobal, nyGlobal, ngOut, x0, x1, y0, y1, gamma);
}
