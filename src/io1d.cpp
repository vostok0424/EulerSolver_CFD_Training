#include "io1d.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <type_traits>
#include <vector>

// io1d.cpp
// --------
// Legacy VTK output for 1D solutions.
//
// This file writes a single ASCII legacy VTK rectilinear-grid file (.vtk)
// for post-processing in ParaView or other VTK-capable tools.
//
// The solution is cell-centered and stored as CELL_DATA on a 1D
// RECTILINEAR_GRID. The x-coordinates written to the VTK file are the
// cell-edge coordinates spanning the physical domain [x0, x1].
//
// Output fields:
//   rho   : density
//   u     : velocity
//   p     : pressure
//   rho_u : momentum
//   E     : total energy from the conservative state
//
// MPI:
// - writeVTK1D_GatherMPI gathers each rank's *interior* cell data to rank 0,
//   assembles a global field, then calls writeVTK1D to write a single merged VTK file.

// MPI gather/assembly packs Vec3 into raw doubles. Ensure it is tightly-packed.
static_assert(std::is_trivially_copyable_v<Vec3>, "Vec3 must be trivially copyable for MPI I/O packing");
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be exactly 3 doubles (no padding)");

void writeVTK1D(const std::string& filename,
                const std::vector<Vec3>& U,
                int nx, int ng,
                double x0, double x1,
                double gamma)
{
    if (nx <= 0) throw std::runtime_error("writeVTK1D: nx must be > 0");
    if (ng < 0)  throw std::runtime_error("writeVTK1D: ng must be >= 0");
    if (static_cast<int>(U.size()) < ng + nx) {
        throw std::runtime_error("writeVTK1D: state array is smaller than ng + nx");
    }

    std::ofstream ofs(filename);
    if (!ofs) throw std::runtime_error("writeVTK1D: failed to open output file: " + filename);

    ofs << std::setprecision(15);

    const double dx = (x1 - x0) / static_cast<double>(nx);
    const int nxPts = nx + 1;

    // Legacy VTK header.
    ofs << "# vtk DataFile Version 3.0\n";
    ofs << "1D Euler solution\n";
    ofs << "ASCII\n";
    ofs << "DATASET RECTILINEAR_GRID\n";
    ofs << "DIMENSIONS " << nxPts << " 1 1\n";

    // Grid coordinates are cell-edge coordinates.
    ofs << "X_COORDINATES " << nxPts << " double\n";
    for (int i = 0; i < nxPts; ++i) {
        const double x = x0 + static_cast<double>(i) * dx;
        ofs << x << "\n";
    }

    ofs << "Y_COORDINATES 1 double\n";
    ofs << 0.0 << "\n";
    ofs << "Z_COORDINATES 1 double\n";
    ofs << 0.0 << "\n";

    // Cell-centered solution fields.
    ofs << "CELL_DATA " << nx << "\n";

    ofs << "SCALARS rho double 1\n";
    ofs << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx; ++i) {
        const Vec3& Ui = U[ng + i];
        const auto Wi = EosIdealGas<1>::consToPrim(Ui, gamma);
        ofs << Wi.rho << "\n";
    }

    ofs << "SCALARS u double 1\n";
    ofs << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx; ++i) {
        const Vec3& Ui = U[ng + i];
        const auto Wi = EosIdealGas<1>::consToPrim(Ui, gamma);
        ofs << Wi.u[0] << "\n";
    }

    ofs << "SCALARS p double 1\n";
    ofs << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx; ++i) {
        const Vec3& Ui = U[ng + i];
        const auto Wi = EosIdealGas<1>::consToPrim(Ui, gamma);
        ofs << Wi.p << "\n";
    }

    ofs << "SCALARS rho_u double 1\n";
    ofs << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx; ++i) {
        const Vec3& Ui = U[ng + i];
        ofs << Ui[1] << "\n";
    }

    ofs << "SCALARS E double 1\n";
    ofs << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nx; ++i) {
        const Vec3& Ui = U[ng + i];
        ofs << Ui[2] << "\n";
    }
}

// Gather local interior cell data to rank 0 and write a single merged VTK file.
// Each rank provides ONLY its interior cells (ghost cells are excluded).
void writeVTK1D_GatherMPI(const std::string& filename,
                          const std::vector<Vec3>& Ulocal,
                          int nxLocal, int ng,
                          int iBeg,
                          int nxGlobal,
                          double x0, double x1,
                          double gamma,
                          MPI_Comm comm)
{
    if (nxLocal <= 0) throw std::runtime_error("writeVTK1D_GatherMPI: nxLocal must be > 0");
    if (nxGlobal <= 0) throw std::runtime_error("writeVTK1D_GatherMPI: nxGlobal must be > 0");
    if (ng < 1) throw std::runtime_error("writeVTK1D_GatherMPI: ng must be >= 1");

    // Rank information for collective operations.
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Minimal MPI error checker (throws with context string).
    auto mpiCheck = [](int err, const char* what) {
        if (err != MPI_SUCCESS) {
            throw std::runtime_error(std::string("MPI error in ") + what);
        }
    };

    // Pack this rank's interior conservative variables (no ghosts) into a raw double buffer.
    // This keeps MPI_Gatherv simple and portable.
    constexpr int ncomp = 3;
    const int localCount = nxLocal * ncomp; // in doubles
    std::vector<double> send(static_cast<std::size_t>(localCount));

    for (int i = 0; i < nxLocal; ++i) {
        const Vec3& Q = Ulocal[ng + i];
        const int t = i * ncomp;
        send[t + 0] = Q[0];
        send[t + 1] = Q[1];
        send[t + 2] = Q[2];
    }

    // Send metadata so root knows where this rank's block sits in the global array.
    int meta[2] = { iBeg, nxLocal };
    std::vector<int> allMeta; // size*2 on root
    if (rank == 0) allMeta.resize(static_cast<std::size_t>(size) * 2u);

    mpiCheck(MPI_Gather(meta, 2, MPI_INT,
                        rank == 0 ? allMeta.data() : nullptr, 2, MPI_INT,
                        0, comm),
             "MPI_Gather(meta)");

    // Gather payload sizes, then gather the packed buffers using MPI_Gatherv.
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
    }

    mpiCheck(MPI_Gather(&localCount, 1, MPI_INT,
                        rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
                        0, comm),
             "MPI_Gather(counts)");

    std::vector<double> recv;
    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = total;
            total += counts[r];
        }
        recv.resize(static_cast<std::size_t>(total));
    }

    mpiCheck(MPI_Gatherv(send.data(), localCount, MPI_DOUBLE,
                         rank == 0 ? recv.data() : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displs.data() : nullptr,
                         MPI_DOUBLE,
                         0, comm),
             "MPI_Gatherv(data)");

    // Only rank 0 assembles and writes the merged file.
    if (rank != 0) return;

    // Assemble a global field with a single ghost layer (ngOut=1) to match writeVTK1D's interface.
    // The ghost values themselves are not important for output; writeVTK1D reads only interior.
    const int ngOut = 1;
    const int nxTotG = nxGlobal + 2 * ngOut;
    std::vector<Vec3> Uglobal(static_cast<std::size_t>(nxTotG), Vec3{});

    // Unpack each rank's block into its correct global location.
    for (int r = 0; r < size; ++r) {
        const int ib  = allMeta[2 * r + 0];
        const int nxr = allMeta[2 * r + 1];

        const int base = displs[r];
        const int expected = nxr * ncomp;
        if (counts[r] != expected) {
            throw std::runtime_error("writeVTK1D_GatherMPI: inconsistent counts vs meta for rank " + std::to_string(r));
        }

        for (int i = 0; i < nxr; ++i) {
            const int gi = ib + i;
            if (gi < 0 || gi >= nxGlobal) {
                throw std::runtime_error("writeVTK1D_GatherMPI: rank block out of global bounds");
            }
            const int t = i * ncomp;
            Vec3 Q{};
            Q[0] = recv[base + t + 0];
            Q[1] = recv[base + t + 1];
            Q[2] = recv[base + t + 2];
            Uglobal[ngOut + gi] = Q;
        }
    }

    // Write merged VTK containing cell-centered data.
    writeVTK1D(filename, Uglobal, nxGlobal, ngOut, x0, x1, gamma);
}
