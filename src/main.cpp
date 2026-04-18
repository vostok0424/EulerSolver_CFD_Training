// main.cpp
// -------
// Program entry point.
//
// Responsibilities:
// - Initialise MPI (even for serial runs).
// - Load a case configuration file (.cfg).
// - Create an MPI Cartesian decomposition helper (mpi_parallel::MpiParallel).
// - Run the 2D solver only; reject legacy dim=1 configurations.
// - Ensure all ranks return the same exit code.

#include "cfg.hpp"
#include "solver.hpp"
#include "mpi_parallel.hpp"

// MPI is used for domain decomposition and halo exchange.
#include <mpi.h>

// Console logging (root rank only).
#include <iostream>

int main(int argc, char** argv) {
    // Always initialise MPI (even for serial runs).
    // If you run without mpirun, MPI still creates a single-rank world.
    MPI_Init(&argc, &argv);

    // Discover world rank/size early so we can print errors only on root.
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Local exit code (per rank). We will reduce this to a single global exit code.
    int exitCode = 0;

    try {
        // Default case file. You can override it by passing a path as argv[1].
        std::string cfgFile = "cases/case2d_sod.cfg";

        // Command-line override: ./EulerSolver path/to/case.cfg
        if (argc >= 2) cfgFile = argv[1];

        // Load configuration (simple key=value file).
        Cfg cfg;
        cfg.load(cfgFile);

        // Legacy compatibility: read dim from cfg, but only dim=2 is supported now.
        const int dim = cfg.getInt("dim", 2);

        // MPI decomposition parameters.
        // Must satisfy: px * py == MPI world size.
        // This executable now targets 2D runs only.
        const int px = cfg.getInt("mpi.px", 1);
        const int py = cfg.getInt("mpi.py", 1);

        // Construct MPI topology (throws if px*py != world size).
        // This object also provides neighbor ranks and halo exchange helpers.
        mpi_parallel::MpiParallel mp(px, py, MPI_COMM_WORLD);

        if (mp.isRoot()) {
            std::cout << "[main] MPI world size=" << size
                      << ", mpi.px=" << px << ", mpi.py=" << py << "\n";
        }

        // Run the 2D solver only.
        if (dim != 2) {
            if (mp.isRoot()) {
                std::cerr << "Unsupported dim=" << dim
                          << ". This executable now supports 2D only."
                          << " Please use dim=2 and a 2D case file.\n";
            }
            exitCode = 1;
        } else {
            if (mp.isRoot()) {
                std::cout << "[main] dim=2, 2D solver\n";
            }
            // Solver receives the MPI helper so it can exchange halos and write merged output.
            Solver solver(cfg, mp);
            solver.run();
        }
    }
    // Any exception becomes a non-zero exit code. Print only on root to avoid spam.
    catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Fatal: " << e.what() << "\n";
        }
        exitCode = 1;
    }

    // Ensure all ranks exit consistently:
    // take the maximum exitCode across ranks (0 if all succeeded, 1 if any failed).
    int exitCodeGlobal = 0;
    MPI_Allreduce(&exitCode, &exitCodeGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Clean shutdown of MPI.
    MPI_Finalize();
    return exitCodeGlobal;
}
