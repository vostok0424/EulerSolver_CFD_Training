// main.cpp
// -------
// Program entry point.
//
// Responsibilities:
// - Initialise MPI (even for serial runs).
// - Load a case configuration file (.cfg).
// - Create an MPI Cartesian decomposition helper (mpi_parallel::MpiParallel).
// - Dispatch to the 1D or 2D solver based on `dim` in the cfg.
// - Ensure all ranks return the same exit code.

#include "cfg.hpp"
#include "solver1d.hpp"
#include "solver2d.hpp"
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

        // Select solver dimension: dim=1 (1D) or dim=2 (2D).
        const int dim = cfg.getInt("dim", 1);

        // MPI decomposition parameters.
        // Must satisfy: px * py == MPI world size.
        // For pure 1D runs, set py=1 so the domain is split only along x.
        const int px = cfg.getInt("mpi.px", 1);
        const int py = cfg.getInt("mpi.py", 1);

        // Construct MPI topology (throws if px*py != world size).
        // This object also provides neighbor ranks and halo exchange helpers.
        mpi_parallel::MpiParallel mp(px, py, MPI_COMM_WORLD);

        if (mp.isRoot()) {
            std::cout << "[main] MPI world size=" << size
                      << ", mpi.px=" << px << ", mpi.py=" << py << "\n";
        }

        // Dispatch to the selected solver.
        if (dim == 1) {
            if (mp.isRoot()) {
                std::cout << "[main] dim=1, 1D solver\n";
            }
            Solver1D solver(cfg, mp);
            solver.run();
        } else if (dim == 2) {
            if (mp.isRoot()) {
                std::cout << "[main] dim=2, 2D solver\n";
            }
            // Solver2D receives the MPI helper so it can exchange halos and write merged output.
            Solver2D solver(cfg, mp);
            solver.run();
        } else {
            if (mp.isRoot()) {
                std::cerr << "Unsupported dim=" << dim << ", only 1 or 2\n";
            }
            exitCode = 1;
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
