#include "read_parameters.hpp"
#include "solver.hpp"
#include "mpi_parallel.hpp"


#include <mpi.h>


#include <iostream>

// Run the MPI solver with an explicit case path.
//
// argc (in) - number of command-line arguments
// argv (in) - command-line arguments; argv[1] is the explicitly supplied case path
//
// Configuration failures and numerical exceptions abort the complete MPI job.
int main( int argc, char **argv )
{
    // Initialize MPI for both direct and mpirun execution.
    MPI_Init( &argc, &argv );

    int rank = 0;
    int size = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    // Invalid command lines exit collectively before creating the Cartesian communicator.
    if ( argc != 2 )
    {
        if ( rank == 0 )
        {
            std::cerr << "Usage: " << argv[0] << " <case.cfg>\n";
        }

        MPI_Finalize();
        return 1;
    }

    int exit_code = 0;

    try
    {
        const std::string cfg_file = argv[1];

        // Reading writes directly to parameters; no member is used before checking the status.
        CaseParameters parameters{};
        std::string parameter_error;
        const ReturnCodes code = read_parameters( cfg_file, parameters, parameter_error );
        if ( code != SUCCESS )
        {
            std::cerr << "Parameter error [rank " << rank << ", code " << static_cast<int>( code )
                      << "]: " << parameter_error << std::endl;
            MPI_Abort( MPI_COMM_WORLD, 1 );
            return 1;
        }
        const int dim = parameters.dim;
        const int px = parameters.mpi_px;
        const int py = parameters.mpi_py;

        // Build the configured process layout, then pass parsed settings to the solver.
        MpiParallel mp( px, py, MPI_COMM_WORLD );

        if ( mp.is_root() )
        {
            std::cout << "[main] MPI world size=" << size << ", mpi.px=" << px << ", mpi.py=" << py << "\n";
        }

        if ( dim != spatial_dimension )
        {
            if ( mp.is_root() )
            {
                std::cerr << "Unsupported dim=" << dim << ". This executable now supports 2D only."
                          << " Please use dim=2 and a 2D case file.\n";
            }

            exit_code = 1;
        }
        else
        {
            if ( mp.is_root() )
            {
                std::cout << "[main] dim=2, 2D solver\n";
            }

            Solver solver( parameters, mp );

            solver.run();
        }
    }

    // A failed rank may leave peers in communication; report the rank and abort the entire job.
    catch ( const std::exception &e )
    {
        std::cerr << "Fatal [rank " << rank << "]: " << e.what() << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
        return 1;
    }
    catch ( ... )
    {
        std::cerr << "Fatal [rank " << rank << "]: unknown exception" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
        return 1;
    }

    int exit_code_global = 0;
    // Combine normal exit statuses before finalizing MPI.
    MPI_Allreduce( &exit_code, &exit_code_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

    MPI_Finalize();
    return exit_code_global;
}
