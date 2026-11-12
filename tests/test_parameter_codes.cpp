#include "read_parameters.hpp"
#include <cassert>
#include <fstream>
#include <string>


// Check parameter return codes, diagnostics, direct writes and repeated reads.
//
// argc (in) - number of command-line arguments
// argv (in) - command-line arguments; argv[1] is the scratch filename owned by the test caller
//
// argv[1] is a scratch filename owned by this test's caller.
int main( int argc, char **argv )
{
    assert( argc == 2 );
    const std::string filename = argv[1];
    CaseParameters parameters{};
    std::string error = "previous error";

    {
        std::ofstream file( filename );
        file << "nx=17\nsetFields.nRegions=1\nsetFields.region1.rho=2\n";
    }
    assert( read_parameters( filename, parameters, error ) == SUCCESS );
    assert( error.empty() );
    assert( parameters.grid.nx == 17 );
    assert( parameters.set_fields.regions.size() == 1 );

    {
        std::ofstream file( filename );
        file << "nx=23\nny=invalid\n";
    }
    assert( read_parameters( filename, parameters, error ) == INVALID_PARAMETER_VALUE );
    assert( parameters.grid.nx == 23 ); // Direct assignment survives a later failure.
    assert( error.find( filename + ":2: ny:" ) != std::string::npos );

    {
        std::ofstream file( filename );
        file << "nx=19\n";
    }
    assert( read_parameters( filename, parameters, error ) == SUCCESS );
    assert( error.empty() );
    assert( parameters.set_fields.regions.empty() );

    {
        std::ofstream file( filename );
        file << "bc.left=inlet\n";
    }
    assert( read_parameters( filename, parameters, error ) == MISSING_PARAMETER );
    assert( error.find( "bc.inlet.rho" ) != std::string::npos );

    {
        std::ofstream file( filename );
        file << "missing equals\n";
    }
    assert( read_parameters( filename, parameters, error ) == PARAMETER_PARSE_ERROR );
    assert( read_parameters( filename + "/missing", parameters, error ) == PARAMETER_FILE_ERROR );
}
