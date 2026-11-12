#pragma once

#include "structures.hpp"
#include <string>


// Read all computational controls directly into the caller's parameter structure.
//
// filename (in) - explicit input case file path
// parameters (out) - typed computational experiment parameters
// error (out) - file, line when available, key and failure reason; empty on success
//
// Returns SUCCESS, PARAMETER_FILE_ERROR, PARAMETER_PARSE_ERROR, MISSING_PARAMETER or INVALID_PARAMETER_VALUE.
// On failure, parameters may be partially assigned and must not be used; no rollback is performed.
ReturnCodes read_parameters( const std::string &filename, CaseParameters &parameters, std::string &error );
