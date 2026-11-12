#pragma once
#include "structures.hpp"
#include <vector>


// Fill the interior background, then apply regions in order using the supplied settings.
//
// state (in/out) - conserved variables in the ghosted cell field
// nx (in) - number of interior cells along x
// ny (in) - number of interior cells along y
// ng (in) - number of ghost-cell layers on each side
// x0 (in) - left physical bound of the represented domain
// x1 (in) - right physical bound of the represented domain
// y0 (in) - bottom physical bound of the represented domain
// y1 (in) - top physical bound of the represented domain
// gamma (in) - ideal-gas heat-capacity ratio
// options (in) - parsed options consumed by this module
void set_fields_2d( std::vector<Conservative_vector> &state, int nx, int ny, int ng, double x0, double x1, double y0,
    double y1, double gamma, const SetFieldsOptions &options );
