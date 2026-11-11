#pragma once
#include "structures.hpp"
#include "cfg.hpp"
#include <vector>
#include <string>


// Fill the interior with a background state, then apply rectangular regions in index order.
// Read setFields.bg.rho/u/v/p and setFields.nRegions.
// Region K uses setFields.regionK.xMin/xMax/yMin/yMax and rho/u/v/p.
// A positive shockMach requests the post-shock state and must exceed one; shockDir selects its direction.
void set_fields_2d( std::vector<Conservative_vector> &state, int nx, int ny, int ng, double x0, double x1, double y0,
    double y1, double gamma, const Cfg &cfg );
