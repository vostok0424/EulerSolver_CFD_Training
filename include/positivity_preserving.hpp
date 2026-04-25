
#ifndef POSITIVITY_PRESERVING_HPP
#define POSITIVITY_PRESERVING_HPP


#include <string>
#include "state.hpp"

// ================================================================
// Positivity-preserving flux limiting module
//
// This module implements an a posteriori positivity-preserving
// limiter for conservative finite-volume Euler solvers.  It blends
// a high-order numerical flux with a robust first-order low-order
// flux whenever the high-order flux may produce negative density or
// pressure in the local forward-Euler positivity test.
//
// The intended usage is inside the solver after high-order face
// fluxes have been computed and before the residual/RHS is assembled.
// Reconstruction and Riemann solvers remain independent from this
// module; cell_repair remains only an emergency fallback.
// ================================================================

namespace positivity_preserving {

using State = Vec4;

enum class Direction {
   X,
   Y
};

enum class LowOrderFluxType {
   Rusanov,
   GlobalLaxFriedrichs
};

enum class AlphaMode {
   Constant,
   WaveSpeedWeighted
};

struct Options {
   bool enable = false;

   double rhoFloor = 1.0e-13;
   double pFloor = 1.0e-13;

   LowOrderFluxType lowOrderFlux = LowOrderFluxType::Rusanov;
   AlphaMode alphaMode = AlphaMode::Constant;

   int pressureBisectionIters = 20;

   // Safety tolerance used when denominators are extremely small.
   double tiny = 1.0e-300;
};

struct Stats {
   int limitedFaceCount = 0;
   int densityLimitedFaceCount = 0;
   int pressureLimitedFaceCount = 0;
   int failedFaceCount = 0;

   double minThetaDensity = 1.0;
   double minThetaPressure = 1.0;
   double minThetaFinal = 1.0;

   void reset();
};

struct FaceLimitResult {
   State flux;

   double thetaDensity = 1.0;
   double thetaPressure = 1.0;
   double thetaFinal = 1.0;

   bool densityLimited = false;
   bool pressureLimited = false;
   bool failed = false;
};

// ----------------------------------------------------------------
// Parser helpers for configuration strings.
// ----------------------------------------------------------------

LowOrderFluxType parseLowOrderFluxType(const std::string& name);
AlphaMode parseAlphaMode(const std::string& name);

// ----------------------------------------------------------------
// Main face limiter.
//
// leftCell and rightCell are conservative cell-average states on the
// two sides of a face. highOrderFlux is the already-computed high-order
// numerical flux at that face.
//
// The scale parameter is the one-sided positivity-test multiplier:
//
//   1D: scale = 2.0 * dt / dx
//
//   2D with directional partition alphaX + alphaY = 1:
//       scaleX = 2.0 * dt / (alphaX * dx)
//       scaleY = 2.0 * dt / (alphaY * dy)
//
// With the first implementation using alphaX = alphaY = 0.5:
//       scaleX = 4.0 * dt / dx
//       scaleY = 4.0 * dt / dy
// ----------------------------------------------------------------

FaceLimitResult limitFaceFlux(
   const State& leftCell,
   const State& rightCell,
   const State& highOrderFlux,
   double scale,
   Direction direction,
   double gamma,
   const Options& options,
   Stats& stats
);

// ----------------------------------------------------------------
// Low-order flux construction.
// ----------------------------------------------------------------

State computeLocalRusanovFlux(
   const State& leftCell,
   const State& rightCell,
   Direction direction,
   double gamma
);

State computeGlobalLaxFriedrichsFlux(
   const State& leftCell,
   const State& rightCell,
   Direction direction,
   double gamma,
   double globalMaxWaveSpeed
);

// ----------------------------------------------------------------
// Utility functions.
// ----------------------------------------------------------------

double clamp01(double value);

double computeDensityTheta(
   double rhoLowOrder,
   double rhoHighOrder,
   double rhoFloor,
   double tiny
);

double findPressureThetaByBisection(
   const State& lowOrderState,
   const State& candidateState,
   double pFloor,
   double gamma,
   int maxIter
);

bool isAdmissibleForLimiter(
   const State& U,
   double gamma,
   const Options& options
);

} // namespace positivity_preserving

#endif // POSITIVITY_PRESERVING_HPP
