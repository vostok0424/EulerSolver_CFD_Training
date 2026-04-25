

#include "positivity_preserving.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <string>

namespace positivity_preserving {


namespace {

// This implementation is specialized to the current 2-D Euler state
// layout: [rho, rho*u, rho*v, E].  Keep this constant synchronized
// with the conservative state dimension used by state.hpp.

constexpr int kNumEq = 4;

// Normalize user-facing configuration strings so that variants such as
// "global_lf", "global-lf", and "Global LF" are parsed consistently.
std::string normalizeName(const std::string& name)
{
    std::string out;
    out.reserve(name.size());

    for (char c : name) {
        if (c == '-' || c == '_' || c == ' ') {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    return out;
}

// Build a zero conservative vector.  This is used as a safe default
// return value when a low-order flux cannot be constructed from invalid
// input states.
State makeZeroState()
{
    State U;
    for (int k = 0; k < kNumEq; ++k) {
        U[k] = 0.0;
    }
    return U;
}

// Convexly blend two conservative states.  The pressure limiter uses this
// routine during bisection to find the largest admissible theta.
State blendState(const State& lowOrderState, const State& candidateState, double theta)
{
    State U;
    const double oneMinusTheta = 1.0 - theta;

    for (int k = 0; k < kNumEq; ++k) {
        U[k] = oneMinusTheta * lowOrderState[k] + theta * candidateState[k];
    }

    return U;
}

// Convexly blend a robust low-order flux and a candidate high-order flux.
// theta = 1 keeps the high-order flux; theta = 0 fully falls back to the
// low-order positivity-preserving flux.
State blendFlux(const State& lowOrderFlux, const State& highOrderFlux, double theta)
{
    State F;
    const double oneMinusTheta = 1.0 - theta;

    for (int k = 0; k < kNumEq; ++k) {
        F[k] = oneMinusTheta * lowOrderFlux[k] + theta * highOrderFlux[k];
    }

    return F;
}

// Construct the one-sided forward-Euler test state adjacent to a face.
// For the left cell, sign = -1; for the right cell, sign = +1.
// The scale factor follows the Hu-Adams-Shu convex-decomposition test.
State makeOneSidedUpdate(const State& cellState, const State& faceFlux, double sign, double scale)
{
    State U;

    for (int k = 0; k < kNumEq; ++k) {
        U[k] = cellState[k] + sign * scale * faceFlux[k];
    }

    return U;
}

// Reject NaN/Inf states before evaluating pressure or wave speed.
bool hasFiniteComponents(const State& U)
{
    for (int k = 0; k < kNumEq; ++k) {
        if (!std::isfinite(U[k])) {
            return false;
        }
    }

    return true;
}

// Pressure evaluation guarded against non-finite states and non-positive
// density.  A negative sentinel value marks an inadmissible state.
double safePressure(const State& U, double gamma)
{
    if (!hasFiniteComponents(U) || U[0] <= 0.0) {
        return -1.0;
    }

    const double p = pressureFromConservative(U, gamma);
    return std::isfinite(p) ? p : -1.0;
}

// Sound speed used by the local Rusanov fallback flux.  Invalid states
// return zero because the caller will reject them before using the flux.
double safeSoundSpeed(const State& U, double gamma)
{
    const double rho = U[0];
    const double p = safePressure(U, gamma);

    if (rho <= 0.0 || p <= 0.0) {
        return 0.0;
    }

    const double c2 = gamma * p / rho;
    return c2 > 0.0 && std::isfinite(c2) ? std::sqrt(c2) : 0.0;
}

} // namespace

// Reset per-RHS or per-time-step limiter statistics.
void Stats::reset()
{
    limitedFaceCount = 0;
    densityLimitedFaceCount = 0;
    pressureLimitedFaceCount = 0;
    failedFaceCount = 0;

    minThetaDensity = 1.0;
    minThetaPressure = 1.0;
    minThetaFinal = 1.0;
}

// Clamp a limiter coefficient into the convex-combination interval.
double clamp01(double value)
{
    if (!std::isfinite(value)) {
        return 0.0;
    }

    if (value < 0.0) {
        return 0.0;
    }

    if (value > 1.0) {
        return 1.0;
    }

    return value;
}

// Parse the requested low-order flux type.  Unknown names intentionally
// fall back to local Rusanov because it is the safest default here.
LowOrderFluxType parseLowOrderFluxType(const std::string& name)
{
    const std::string key = normalizeName(name);

    if (key == "globallaxfriedrichs" || key == "globallf" ||
        key == "laxfriedrichs" || key == "lf") {
        return LowOrderFluxType::GlobalLaxFriedrichs;
    }

    return LowOrderFluxType::Rusanov;
}

// Parse the multidimensional alpha-partition mode used by the solver when
// it computes the one-sided positivity-test scale factors.
AlphaMode parseAlphaMode(const std::string& name)
{
    const std::string key = normalizeName(name);

    if (key == "wavespeedweighted" || key == "weighted" || key == "wavespeed") {
        return AlphaMode::WaveSpeedWeighted;
    }

    return AlphaMode::Constant;
}

// Compute the analytic density limiter coefficient.  Density is linear in
// the conservative variables, so the largest admissible theta is obtained
// directly from the line segment between low-order and high-order states.
double computeDensityTheta(
    double rhoLowOrder,
    double rhoHighOrder,
    double rhoFloor,
    double tiny)
{
    if (rhoHighOrder >= rhoFloor) {
        return 1.0;
    }

    const double denom = rhoLowOrder - rhoHighOrder;
    if (!std::isfinite(denom) || std::fabs(denom) <= tiny) {
        return 0.0;
    }

    return clamp01((rhoLowOrder - rhoFloor) / denom);
}

// Check the minimum admissibility required by the positivity limiter:
// finite conservative components, rho >= rhoFloor, and p >= pFloor.
bool isAdmissibleForLimiter(
    const State& U,
    double gamma,
    const Options& options)
{
    if (!hasFiniteComponents(U)) {
        return false;
    }

    if (U[0] < options.rhoFloor) {
        return false;
    }

    const double p = pressureFromConservative(U, gamma);
    return std::isfinite(p) && p >= options.pFloor;
}

// Compute the pressure limiter coefficient by bisection.  Unlike density,
// pressure is nonlinear in conservative variables, so bisection is more
// robust than trying to maintain a hand-derived closed form.
double findPressureThetaByBisection(
    const State& lowOrderState,
    const State& candidateState,
    double pFloor,
    double gamma,
    int maxIter)
{
    if (safePressure(candidateState, gamma) >= pFloor) {
        return 1.0;
    }

    if (safePressure(lowOrderState, gamma) < pFloor) {
        return 0.0;
    }

    double lo = 0.0;
    double hi = 1.0;

    for (int iter = 0; iter < maxIter; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const State U = blendState(lowOrderState, candidateState, mid);
        const double p = safePressure(U, gamma);

        if (p >= pFloor) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return clamp01(lo);
}

// Construct the local Lax-Friedrichs/Rusanov low-order flux from cell
// averages.  This flux is intentionally more dissipative than the selected
// high-order flux and acts as the positivity-preserving fallback.
State computeLocalRusanovFlux(
    const State& leftCell,
    const State& rightCell,
    Direction direction,
    double gamma)
{
    State flux = makeZeroState();

    if (!hasFiniteComponents(leftCell) || !hasFiniteComponents(rightCell) ||
        leftCell[0] <= 0.0 || rightCell[0] <= 0.0) {
        return flux;
    }

    const double rhoL = leftCell[0];
    const double rhoR = rightCell[0];

    const double uL = leftCell[1] / rhoL;
    const double vL = leftCell[2] / rhoL;
    const double uR = rightCell[1] / rhoR;
    const double vR = rightCell[2] / rhoR;

    const double pL = safePressure(leftCell, gamma);
    const double pR = safePressure(rightCell, gamma);

    if (pL <= 0.0 || pR <= 0.0) {
        return flux;
    }

    const double cL = safeSoundSpeed(leftCell, gamma);
    const double cR = safeSoundSpeed(rightCell, gamma);

    State FL = makeZeroState();
    State FR = makeZeroState();

    double a = 0.0;

    if (direction == Direction::X) {
        FL[0] = leftCell[1];
        FL[1] = leftCell[1] * uL + pL;
        FL[2] = leftCell[1] * vL;
        FL[3] = (leftCell[3] + pL) * uL;

        FR[0] = rightCell[1];
        FR[1] = rightCell[1] * uR + pR;
        FR[2] = rightCell[1] * vR;
        FR[3] = (rightCell[3] + pR) * uR;

        a = std::max(std::fabs(uL) + cL, std::fabs(uR) + cR);
    } else {
        FL[0] = leftCell[2];
        FL[1] = leftCell[2] * uL;
        FL[2] = leftCell[2] * vL + pL;
        FL[3] = (leftCell[3] + pL) * vL;

        FR[0] = rightCell[2];
        FR[1] = rightCell[2] * uR;
        FR[2] = rightCell[2] * vR + pR;
        FR[3] = (rightCell[3] + pR) * vR;

        a = std::max(std::fabs(vL) + cL, std::fabs(vR) + cR);
    }

    for (int k = 0; k < kNumEq; ++k) {
        flux[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * a * (rightCell[k] - leftCell[k]);
    }

    return flux;
}

// Construct a global Lax-Friedrichs flux using a solver-provided maximum
// wave speed.  This is closer to the paper's global LF fallback, but the
// first solver integration can still use local Rusanov for simplicity.
State computeGlobalLaxFriedrichsFlux(
    const State& leftCell,
    const State& rightCell,
    Direction direction,
    double gamma,
    double globalMaxWaveSpeed)
{
    State flux = makeZeroState();

    if (!hasFiniteComponents(leftCell) || !hasFiniteComponents(rightCell) ||
        leftCell[0] <= 0.0 || rightCell[0] <= 0.0) {
        return flux;
    }

    const double rhoL = leftCell[0];
    const double rhoR = rightCell[0];

    const double uL = leftCell[1] / rhoL;
    const double vL = leftCell[2] / rhoL;
    const double uR = rightCell[1] / rhoR;
    const double vR = rightCell[2] / rhoR;

    const double pL = safePressure(leftCell, gamma);
    const double pR = safePressure(rightCell, gamma);

    if (pL <= 0.0 || pR <= 0.0) {
        return flux;
    }

    State FL = makeZeroState();
    State FR = makeZeroState();

    if (direction == Direction::X) {
        FL[0] = leftCell[1];
        FL[1] = leftCell[1] * uL + pL;
        FL[2] = leftCell[1] * vL;
        FL[3] = (leftCell[3] + pL) * uL;

        FR[0] = rightCell[1];
        FR[1] = rightCell[1] * uR + pR;
        FR[2] = rightCell[1] * vR;
        FR[3] = (rightCell[3] + pR) * uR;
    } else {
        FL[0] = leftCell[2];
        FL[1] = leftCell[2] * uL;
        FL[2] = leftCell[2] * vL + pL;
        FL[3] = (leftCell[3] + pL) * vL;

        FR[0] = rightCell[2];
        FR[1] = rightCell[2] * uR;
        FR[2] = rightCell[2] * vR + pR;
        FR[3] = (rightCell[3] + pR) * vR;
    }

    const double a = std::max(0.0, globalMaxWaveSpeed);

    for (int k = 0; k < kNumEq; ++k) {
        flux[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * a * (rightCell[k] - leftCell[k]);
    }

    return flux;
}

// Limit a single face flux.  The method tests whether the high-order flux
// can produce inadmissible one-sided states; if so, it first limits for
// density and then for pressure by blending toward the low-order flux.
FaceLimitResult limitFaceFlux(
    const State& leftCell,
    const State& rightCell,
    const State& highOrderFlux,
    double scale,
    Direction direction,
    double gamma,
    const Options& options,
    Stats& stats)
{
    FaceLimitResult result;
    result.flux = highOrderFlux;

    if (!options.enable) {
        return result;
    }

    // First implementation path: use local Rusanov as the low-order
    // fallback regardless of the configured enum.  The enum is kept so the
    // solver can later switch between local and global LF without changing
    // this public interface.
    const State lowOrderFlux = computeLocalRusanovFlux(leftCell, rightCell, direction, gamma);

    const State leftHighOrderState = makeOneSidedUpdate(leftCell, highOrderFlux, -1.0, scale);
    const State rightHighOrderState = makeOneSidedUpdate(rightCell, highOrderFlux, 1.0, scale);
    const State leftLowOrderState = makeOneSidedUpdate(leftCell, lowOrderFlux, -1.0, scale);
    const State rightLowOrderState = makeOneSidedUpdate(rightCell, lowOrderFlux, 1.0, scale);

    // If even the low-order one-sided states fail, the limiter cannot prove
    // positivity for this face.  Return the low-order flux and let the later
    // diagnostics/cell_repair safety net handle the remaining issue.
    if (!isAdmissibleForLimiter(leftLowOrderState, gamma, options) ||
        !isAdmissibleForLimiter(rightLowOrderState, gamma, options)) {
        result.flux = lowOrderFlux;
        result.thetaDensity = 0.0;
        result.thetaPressure = 0.0;
        result.thetaFinal = 0.0;
        result.densityLimited = true;
        result.pressureLimited = true;
        result.failed = true;

        ++stats.limitedFaceCount;
        ++stats.densityLimitedFaceCount;
        ++stats.pressureLimitedFaceCount;
        ++stats.failedFaceCount;
        stats.minThetaDensity = 0.0;
        stats.minThetaPressure = 0.0;
        stats.minThetaFinal = 0.0;

        return result;
    }

    // Density limiting stage: take the most restrictive theta required by
    // the left and right one-sided states sharing this face.
    double thetaDensity = 1.0;
    thetaDensity = std::min(thetaDensity, computeDensityTheta(
        leftLowOrderState[0], leftHighOrderState[0], options.rhoFloor, options.tiny));
    thetaDensity = std::min(thetaDensity, computeDensityTheta(
        rightLowOrderState[0], rightHighOrderState[0], options.rhoFloor, options.tiny));
    thetaDensity = clamp01(thetaDensity);

    const State densityLimitedFlux = blendFlux(lowOrderFlux, highOrderFlux, thetaDensity);

    const State leftDensityLimitedState = makeOneSidedUpdate(
        leftCell, densityLimitedFlux, -1.0, scale);
    const State rightDensityLimitedState = makeOneSidedUpdate(
        rightCell, densityLimitedFlux, 1.0, scale);

    // Pressure limiting stage: after density has been secured, further blend
    // the density-limited flux toward the low-order flux only if needed.
    double thetaPressure = 1.0;

    if (safePressure(leftDensityLimitedState, gamma) < options.pFloor) {
        thetaPressure = std::min(thetaPressure, findPressureThetaByBisection(
            leftLowOrderState,
            leftDensityLimitedState,
            options.pFloor,
            gamma,
            options.pressureBisectionIters));
    }

    if (safePressure(rightDensityLimitedState, gamma) < options.pFloor) {
        thetaPressure = std::min(thetaPressure, findPressureThetaByBisection(
            rightLowOrderState,
            rightDensityLimitedState,
            options.pFloor,
            gamma,
            options.pressureBisectionIters));
    }

    thetaPressure = clamp01(thetaPressure);

    // Final flux remains conservative because both neighboring cells use the
    // same limited face flux.
    const State finalFlux = blendFlux(lowOrderFlux, densityLimitedFlux, thetaPressure);
    const double thetaFinal = thetaDensity * thetaPressure;

    result.flux = finalFlux;
    result.thetaDensity = thetaDensity;
    result.thetaPressure = thetaPressure;
    result.thetaFinal = thetaFinal;
    result.densityLimited = thetaDensity < 1.0;
    result.pressureLimited = thetaPressure < 1.0;
    result.failed = false;

    if (result.densityLimited) {
        ++stats.densityLimitedFaceCount;
        stats.minThetaDensity = std::min(stats.minThetaDensity, thetaDensity);
    }

    if (result.pressureLimited) {
        ++stats.pressureLimitedFaceCount;
        stats.minThetaPressure = std::min(stats.minThetaPressure, thetaPressure);
    }

    if (result.densityLimited || result.pressureLimited) {
        ++stats.limitedFaceCount;
        stats.minThetaFinal = std::min(stats.minThetaFinal, thetaFinal);
    }

    return result;
}

} // namespace positivity_preserving
