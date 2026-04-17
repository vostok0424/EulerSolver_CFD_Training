
#include "state.hpp"

#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// Internal helpers used only inside this translation unit.
// These routines support primitive-state validation and positivity repair for
// the current 2D-only solver path.
// -----------------------------------------------------------------------------
namespace {

// Shared primitive-state validation for array-based primitive containers.
// Checks finiteness first, then density/pressure admissibility against the
// configured lower bounds in StateLimits.
template<typename Prim>
StateCheckResult checkPrimitiveImpl(const Prim& W, const StateLimits& limits) {
    StateCheckResult result{};
    result.rho = W.rho;
    result.p = W.p;

    if (!std::isfinite(W.rho) || !std::isfinite(W.p)) {
        result.ok = false;
        result.status = StateStatus::NonFinite;
        return result;
    }

    for (double ui : W.u) {
        if (!std::isfinite(ui)) {
            result.ok = false;
            result.status = StateStatus::NonFinite;
            return result;
        }
    }

    if (W.rho < 0.0) {
        result.ok = false;
        result.status = StateStatus::NegativeDensity;
        return result;
    }
    if (W.rho <= limits.rhoMin) {
        result.ok = false;
        result.status = StateStatus::DensityTooSmall;
        return result;
    }
    if (W.p < 0.0) {
        result.ok = false;
        result.status = StateStatus::NegativePressure;
        return result;
    }
    if (W.p <= limits.pMin) {
        result.ok = false;
        result.status = StateStatus::PressureTooSmall;
        return result;
    }

    return result;
}

// Shared primitive-state repair.
// Applies simple density/pressure floors while leaving velocity components
// unchanged.
template<typename Prim>
void repairPrimitiveImpl(Prim& W, const StateLimits& limits) {
    const double rhoFloor = std::max(limits.eps, limits.rhoMin);
    const double pFloor = std::max(limits.eps, limits.pMin);
    W.rho = std::max(W.rho, rhoFloor);
    W.p = std::max(W.p, pFloor);
}

} // namespace

// Conservative -> primitive conversion for an ideal gas.
// U = (rho, rho*u[, rho*v], rho*E) is converted into
// W = (rho, u[, v], p).
template<int Dim>
typename EosIdealGas<Dim>::Prim
EosIdealGas<Dim>::consToPrim(const Cons& U, double gamma) {
    Prim W{};
    W.rho = U[0];

    double kinetic = 0.0;
    const double E = U[Dim + 1];

    for (int d = 0; d < Dim; ++d) {
        const double rhou = U[1 + d];
        W.u[d] = rhou / W.rho;
        kinetic += 0.5 * W.rho * W.u[d] * W.u[d];
    }

    const double e = E - kinetic;
    W.p = (gamma - 1.0) * e;
    return W;
}

// Primitive -> conservative conversion for an ideal gas.
// The total energy is rebuilt from pressure and kinetic energy.
template<int Dim>
typename EosIdealGas<Dim>::Cons
EosIdealGas<Dim>::primToCons(const Prim& W, double gamma) {
    Cons U{};
    U[0] = W.rho;

    double kinetic = 0.0;
    for (int d = 0; d < Dim; ++d) {
        U[1 + d] = W.rho * W.u[d];
        kinetic += 0.5 * W.rho * W.u[d] * W.u[d];
    }

    const double e = W.p / (gamma - 1.0);
    U[Dim + 1] = e + kinetic;
    return U;
}


// Ideal-gas sound speed.
template<int Dim>
double EosIdealGas<Dim>::soundSpeed(const Prim& W, double gamma) {
    return std::sqrt(std::max(0.0, gamma * W.p / W.rho));
}

// Cached primitive / thermodynamic quantities used in hot paths such as
// numerical flux evaluation and CFL-related scans.
template<>
FlowVars2 EosIdealGas<2>::evalFlowVars(const ConsD<2>& U, double gamma) {
    FlowVars2 W{};
    W.rho = U[0];
    W.u = U[1] / W.rho;
    W.v = U[2] / W.rho;

    const double kinetic = 0.5 * W.rho * (W.u * W.u + W.v * W.v);
    const double eInt = U[3] - kinetic;
    W.p = (gamma - 1.0) * eInt;
    W.a = std::sqrt(std::max(0.0, gamma * W.p / W.rho));
    W.H = (U[3] + W.p) / W.rho;
    return W;
}

// Physical Euler flux in the requested coordinate direction.
// dir = 0 selects the x-flux, dir = 1 selects the y-flux in 2D.
template<int Dim>
typename EosIdealGas<Dim>::Cons
EosIdealGas<Dim>::physFlux(const Cons& U, int dir, double gamma) {
    const Prim W = consToPrim(U, gamma);

    Cons F{};
    F[0] = U[1 + dir];

    for (int d = 0; d < Dim; ++d) {
        if (d == dir) {
            F[1 + d] = U[1 + d] * W.u[dir] + W.p;
        } else {
            F[1 + d] = U[1 + d] * W.u[dir];
        }
    }

    F[Dim + 1] = (U[Dim + 1] + W.p) * W.u[dir];
    return F;
}

// Fast finite check on a 2D conservative state vector.
bool isFiniteState(const Vec4& U) {
    for (double value : U) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

// Recover internal energy from a 2D conservative state.
static inline double internalEnergy2D(const Vec4& U) {
    const double rho = U[0];
    const double invRho = 1.0 / rho;
    const double u = U[1] * invRho;
    const double v = U[2] * invRho;
    const double kinetic = 0.5 * rho * (u * u + v * v);
    return U[3] - kinetic;
}

// Ideal-gas pressure from internal energy density.
static inline double pressureFromInternalEnergy(double eInt, double gamma) {
    return (gamma - 1.0) * eInt;
}

// Early density-only screen used before the more expensive conservative-state
// checks.
static inline StateCheckResult quickRejectDensity(double rho, const StateLimits& limits) {
    StateCheckResult result{};
    result.rho = rho;

    if (!std::isfinite(rho)) {
        result.ok = false;
        result.status = StateStatus::NonFinite;
        return result;
    }
    if (rho < 0.0) {
        result.ok = false;
        result.status = StateStatus::NegativeDensity;
        return result;
    }
    if (rho <= limits.rhoMin) {
        result.ok = false;
        result.status = StateStatus::DensityTooSmall;
        return result;
    }
    return result;
}

// Public 2D primitive-state validation entry point.
StateCheckResult checkPrimitive(const Prim2& W, const StateLimits& limits) {
    return checkPrimitiveImpl(W, limits);
}

// Fast conservative-state admissibility screen.
// This checks finiteness, density, internal energy, and pressure without doing
// the full primitive-state reconstruction/reporting chain.
StateCheckResult quickCheckConservative(const Vec4& U, double gamma, const StateLimits& limits) {
    if (!isFiniteState(U)) {
        StateCheckResult result{};
        result.ok = false;
        result.status = StateStatus::NonFinite;
        return result;
    }

    StateCheckResult result = quickRejectDensity(U[0], limits);
    if (!result.ok) {
        return result;
    }

    result.eInt = internalEnergy2D(U);
    if (!std::isfinite(result.eInt)) {
        result.ok = false;
        result.status = StateStatus::NonFinite;
        return result;
    }
    if (result.eInt < 0.0) {
        result.ok = false;
        result.status = StateStatus::NegativeInternalEnergy;
        return result;
    }

    result.p = pressureFromInternalEnergy(result.eInt, gamma);
    if (!std::isfinite(result.p)) {
        result.ok = false;
        result.status = StateStatus::NonFinite;
        return result;
    }
    if (result.p < 0.0) {
        result.ok = false;
        result.status = StateStatus::NegativePressure;
        return result;
    }
    if (result.p <= limits.pMin) {
        result.ok = false;
        result.status = StateStatus::PressureTooSmall;
        return result;
    }

    return result;
}

// Full conservative-state check.
// Runs the quick screen first, then converts to primitive variables for the
// fuller admissibility check.
StateCheckResult checkConservative(const Vec4& U, double gamma, const StateLimits& limits) {
    StateCheckResult result = quickCheckConservative(U, gamma, limits);
    if (!result.ok) {
        return result;
    }

    const Prim2 W = EosIdealGas<2>::consToPrim(U, gamma);
    const StateCheckResult primResult = checkPrimitive(W, limits);
    if (!primResult.ok) {
        return primResult;
    }

    return result;
}

// Public 2D primitive-state repair entry point.
void repairPrimitive(Prim2& W, const StateLimits& limits) {
    repairPrimitiveImpl(W, limits);
}

// Conservative-state repair via primitive conversion, floor enforcement, and
// back-conversion to conservative form.
bool repairConservative(Vec4& U, double gamma, const StateLimits& limits) {
    Prim2 W = EosIdealGas<2>::consToPrim(U, gamma);
    repairPrimitive(W, limits);
    U = EosIdealGas<2>::primToCons(W, gamma);
    return quickCheckConservative(U, gamma, limits).ok;
}

// Convenience wrapper for cached 2D flow-variable evaluation.
FlowVars2 evalFlowVars(const Vec4& U, double gamma) {
    return EosIdealGas<2>::evalFlowVars(U, gamma);
}

// Physical flux assembled from conservative storage plus cached primitive /
// thermodynamic quantities.
Vec4 physFluxFromFlowVars(const Vec4& U, const FlowVars2& W, int dir) {
    Vec4 F{};
    const double un = (dir == 0) ? W.u : W.v;

    F[0] = U[1 + dir];

    if (dir == 0) {
        F[1] = U[1] * un + W.p;
        F[2] = U[2] * un;
    } else {
        F[1] = U[1] * un;
        F[2] = U[2] * un + W.p;
    }

    F[3] = (U[3] + W.p) * un;
    return F;
}

// Physical flux assembled directly from primitive variables.
Vec4 physFluxFromPrim(const Prim2& W, int dir, double gamma) {
    Vec4 F{};

    const double rho = W.rho;
    const double u   = W.u[0];
    const double v   = W.u[1];
    const double p   = W.p;
    const double E   = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);

    if (dir == 0) {
        F[0] = rho * u;
        F[1] = rho * u * u + p;
        F[2] = rho * v * u;
        F[3] = (E + p) * u;
    } else {
        F[0] = rho * v;
        F[1] = rho * u * v;
        F[2] = rho * v * v + p;
        F[3] = (E + p) * v;
    }

    return F;
}

// Explicit template instantiation for the dimension currently used in this solver.
template struct EosIdealGas<2>;
