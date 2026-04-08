//
//  state.cpp
//  EulerSolver_CFD_Training
//
//  Created by SHUANG QIU on 2026/3/23.
//

#include "state.hpp"

#include <algorithm>
#include <cmath>

namespace {

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

template<typename Prim>
void clampPrimitiveImpl(Prim& W, const StateLimits& limits) {
    const double rhoFloor = std::max(limits.eps, limits.rhoMin);
    const double pFloor = std::max(limits.eps, limits.pMin);
    W.rho = std::max(W.rho, rhoFloor);
    W.p = std::max(W.p, pFloor);
}

} // namespace

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


template<int Dim>
double EosIdealGas<Dim>::soundSpeed(const Prim& W, double gamma) {
    return std::sqrt(std::max(0.0, gamma * W.p / W.rho));
}

template<>
FlowVars1 EosIdealGas<1>::evalFlowVars(const ConsD<1>& U, double gamma) {
    FlowVars1 W{};
    W.rho = U[0];
    W.u = U[1] / W.rho;

    const double kinetic = 0.5 * W.rho * W.u * W.u;
    const double eInt = U[2] - kinetic;
    W.p = (gamma - 1.0) * eInt;
    W.a = std::sqrt(std::max(0.0, gamma * W.p / W.rho));
    W.H = (U[2] + W.p) / W.rho;
    return W;
}

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

bool isFiniteState(const Vec3& U) {
    for (double value : U) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool isFiniteState(const Vec4& U) {
    for (double value : U) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

StateCheckResult checkPrimitive(const Prim1& W, const StateLimits& limits) {
    return checkPrimitiveImpl(W, limits);
}

StateCheckResult checkPrimitive(const Prim2& W, const StateLimits& limits) {
    return checkPrimitiveImpl(W, limits);
}

StateCheckResult checkConservative(const Vec3& U, double gamma, const StateLimits& limits) {
    StateCheckResult result{};

    if (!isFiniteState(U)) {
        result.ok = false;
        result.status = StateStatus::NonFinite;
        return result;
    }

    result.rho = U[0];
    if (result.rho < 0.0) {
        result.ok = false;
        result.status = StateStatus::NegativeDensity;
        return result;
    }
    if (result.rho <= limits.rhoMin) {
        result.ok = false;
        result.status = StateStatus::DensityTooSmall;
        return result;
    }

    const double invRho = 1.0 / result.rho;
    const double u = U[1] * invRho;
    const double kinetic = 0.5 * result.rho * u * u;
    result.eInt = U[2] - kinetic;

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

    result.p = (gamma - 1.0) * result.eInt;
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

StateCheckResult checkConservative(const Vec4& U, double gamma, const StateLimits& limits) {
    StateCheckResult result{};

    if (!isFiniteState(U)) {
        result.ok = false;
        result.status = StateStatus::NonFinite;
        return result;
    }

    result.rho = U[0];
    if (result.rho < 0.0) {
        result.ok = false;
        result.status = StateStatus::NegativeDensity;
        return result;
    }
    if (result.rho <= limits.rhoMin) {
        result.ok = false;
        result.status = StateStatus::DensityTooSmall;
        return result;
    }

    const double invRho = 1.0 / result.rho;
    const double u = U[1] * invRho;
    const double v = U[2] * invRho;
    const double kinetic = 0.5 * result.rho * (u * u + v * v);
    result.eInt = U[3] - kinetic;

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

    result.p = (gamma - 1.0) * result.eInt;
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

void clampPrimitive(Prim1& W, const StateLimits& limits) {
    clampPrimitiveImpl(W, limits);
}

void clampPrimitive(Prim2& W, const StateLimits& limits) {
    clampPrimitiveImpl(W, limits);
}

bool repairConservative(Vec3& U, double gamma, const StateLimits& limits) {
    Prim1 W = EosIdealGas<1>::consToPrim(U, gamma);
    clampPrimitive(W, limits);
    U = EosIdealGas<1>::primToCons(W, gamma);
    return checkConservative(U, gamma, limits).ok;
}

bool repairConservative(Vec4& U, double gamma, const StateLimits& limits) {
    Prim2 W = EosIdealGas<2>::consToPrim(U, gamma);
    clampPrimitive(W, limits);
    U = EosIdealGas<2>::primToCons(W, gamma);
    return checkConservative(U, gamma, limits).ok;
}

FlowVars1 evalFlowVars(const Vec3& U, double gamma) {
    return EosIdealGas<1>::evalFlowVars(U, gamma);
}

FlowVars2 evalFlowVars(const Vec4& U, double gamma) {
    return EosIdealGas<2>::evalFlowVars(U, gamma);
}

Vec3 physFluxFromFlowVars(const Vec3& U, const FlowVars1& W, int dir) {
    (void)dir;
    Vec3 F{};
    F[0] = U[1];
    F[1] = U[1] * W.u + W.p;
    F[2] = (U[2] + W.p) * W.u;
    return F;
}

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

Vec3 physFluxFromPrim(const Prim1& W, int dir, double gamma) {
    (void)dir;
    Vec3 F{};

    const double rho = W.rho;
    const double u   = W.u[0];
    const double p   = W.p;
    const double E   = p / (gamma - 1.0) + 0.5 * rho * u * u;

    F[0] = rho * u;
    F[1] = rho * u * u + p;
    F[2] = (E + p) * u;
    return F;
}

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

// Explicit template instantiations for the dimensions used in this solver.
template struct EosIdealGas<1>;
template struct EosIdealGas<2>;
