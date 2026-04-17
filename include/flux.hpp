#pragma once

// flux.hpp
// Numerical fluxes for the finite-volume solver.
//
// In a Godunov-type finite-volume method, the semi-discrete update uses the
// difference of numerical face fluxes across each control volume in 2D.
//
// The role of a numerical flux is to approximate the physical flux across a face
// using the left/right states (UL, UR) reconstructed at that face.
//
// This file defines:
//   - FluxD<Dim>: an abstract interface used by the solver
//   - Several concrete 2D fluxes (Rusanov, HLLC, AUSM, GodunovExact)
//   - makeFluxD<Dim>(name): a factory that selects a flux by string name
//
// Notes:
// - UL/UR are conservative variables (rho, rho*u, rho*E).
// - `dir` selects the spatial direction (0 = x, 1 = y).
// - `gamma` is the ratio of specific heats for an ideal gas.

#include "state.hpp"
#include "riemann_exact.hpp"
#include <memory>
#include <string>

// FluxD<Dim>
// ----------
// Abstract interface for a numerical flux in the 2D solver.
//
// A concrete flux must implement:
//   - name(): identifier used in cfg (e.g. "rusanov", "hllc", "ausm")
//   - numericalFlux(UL, UR, dir, gamma): face flux given left/right conservative states
//
// The solver constructs the flux once via makeFluxD<Dim>(...) and calls it for each face.

template<int Dim>
class FluxD {
public:
    using Cons = ConsD<Dim>;
    virtual ~FluxD() = default;
    virtual std::string name() const = 0;
    virtual Cons numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const = 0;
};

// ------------------------------------------------------------
// Rusanov / Local Lax-Friedrichs (LLF)
// ------------------------------------------------------------
// Robust and simple, but relatively diffusive.
// Uses the maximum characteristic speed to add dissipation.
template<int Dim>
class FluxRusanov final : public FluxD<Dim> {
public:
    using Cons = ConsD<Dim>;
    using Prim = PrimD<Dim>;

    std::string name() const override;
    Cons numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const override;
};

// ------------------------------------------------------------
// HLLC (Harten–Lax–van Leer–Contact)
// ------------------------------------------------------------
// Captures contact and shear waves better than HLL/Rusanov.
// Implemented as an explicit specialization for Dim=2.
template<int Dim>
class FluxHLLC;


template<>
class FluxHLLC<2> final : public FluxD<2> {
public:
    using Cons = ConsD<2>;
    using Prim = PrimD<2>;
    std::string name() const override { return "hllc"; }
    Cons numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const override;
};

// ------------------------------------------------------------
// AUSM (Advection Upstream Splitting Method)
// ------------------------------------------------------------
// Flux splitting based on Mach number; often sharp for shocks.
// Implemented as an explicit specialization for Dim=2.
template<int Dim>
class FluxAUSM;


template<>
class FluxAUSM<2> final : public FluxD<2> {
public:
    using Cons = ConsD<2>;
    using Prim = PrimD<2>;
    std::string name() const override { return "ausm"; }
    Cons numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const override;
};

// ------------------------------------------------------------
// Godunov Exact (reference)
// ------------------------------------------------------------
// Dim=2: directional reduction (apply exact 1D Riemann logic in the requested dir).
template<int Dim>
class FluxGodunovExact;


template<>
class FluxGodunovExact<2> final : public FluxD<2> {
public:
    using Cons = ConsD<2>;
    using Prim = PrimD<2>;
    std::string name() const override { return "godunovExact"; }
    Cons numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const override;
};

// ------------------------------------------------------------
// Factory
// ------------------------------------------------------------
// Select a flux implementation by name (from cfg).
//
// Examples:
//   auto flux = makeFluxD<2>("hllc");
//   auto flux = makeFluxD<2>("ausm");
template<int Dim>
std::unique_ptr<FluxD<Dim>> makeFluxD(const std::string& name);

// Explicit instantiations are provided in src/flux.cpp.
// This avoids multiple-definition issues for the factory template across translation units.
extern template std::unique_ptr<FluxD<2>> makeFluxD<2>(const std::string& name);
