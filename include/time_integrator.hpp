#pragma once

// time_integrator.hpp
// -------------------
// Explicit time integration schemes for advancing the semi-discrete system:
//
//   dU/dt = RHS(U)
//
// In a finite-volume solver, RHS(U) is produced by spatial operators:
// reconstruction -> numerical flux -> divergence.
//
// This header provides:
// - A small interface TimeIntegratorT<State>
// - Several explicit schemes: Euler, RK2 (SSPRK2/Heun), SSPRK3, RK4
// - A factory makeTimeIntegratorT<State>(name) to select by cfg string
//
// Design notes:
// - The integrator is templated on State (currently used with Vec4 in the 2D solver).
// - The RHS callback receives (U, R) and is allowed to modify U in-place before
//   computing R. This is intentional: solvers often need to fill ghost cells or
//   exchange MPI halos at each stage.
// - All methods are declared here; implementations live in time_integrator.cpp

#include <functional>
#include <memory>
#include <string>
#include <vector>

// TimeIntegratorT<State>
// ----------------------
// Abstract interface for an explicit time integrator.
//
// step(U, dt, rhs): advance U by one time step of size dt.
//
// The integrator calls rhs(Ustage, R) one or more times per step.
// rhs must fill R with the spatial RHS evaluated at the current stage state.
template<typename State>
class TimeIntegratorT {
public:
    // RHS callback:
    //   rhs(Ustage, R)
    //
    // - Ustage: current stage state (modifiable). The solver may update ghost cells
    //           or perform MPI halo exchange here.
    // - R:      output vector for RHS(Ustage) with the same size/layout as Ustage.
    using RHSFunc = std::function<void(std::vector<State>&, std::vector<State>&)>;

    virtual ~TimeIntegratorT() = default;
    virtual std::string name() const = 0;
    virtual void step(std::vector<State>& U, double dt, const RHSFunc& rhs) = 0;

protected:
    // Ensure a workspace vector has the same size as U.
    static void ensureSize(std::vector<State>& buf, size_t n) {
        if (buf.size() != n) buf.resize(n);
    }
};

// ------------------------------------------------------------
// Forward Euler (1st order)
// ------------------------------------------------------------
// U^{n+1} = U^n + dt * RHS(U^n)
// Simple and cheap, but least stable/accurate.
template<typename State>
class TI_EulerT final : public TimeIntegratorT<State> {
public:
    using RHSFunc = typename TimeIntegratorT<State>::RHSFunc;

    std::string name() const override;
    void step(std::vector<State>& U, double dt, const RHSFunc& rhs) override;

private:
    std::vector<State> R_;
};

// ------------------------------------------------------------
// Explicit RK2 (SSPRK2 / Heun, 2nd order)
// ------------------------------------------------------------
// Two-stage method. Often used as a robust default for hyperbolic problems.
template<typename State>
class TI_RK2T final : public TimeIntegratorT<State> {
public:
    using RHSFunc = typename TimeIntegratorT<State>::RHSFunc;

    std::string name() const override;
    void step(std::vector<State>& U, double dt, const RHSFunc& rhs) override;

private:
    std::vector<State> U1_;
    std::vector<State> R_;
};


// ------------------------------------------------------------
// Explicit SSPRK3 (Shu–Osher, 3rd order strong-stability-preserving)
// ------------------------------------------------------------
// Three-stage SSP scheme; maintains TVD-like stability properties under a CFL limit.
template<typename State>
class TI_SSPRK3T final : public TimeIntegratorT<State> {
public:
    using RHSFunc = typename TimeIntegratorT<State>::RHSFunc;

    std::string name() const override;
    void step(std::vector<State>& U, double dt, const RHSFunc& rhs) override;

private:
    std::vector<State> U1_;
    std::vector<State> U2_;
    std::vector<State> R_;
};

// ------------------------------------------------------------
// Classic Explicit RK4 (4th order)
// ------------------------------------------------------------
// Four-stage classical Runge–Kutta. Accurate for smooth problems, more expensive.
template<typename State>
class TI_RK4T final : public TimeIntegratorT<State> {
public:
    using RHSFunc = typename TimeIntegratorT<State>::RHSFunc;

    std::string name() const override;
    void step(std::vector<State>& U, double dt, const RHSFunc& rhs) override;

private:
    std::vector<State> U0_;
    std::vector<State> U1_;
    std::vector<State> U2_;
    std::vector<State> U3_;
    std::vector<State> k1_;
    std::vector<State> k2_;
    std::vector<State> k3_;
    std::vector<State> k4_;
};

// ------------------------------------------------------------
// Factory
// ------------------------------------------------------------
// Supported names (case-sensitive):
//   euler, rk2, ssprk3, rk4
//
// The factory is explicitly instantiated for Vec4 in time_integrator.cpp.
// This keeps the header lightweight and avoids multiple-definition issues.
template<typename State>
std::unique_ptr<TimeIntegratorT<State>> makeTimeIntegratorT(const std::string& name);
