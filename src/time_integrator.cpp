// time_integrator.cpp
// -------------------
// Explicit time integration schemes used by the solvers.
//
// General pattern in this codebase
// -------------------------------
// A time integrator advances the solution vector U by one time step:
//   U^{n+1} = Advance(U^n, dt, rhs)
//
// The integrator does NOT know about meshes, boundary conditions, or MPI.
// Instead, it receives a callback:
//   rhs(U_in, R_out)
// which must fill R_out = RHS(U_in) for the semi-discrete system:
//   dU/dt = RHS(U)
//
// Important convention:
// - The rhs callback is allowed to modify U_in in-place (typically to fill
//   ghost cells / exchange MPI halos) before computing R_out.
//   Therefore, each stage calls rhs(stageState, stageRHS) *after* the stage
//   state has been constructed.
//
// This file implements:
// - Forward Euler
// - RK2 (Heun / SSPRK2 form)
// - SSPRK3 (Shu–Osher)
// - Classical RK4
//
// Adding a new explicit scheme is usually:
//   1) Implement a new TI_XXX<T> with name() and step(...)
//   2) Register it in makeTimeIntegratorT(...)
//   3) Select it in cfg: timeIntegrator = xxx
//
// -----------------------------------------------------------------------------
// Template code: adding a new explicit integrator (copy/paste starter)
// -----------------------------------------------------------------------------
// Example: a generic 2-stage Runge–Kutta method (user-specified coefficients).
// Replace the coefficients with your desired scheme.
//
//   template<typename State>
//   struct TI_MyRK2T final : public TimeIntegratorT<State> {
//       using RHSFunc = typename TimeIntegratorT<State>::RHSFunc;
//       std::string name() const override { return "myrk2"; }
//
//       void step(std::vector<State>& U, double dt, const RHSFunc& rhs) const override {
//           const size_t N = U.size();
//           std::vector<State> U1(U), R(N);
//
//           // Stage 1: k1 = RHS(U^n)
//           rhs(U, R);
//           for (size_t i = 0; i < N; ++i)
//               for (size_t k = 0; k < U[i].size(); ++k)
//                   U1[i][k] = U[i][k] + dt * R[i][k];
//
//           // Stage 2: k2 = RHS(U1)
//           rhs(U1, R);
//
//           // Combine stages (Heun / SSPRK2 form shown)
//           for (size_t i = 0; i < N; ++i)
//               for (size_t k = 0; k < U[i].size(); ++k)
//                   U[i][k] = 0.5 * U[i][k] + 0.5 * (U1[i][k] + dt * R[i][k]);
//       }
//   };
//
// Factory registration:
//   if (name == "myrk2") return std::make_unique<TI_MyRK2T<State>>();
//
// Debugging tip:
// - If you see "Undefined symbol ... TI_MyRK2T", ensure time_integrator.cpp is
//   compiled/linked into your target (Xcode build phases / CMakeLists).
// -----------------------------------------------------------------------------

#include "time_integrator.hpp"
#include "state.hpp"   // for Vec3 / Vec4 explicit instantiation
#include <stdexcept>


// Forward Euler:
//U^{n+1} = U^n + dt * RHS(U^n)
// -----------------------------------------------------------------------------
template<typename State>
std::string TI_EulerT<State>::name() const { return "euler"; }

template<typename State>

void TI_EulerT<State>::step(std::vector<State>& U, double dt, const RHSFunc& rhs) const {
    std::vector<State> R(U.size());
    // RHS callback may update U in-place (e.g., fill ghost cells / MPI halo exchange) before computing R.
    rhs(U, R);
    for (size_t i = 0; i < U.size(); ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U[i][k] += dt * R[i][k];
        }
    }
}


// RK2 (Heun / RK2):
//   U1      = U^n + dt * RHS(U^n)
//   U^{n+1} = 0.5*U^n + 0.5*(U1 + dt*RHS(U1))
// -----------------------------------------------------------------------------

template<typename State>
std::string TI_RK2T<State>::name() const { return "rk2"; }

template<typename State>

void TI_RK2T<State>::step(std::vector<State>& U, double dt, const RHSFunc& rhs) const {
    const size_t N = U.size();
    std::vector<State> U1(U), R(N);

    // RHS callback may update U in-place (e.g., fill ghost cells / MPI halo exchange) before computing R.
    rhs(U, R);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U1[i][k] = U[i][k] + dt * R[i][k];
        }
    }

    // Stage RHS may update U1 ghosts in-place.
    rhs(U1, R);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U[i][k] = 0.5 * U[i][k] + 0.5 * (U1[i][k] + dt * R[i][k]);
        }
    }
}

// SSPRK3 (Shu–Osher / SSPRK3):
//   U1 = U^n + dt*RHS(U^n)
//   U2 = 0.75*U^n + 0.25*(U1 + dt*RHS(U1))
//   U^{n+1} = (1/3)*U^n + (2/3)*(U2 + dt*RHS(U2))
// -----------------------------------------------------------------------------
template<typename State>
std::string TI_SSPRK3T<State>::name() const { return "ssprk3"; }

template<typename State>

void TI_SSPRK3T<State>::step(std::vector<State>& U, double dt, const RHSFunc& rhs) const {
    const size_t N = U.size();
    std::vector<State> U1(U), U2(U), R(N);

    // RHS callback may update U in-place (e.g., fill ghost cells / MPI halo exchange) before computing R.
    rhs(U, R);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U1[i][k] = U[i][k] + dt * R[i][k];
        }
    }

    // Stage RHS may update U1 ghosts in-place.
    rhs(U1, R);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U2[i][k] = 0.75 * U[i][k] + 0.25 * (U1[i][k] + dt * R[i][k]);
        }
    }

    // Stage RHS may update U2 ghosts in-place.
    rhs(U2, R);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U[i][k] = (1.0 / 3.0) * U[i][k] + (2.0 / 3.0) * (U2[i][k] + dt * R[i][k]);
        }
    }
}

// Classical RK4:
//   k1 = RHS(U)
//   k2 = RHS(U + 0.5*dt*k1)
//   k3 = RHS(U + 0.5*dt*k2)
//   k4 = RHS(U + 1.0*dt*k3)
//   U^{n+1} = U + (dt/6)*(k1 + 2k2 + 2k3 + k4)
// -----------------------------------------------------------------------------
template<typename State>
std::string TI_RK4T<State>::name() const { return "rk4"; }

template<typename State>

void TI_RK4T<State>::step(std::vector<State>& U, double dt, const RHSFunc& rhs) const {
    const size_t N = U.size();
    std::vector<State> U1(U), U2(U), U3(U);
    std::vector<State> k1(N), k2(N), k3(N), k4(N);

    // RHS callback may update U in-place (e.g., fill ghost cells / MPI halo exchange) before computing R.
    rhs(U, k1);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U1[i][k] = U[i][k] + 0.5 * dt * k1[i][k];
        }
    }

    // Stage RHS may update U1 ghosts in-place.
    rhs(U1, k2);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U2[i][k] = U[i][k] + 0.5 * dt * k2[i][k];
        }
    }

    // Stage RHS may update U2 ghosts in-place.
    rhs(U2, k3);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U3[i][k] = U[i][k] + dt * k3[i][k];
        }
    }

    // Stage RHS may update U3 ghosts in-place.
    rhs(U3, k4);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < U[i].size(); ++k) {
            U[i][k] += dt * (k1[i][k] + 2.0 * k2[i][k] + 2.0 * k3[i][k] + k4[i][k]) / 6.0;
        }
    }
}


template<typename State>
std::unique_ptr<TimeIntegratorT<State>> makeTimeIntegratorT(const std::string& name) {
    if (name == "euler")  return std::make_unique<TI_EulerT<State>>();
    if (name == "rk2")    return std::make_unique<TI_RK2T<State>>();
    if (name == "ssprk3") return std::make_unique<TI_SSPRK3T<State>>();
    if (name == "rk4")    return std::make_unique<TI_RK4T<State>>();
    throw std::runtime_error("Unknown timeIntegrator: " + name);
}

// ---- Explicit template instantiations ----
// This training project only uses Vec3 (1D conservative) and Vec4 (2D conservative).
// Explicit instantiation keeps compile times reasonable and avoids linker surprises.
template class TI_EulerT<Vec3>;
template class TI_RK2T<Vec3>;
template class TI_SSPRK3T<Vec3>;
template class TI_RK4T<Vec3>;
template std::unique_ptr<TimeIntegratorT<Vec3>> makeTimeIntegratorT<Vec3>(const std::string&);

template class TI_EulerT<Vec4>;
template class TI_RK2T<Vec4>;
template class TI_SSPRK3T<Vec4>;
template class TI_RK4T<Vec4>;
template std::unique_ptr<TimeIntegratorT<Vec4>> makeTimeIntegratorT<Vec4>(const std::string&);
