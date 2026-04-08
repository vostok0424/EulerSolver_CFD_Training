

#include "flux.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// flux.cpp
// --------
// Implementations of numerical fluxes used by the finite-volume solvers.
//
// General interface used throughout this module:
//   numericalFlux(UL, UR, dir, gamma) -> F
// where:
//   - UL/UR are conservative states on the left/right side of a face
//   - dir selects the face-normal direction (0=x, 1=y)
//   - gamma is the ideal-gas ratio of specific heats
//   - F is the conservative flux vector in direction `dir`
//
// ------------------------------------------------------------
// How to add ("plant") a new numerical flux in this codebase
// ------------------------------------------------------------
// All fluxes share the same calling convention, so adding a new one is mostly
// a matter of implementing one function and registering a name.
//
// Step 1) Implement a new class (recommended location: flux.hpp + flux.cpp)
//
//   template<int Dim>
//   struct FluxMyNewScheme final : public FluxD<Dim> {
//       std::string name() const override { return "myNewScheme"; }
//       Cons numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const override {
//           // 0) Validate direction
//           //    if (Dim==2) dir must be 0 or 1
//
//           // 1) Convert to primitive if needed
//           //    WL = consToPrim(UL), WR = consToPrim(UR)
//
//           // 2) Compute physical fluxes
//           //    FL = physFlux(UL, dir), FR = physFlux(UR, dir)
//
//           // 3) Compute wave-speed estimates / split quantities / star states
//           //    (this is where the scheme differs: LLF, HLLC, AUSM, Roe, etc.)
//
//           // 4) Assemble and return F (size Dim+2)
//           //    (For 2D: treat u[dir] as normal velocity, carry tangential momentum consistently.)
//       }
//   };
//
// Step 2) Register it in the factory at the bottom of this file:
//   if (name == "myNewScheme") return std::make_unique<FluxMyNewScheme<Dim>>();
//
// Step 3) Select it from cfg:
//   flux = myNewScheme
//
// Template code (copy/paste starter)
// ---------------------------------
// The snippet below shows the typical structure used in this project.
// Replace the parts marked "YOUR LOGIC" with your solver-specific details.
//
//   template<int Dim>
//   struct FluxMyNewScheme final : public FluxD<Dim> {
//       using Cons = ConsD<Dim>;
//       std::string name() const override { return "myNewScheme"; }
//
//       Cons numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const override {
//           // (0) Direction validation
//           if constexpr (Dim == 1) {
//               (void)dir; // only x-direction
//           } else {
//               if (dir != 0 && dir != 1)
//                   throw std::runtime_error("myNewScheme flux: dir must be 0 or 1");
//           }
//
//           // (1) Convert to primitive (for wave speeds / splitting)
//           const auto WL = EosIdealGas<Dim>::consToPrim(UL, gamma);
//           const auto WR = EosIdealGas<Dim>::consToPrim(UR, gamma);
//
//           // Normal velocities for this face
//           const double uLn = WL.u[dir];
//           const double uRn = WR.u[dir];
//           const double aL  = EosIdealGas<Dim>::soundSpeed(WL, gamma);
//           const double aR  = EosIdealGas<Dim>::soundSpeed(WR, gamma);
//
//           // (2) Physical fluxes
//           const Cons FL = EosIdealGas<Dim>::physFlux(UL, dir, gamma);
//           const Cons FR = EosIdealGas<Dim>::physFlux(UR, dir, gamma);
//
//           // (3) YOUR LOGIC: estimate wave speeds / build star states / split parts
//           // Example (LLF/Rusanov template):
//           const double smax = std::max(std::abs(uLn) + aL, std::abs(uRn) + aR);
//
//           // (4) Assemble final numerical flux
//           Cons F{};
//           for (int k = 0; k < Dim + 2; ++k) {
//               F[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * smax * (UR[k] - UL[k]);
//           }
//           return F;
//       }
//   };
//
// Notes:
// - If your scheme is HLL/HLLC/Roe-like, step (3) typically produces SL/SR/SM and/or
//   star states U*_L/U*_R, then selects the correct flux by wave-speed sign.
// - For AUSM-like schemes, step (3) computes (mDot, pInt) and then assembles momentum
//   and energy fluxes using an upwinded side.
//
// Debugging tip:
// - If you get "Undefined symbol ... FluxMyNewScheme", it usually means flux.cpp
//   containing the implementation is not compiled/linked into the target.
//
// Most Godunov-type fluxes follow the same high-level pattern:
//
//   1) Convert UL/UR -> primitive states WL/WR (rho, u[,v], p)
//      (needed for wave speeds and/or split forms).
//
//   2) Estimate characteristic signal speeds along the face normal.
//      Typical choices:
//        - Rusanov/LLF: smax = max(|u_n| + a)
//        - HLL/HLLC:    SL and SR bounds for the Riemann fan
//        - Exact:       solve for star region (p*, u*)
//        - AUSM:        split convective and pressure parts via Mach number
//
//   3) Build the flux using one of these common templates:
//
//      (A) Local Lax-Friedrichs / Rusanov:
//          F = 0.5*(F(UL)+F(UR)) - 0.5*smax*(UR-UL)
//
//      (B) HLL-family (piecewise constant Riemann fan):
//          if (SL >= 0)  return F(UL)
//          if (SR <= 0)  return F(UR)
//          else          return HLL/HLLC middle-state flux
//
//      (C) Exact Godunov:
//          solve exact Riemann -> sample at S=x/t=0 -> compute F(W0)
//
//      (D) Flux-splitting (AUSM):
//          mDot  = convective mass flux from split Mach polynomials
//          pInt  = interface pressure from split pressure polynomials
//          then assemble momentum/energy fluxes using upwinded state
//
// In 2D, all of the above are applied in the requested direction `dir`:
//   - un = u[dir] is the normal velocity component
//   - tangential component(s) are carried through consistently
//

// --------------------
// FluxRusanov (Rusanov / Local Lax-Friedrichs)
// --------------------
// General LLF template used here:
//   smax = max(|u_n| + a) over left/right
//   F    = 0.5*(F_L + F_R) - 0.5*smax*(U_R - U_L)
//
// Implementation pattern:
//   - convert UL/UR -> WL/WR to compute u_n and sound speed a
//   - compute physical fluxes F(UL), F(UR)
//   - apply the LLF formula component-wise
template<int Dim>
std::string FluxRusanov<Dim>::name() const { return "rusanov"; }

template<int Dim>
typename FluxRusanov<Dim>::Cons
FluxRusanov<Dim>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);
    double unL, unR;
    if constexpr (Dim == 1) {
        unL = WL.u;
        unR = WR.u;
    } else {
        unL = (dir == 0) ? WL.u : WL.v;
        unR = (dir == 0) ? WR.u : WR.v;
    }
    const double smax = std::max(std::abs(unL) + WL.a, std::abs(unR) + WR.a);
    const Cons FL = physFluxFromFlowVars(UL, WL, dir);
    const Cons FR = physFluxFromFlowVars(UR, WR, dir);

    Cons F{};
    for (int k = 0; k < Dim + 2; ++k)
        F[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * smax * (UR[k] - UL[k]);
    return F;
}

// Explicit instantiations for Dim=1,2
template class FluxRusanov<1>;
template class FluxRusanov<2>;

// HLLC (Harten–Lax–van Leer–Contact) in this module
// -----------------------------------------------
// HLLC represents the Riemann fan by three waves:
//   SL  |  SM  |  SR
// and two star states U*_L and U*_R separated by the contact wave SM.
//
// The common implementation form used here:
//   - Compute primitive states WL/WR and sound speeds aL/aR
//   - Estimate SL and SR (here: min/max of u_n ± a)
//   - Compute physical fluxes FL=F(UL), FR=F(UR)
//   - Upwind selection:
//       if SL >= 0: return FL
//       if SR <= 0: return FR
//       else:
//         compute contact speed SM (from Rankine–Hugoniot across the two waves)
//         construct star states U*_L and U*_R
//         return FL + SL*(U*_L-UL)   if SM >= 0
//         return FR + SR*(U*_R-UR)   otherwise
//
// This structure (speed estimates + star-state construction + piecewise selection)
// is the typical "HLL-family" programming pattern for approximate Riemann solvers.
// --------------------
// FluxHLLC<1>
// --------------------
ConsD<1> FluxHLLC<1>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    (void)dir;
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    const double aL = WL.a;
    const double aR = WR.a;
    const double SL = std::min(WL.u - aL, WR.u - aR);
    const double SR = std::max(WL.u + aL, WR.u + aR);

    const Cons FL = physFluxFromFlowVars(UL, WL, 0);
    const Cons FR = physFluxFromFlowVars(UR, WR, 0);

    // Wave-speed selection (three regions):
    //   SL>0 -> left flux, SR<0 -> right flux, otherwise compute star-region flux.
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double rhoL = WL.rho, rhoR = WR.rho;
    const double uL = WL.u, uR = WR.u;
    const double pL = WL.p,    pR = WR.p;

    const double num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR);
    const double den = rhoL * (SL - uL) - rhoR * (SR - uR);
    const double SM  = (std::abs(den) < 1e-14) ? 0.0 : num / den;

    auto star = [&](const Cons& U, const FlowVars1& W, double S, double Sstar) -> Cons {
        const double rho = W.rho;
        const double un  = W.u;
        const double p   = W.p;
        const double E   = U[2];

        const double rho_star = rho * (S - un) / (S - Sstar);
        const double p_star   = p + rho * (S - un) * (Sstar - un);
        const double E_star   = ((S - un) * E - p * un + p_star * Sstar) / (S - Sstar);

        Cons Us{};
        Us[0] = rho_star;
        Us[1] = rho_star * Sstar;
        Us[2] = E_star;
        return Us;
    };

    const Cons ULs = star(UL, WL, SL, SM);
    const Cons URs = star(UR, WR, SR, SM);

    Cons F{};
    if (SM >= 0.0) {
        for (int k = 0; k < 3; ++k) F[k] = FL[k] + SL * (ULs[k] - UL[k]);
    } else {
        for (int k = 0; k < 3; ++k) F[k] = FR[k] + SR * (URs[k] - UR[k]);
    }
    return F;
}

// 2D HLLC notes
// ------------
// The same HLLC template is applied in direction `dir`.
// - un = u[dir] is the normal velocity component used for wave speeds.
// - Tangential velocity components are preserved across the contact wave.
//   In the star state construction below:
//     - the normal momentum uses rho_star * SM
//     - the tangential momentum uses rho_star * u_t (from the corresponding side)
// --------------------
// FluxHLLC<2>
// --------------------
ConsD<2> FluxHLLC<2>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    const double aL = WL.a;
    const double aR = WR.a;

    const double unL = (dir == 0) ? WL.u : WL.v;
    const double unR = (dir == 0) ? WR.u : WR.v;

    const double SL = std::min(unL - aL, unR - aR);
    const double SR = std::max(unL + aL, unR + aR);

    const Cons FL = physFluxFromFlowVars(UL, WL, dir);
    const Cons FR = physFluxFromFlowVars(UR, WR, dir);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double rhoL = WL.rho, rhoR = WR.rho;
    const double pL   = WL.p,   pR   = WR.p;

    const double num = pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR);
    const double den = rhoL * (SL - unL) - rhoR * (SR - unR);
    const double SM  = (std::abs(den) < 1e-14) ? 0.0 : num / den;

    auto star = [&](const Cons& U, const FlowVars2& W, double S, double Sstar) -> Cons {
        const double rho = W.rho;
        const double un  = (dir == 0) ? W.u : W.v;
        const double ut  = (dir == 0) ? W.v : W.u;
        const double p   = W.p;
        const double E   = U[3];

        const double rho_star = rho * (S - un) / (S - Sstar);
        const double p_star   = p + rho * (S - un) * (Sstar - un);
        const double E_star   = ((S - un) * E - p * un + p_star * Sstar) / (S - Sstar);

        Cons Us{};
        Us[0] = rho_star;
        if (dir == 0) {
            Us[1] = rho_star * Sstar;
            Us[2] = rho_star * ut;
        } else {
            Us[1] = rho_star * ut;
            Us[2] = rho_star * Sstar;
        }
        Us[3] = E_star;
        return Us;
    };

    const Cons ULs = star(UL, WL, SL, SM);
    const Cons URs = star(UR, WR, SR, SM);

    Cons F{};
    if (SM >= 0.0) {
        for (int k = 0; k < 4; ++k) F[k] = FL[k] + SL * (ULs[k] - UL[k]);
    } else {
        for (int k = 0; k < 4; ++k) F[k] = FR[k] + SR * (URs[k] - UR[k]);
    }
    return F;
}

// --------------------
// AUSM helpers
// --------------------
// AUSM is a flux-splitting family. It splits the interface flux into:
//   - a convective (mass) part based on split Mach number polynomials
//   - a pressure part based on split pressure polynomials
//
// The helper functions below implement common polynomial splits:
//   M^+(M), M^-(M)   and   P^+(M), P^-(M)
// with smooth transitions for |M|<1.
namespace ausm_detail {
    inline double Mplus(double M) {
        const double aM = std::abs(M);
        if (aM >= 1.0) return 0.5 * (M + aM);
        return 0.25 * (M + 1.0) * (M + 1.0);
    }
    inline double Mminus(double M) {
        const double aM = std::abs(M);
        if (aM >= 1.0) return 0.5 * (M - aM);
        return -0.25 * (M - 1.0) * (M - 1.0);
    }
    inline double Pplus(double M) {
        const double aM = std::abs(M);
        if (aM >= 1.0) return 0.5 * (1.0 + (M >= 0.0 ? 1.0 : -1.0));
        return 0.25 * (M + 1.0) * (M + 1.0) * (2.0 - M);
    }
    inline double Pminus(double M) {
        const double aM = std::abs(M);
        if (aM >= 1.0) return 0.5 * (1.0 - (M >= 0.0 ? 1.0 : -1.0));
        return 0.25 * (M - 1.0) * (M - 1.0) * (2.0 + M);
    }
}

// AUSM implementation form in this module
// --------------------------------------
// Steps:
//   1) Convert UL/UR -> WL/WR and compute sound speeds aL/aR.
//   2) Choose a reference sound speed a (here: average with a small floor).
//   3) Compute Mach numbers ML = uL/a, MR = uR/a.
//   4) Compute mass flux and interface pressure:
//        mDot = a * ( M^+(ML)*rhoL + M^-(MR)*rhoR )
//        pInt =      P^+(ML)*pL   + P^-(MR)*pR
//   5) Upwind the transported quantities using the sign of mDot.
//      Assemble conservative flux components.
// --------------------
// FluxAUSM<1>
// --------------------
ConsD<1> FluxAUSM<1>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    (void)dir;
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    const double aL = WL.a;
    const double aR = WR.a;
    const double a  = std::max(1e-14, 0.5 * (aL + aR));

    const double ML = WL.u / a;
    const double MR = WR.u / a;

    const double mDot = a * (ausm_detail::Mplus(ML) * WL.rho + ausm_detail::Mminus(MR) * WR.rho);
    const double pInt = ausm_detail::Pplus(ML) * WL.p + ausm_detail::Pminus(MR) * WR.p;

    const bool upL = (mDot >= 0.0);
    const double uUp = upL ? WL.u : WR.u;
    const double HUp = upL ? WL.H : WR.H;

    Cons F{};
    F[0] = mDot;
    F[1] = mDot * uUp + pInt;
    F[2] = mDot * HUp;
    return F;
}

// --------------------
// FluxAUSM<2>
// --------------------
ConsD<2> FluxAUSM<2>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    if (dir != 0 && dir != 1)
        throw std::runtime_error("AUSM flux: dir must be 0 or 1");

    // Directional form: use normal velocity un = u[dir] for Mach splitting.
    // Momentum flux adds the interface pressure only in the normal direction.

    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    const double aL = WL.a;
    const double aR = WR.a;
    const double a  = std::max(1e-14, 0.5 * (aL + aR));

    const double unL = (dir == 0) ? WL.u : WL.v;
    const double unR = (dir == 0) ? WR.u : WR.v;
    const double ML  = unL / a;
    const double MR  = unR / a;

    const double mDot = a * (ausm_detail::Mplus(ML) * WL.rho + ausm_detail::Mminus(MR) * WR.rho);
    const double pInt = ausm_detail::Pplus(ML) * WL.p + ausm_detail::Pminus(MR) * WR.p;

    const bool upL = (mDot >= 0.0);
    const double uUp0 = upL ? WL.u : WR.u;
    const double uUp1 = upL ? WL.v : WR.v;
    const double HUp  = upL ? WL.H : WR.H;

    Cons F{};
    F[0] = mDot;

    if (dir == 0) {
        F[1] = mDot * uUp0 + pInt;
        F[2] = mDot * uUp1;
    } else {
        F[1] = mDot * uUp0;
        F[2] = mDot * uUp1 + pInt;
    }

    F[3] = mDot * HUp;
    return F;
}

// Exact Godunov flux in this module
// --------------------------------
// Programming pattern:
//   - Convert UL/UR to primitive.
//   - Solve the exact Riemann problem for star state (p*, u*).
//   - Sample the self-similar solution at S=x/t=0 to obtain W0.
//   - Convert W0 back to conservative and return the physical flux F(W0).
//
// In 1D we call a dedicated helper: exact_riemann::godunovExactFlux1D.
// --------------------
// FluxGodunovExact<1>
// --------------------
Vec3 FluxGodunovExact<1>::numericalFlux(const Vec3& UL, const Vec3& UR, int dir, double gamma) const {
    (void)dir;
    return exact_riemann::godunovExactFlux1D(UL, UR, gamma);
}

// 2D exact flux here uses a 1D directional reduction:
// - Solve a 1D exact Riemann problem in the normal direction `dir` using (rho, un, p).
// - For the tangential velocity component, upwind from the side that determines the sample.
// - Build a 2D primitive state W0 and evaluate the physical flux in direction `dir`.
// --------------------
// FluxGodunovExact<2>
// --------------------
ConsD<2> FluxGodunovExact<2>::numericalFlux(const Cons& UL, const Cons& UR, int dir, double gamma) const {
    const auto WL = evalFlowVars(UL, gamma);
    const auto WR = evalFlowVars(UR, gamma);

    Prim1 WL1{}, WR1{};
    WL1.rho = WL.rho;
    WL1.u[0] = (dir == 0) ? WL.u : WL.v;
    WL1.p = WL.p;

    WR1.rho = WR.rho;
    WR1.u[0] = (dir == 0) ? WR.u : WR.v;
    WR1.p = WR.p;

    double pStar, uStar;
    exact_riemann::starPU(WL1, WR1, gamma, pStar, uStar);
    const auto S0 = exact_riemann::sampleAtS0(WL1, WR1, gamma, pStar, uStar);

    Prim2 W0{};
    W0.rho = S0.rho;
    W0.p = S0.p;
    W0.u[dir] = S0.u;
    const int tan = 1 - dir;
    W0.u[tan] = (S0.sideTag < 0)
        ? ((tan == 0) ? WL.u : WL.v)
        : ((tan == 0) ? WR.u : WR.v);

    const Cons U0 = EosIdealGas<2>::primToCons(W0, gamma);
    return EosIdealGas<2>::physFlux(U0, dir, gamma);
}

// --------------------
// Factory
// --------------------
// makeFluxD<Dim>(name) is the single place where new fluxes are registered.
//
// Add a new flux in 3 steps:
//   1) Implement numericalFlux(UL,UR,dir,gamma) for your scheme (in flux.cpp).
//      - Keep UL/UR as conservative inputs.
//      - For 2D: treat u[dir] as the normal component, and keep tangential momentum consistent.
//   2) Register the string name -> object mapping below.
//   3) Enable it from cfg:
//        flux = <name>
template<int Dim>
std::unique_ptr<FluxD<Dim>> makeFluxD(const std::string& name) {
    if (name == "rusanov")      return std::make_unique<FluxRusanov<Dim>>();
    if (name == "hllc")         return std::make_unique<FluxHLLC<Dim>>();
    if (name == "ausm")         return std::make_unique<FluxAUSM<Dim>>();
    if (name == "godunovExact") return std::make_unique<FluxGodunovExact<Dim>>();
    throw std::runtime_error("Unknown flux: " + name);
}

template std::unique_ptr<FluxD<1>> makeFluxD<1>(const std::string& name);
template std::unique_ptr<FluxD<2>> makeFluxD<2>(const std::string& name);
