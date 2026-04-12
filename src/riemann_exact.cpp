

#include "riemann_exact.hpp"

// File-local helper routines used by the exact Riemann solver. These are kept
// out of the public namespace because they are implementation details only.
namespace {

// Evaluate the one-sided wave function f(p) and its derivative for a candidate
// star pressure p. The formula switches between shock and rarefaction branches
// according to whether p is above or below the side pressure pK.
void prefunImpl(double p, const Prim1& W, double gamma, double& f, double& df) {
    const double rho = W.rho;
    const double pK  = W.p;
    const double aK  = std::sqrt(std::max(0.0, gamma * pK / rho));

    // Shock branch: use the Rankine-Hugoniot-based pressure function.
    if (p > pK) {
        const double A = 2.0 / ((gamma + 1.0) * rho);
        const double B = (gamma - 1.0) / (gamma + 1.0) * pK;
        const double sqrtTerm = std::sqrt(A / (p + B));
        f  = (p - pK) * sqrtTerm;
        df = sqrtTerm * (1.0 - 0.5 * (p - pK) / (p + B));
    // Rarefaction branch: use the isentropic self-similar pressure function.
    } else {
        const double pratio = p / pK;
        const double expo = (gamma - 1.0) / (2.0 * gamma);
        f  = (2.0 * aK / (gamma - 1.0)) * (std::pow(pratio, expo) - 1.0);
        df = (1.0 / (rho * aK)) * std::pow(pratio, -(gamma + 1.0) / (2.0 * gamma));
    }
}

// Compute a PVRS-based initial guess for p*. This is inexpensive and usually
// provides a good starting point for the Newton iteration.
double guessPressurePVRSImpl(const Prim1& WL, const Prim1& WR, double gamma) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));
    const double pPV = 0.5 * (WL.p + WR.p)
                     - 0.125 * (WR.u[0] - WL.u[0]) * (WL.rho + WR.rho) * (aL + aR);
    return std::max(1e-12, pPV);
}

} // namespace

// Public exact Riemann-solver interface.
namespace exact_riemann {

// Thin public wrapper around the file-local implementation.
void prefun(double p, const Prim1& W, double gamma, double& f, double& df) {
    prefunImpl(p, W, gamma, f, df);
}

// Thin public wrapper returning the PVRS pressure guess.
double guessPressurePVRS(const Prim1& WL, const Prim1& WR, double gamma) {
    return guessPressurePVRSImpl(WL, WR, gamma);
}

// Solve for the star-region pressure p* and velocity u* using a Newton method
// on the standard exact-Riemann pressure equation.
void starPU(const Prim1& WL, const Prim1& WR, double gamma, double& pStar, double& uStar) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));
    // This implementation does not handle vacuum generation.
    if ((WR.u[0] - WL.u[0]) > (2.0/(gamma-1.0))*(aL + aR)) {
        throw std::runtime_error("Exact Riemann: vacuum not handled.");
    }

    double p = guessPressurePVRSImpl(WL, WR, gamma);

    // Newton iteration for p*. The pressure is clamped to a small positive
    // floor after each update for robustness.
    for (int it = 0; it < 40; ++it) {
        double fL, dfL, fR, dfR;
        prefunImpl(p, WL, gamma, fL, dfL);
        prefunImpl(p, WR, gamma, fR, dfR);

        const double g  = fL + fR + (WR.u[0] - WL.u[0]);
        const double dg = dfL + dfR;

        double pNew = p - g / std::max(1e-14, dg);
        pNew = std::max(1e-12, pNew);

        const double rel = std::abs(pNew - p) / (0.5 * (pNew + p) + 1e-14);
        p = pNew;
        if (rel < 1e-10) {
            break;
        }
    }

    double fL, dfL, fR, dfR;
    prefunImpl(p, WL, gamma, fL, dfL);
    prefunImpl(p, WR, gamma, fR, dfR);

    // Once p* is known, recover u* from the left/right wave relations.
    pStar = p;
    uStar = 0.5 * (WL.u[0] + WR.u[0] + fR - fL);
}

// Sample the exact self-similar solution at S = x/t = 0. Depending on the sign
// of u* and on whether each wave is a shock or rarefaction, the sample lies in
// an initial state, a star state, or inside a rarefaction fan.
ExactSample1D sampleAtS0(const Prim1& WL, const Prim1& WR, double gamma, double pStar, double uStar) {
    const double aL = std::sqrt(std::max(0.0, gamma * WL.p / WL.rho));
    const double aR = std::sqrt(std::max(0.0, gamma * WR.p / WR.rho));

    ExactSample1D out{};

    // The interface sample lies on the left side of the contact.
    if (0.0 <= uStar) {
        out.sideTag = -1;
        if (pStar > WL.p) {
            // Left shock case.
            const double qL = std::sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pStar / WL.p - 1.0));
            const double SL = WL.u[0] - aL * qL;
            if (0.0 <= SL) {
                out = {WL.rho, WL.u[0], WL.p, -1};
            } else {
                const double pr = pStar / WL.p;
                const double rhoStar = WL.rho * (pr + (gamma - 1.0) / (gamma + 1.0)) /
                                      ((gamma - 1.0) / (gamma + 1.0) * pr + 1.0);
                out = {rhoStar, uStar, pStar, -1};
            }
        } else {
            // Left rarefaction case.
            const double SHL = WL.u[0] - aL;
            const double aStarL = aL * std::pow(pStar / WL.p, (gamma - 1.0) / (2.0 * gamma));
            const double STL = uStar - aStarL;
            if (0.0 <= SHL) {
                out = {WL.rho, WL.u[0], WL.p, -1};
            } else if (0.0 >= STL) {
                const double rhoStar = WL.rho * std::pow(pStar / WL.p, 1.0 / gamma);
                out = {rhoStar, uStar, pStar, -1};
            } else {
                const double S = 0.0;
                const double u = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * WL.u[0] + S);
                const double a = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * (WL.u[0] - S));
                const double rho = WL.rho * std::pow(a / aL, 2.0 / (gamma - 1.0));
                const double p   = WL.p   * std::pow(a / aL, 2.0 * gamma / (gamma - 1.0));
                out = {rho, u, p, -1};
            }
        }
    // The interface sample lies on the right side of the contact.
    } else {
        out.sideTag = +1;
        if (pStar > WR.p) {
            // Right shock case.
            const double qR = std::sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pStar / WR.p - 1.0));
            const double SR = WR.u[0] + aR * qR;
            if (0.0 >= SR) {
                out = {WR.rho, WR.u[0], WR.p, +1};
            } else {
                const double pr = pStar / WR.p;
                const double rhoStar = WR.rho * (pr + (gamma - 1.0) / (gamma + 1.0)) /
                                      ((gamma - 1.0) / (gamma + 1.0) * pr + 1.0);
                out = {rhoStar, uStar, pStar, +1};
            }
        } else {
            // Right rarefaction case.
            const double SHR = WR.u[0] + aR;
            const double aStarR = aR * std::pow(pStar / WR.p, (gamma - 1.0) / (2.0 * gamma));
            const double STR = uStar + aStarR;
            if (0.0 >= SHR) {
                out = {WR.rho, WR.u[0], WR.p, +1};
            } else if (0.0 <= STR) {
                const double rhoStar = WR.rho * std::pow(pStar / WR.p, 1.0 / gamma);
                out = {rhoStar, uStar, pStar, +1};
            } else {
                const double S = 0.0;
                const double u = (2.0 / (gamma + 1.0)) * (-aR + 0.5 * (gamma - 1.0) * WR.u[0] + S);
                const double a = (2.0 / (gamma + 1.0)) * ( aR - 0.5 * (gamma - 1.0) * (WR.u[0] - S));
                const double rho = WR.rho * std::pow(a / aR, 2.0 / (gamma - 1.0));
                const double p   = WR.p   * std::pow(a / aR, 2.0 * gamma / (gamma - 1.0));
                out = {rho, u, p, +1};
            }
        }
    }

    return out;
}

// Compute the exact Godunov flux by solving the exact Riemann problem,
// sampling the interface state at S=0, converting back to conservative form,
// and then evaluating the physical Euler flux.
Vec3 godunovExactFlux1D(const Vec3& UL, const Vec3& UR, double gamma) {
    const auto WL = EosIdealGas<1>::consToPrim(UL, gamma);
    const auto WR = EosIdealGas<1>::consToPrim(UR, gamma);

    double pStar = 0.0;
    double uStar = 0.0;
    starPU(WL, WR, gamma, pStar, uStar);
    const auto S0 = sampleAtS0(WL, WR, gamma, pStar, uStar);

    Prim1 W0{};
    W0.rho = S0.rho;
    W0.u[0] = S0.u;
    W0.p = S0.p;

    const Vec3 U0 = EosIdealGas<1>::primToCons(W0, gamma);
    return EosIdealGas<1>::physFlux(U0, 0, gamma);
}

} // namespace exact_riemann
