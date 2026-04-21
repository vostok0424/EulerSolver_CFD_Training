#pragma once

// reconstruction.hpp
// Spatial reconstruction (high-order) for finite-volume methods.
//
// In a cell-centered finite-volume solver, we store conservative variables U in cells.
// To compute a numerical flux at a face, we need left/right states at that face:
//   - UL: value extrapolated from the cell on the left of the face
//   - UR: value extrapolated from the cell on the right of the face
//
// This module provides:
//   - A small set of reconstruction schemes (FirstOrder, MUSCL, WENO5)
//   - Characteristic-space reconstruction only
//   - Conservative-characteristic projection with Roe linearization
//   - A reduced MUSCL limiter set (None, Minmod, VanLeer)
//   - Option parsing from cfg
//
// Ghost cells:
// High-order schemes require ghost cells so their stencils do not read out of bounds.
// requiredGhostCells(s) returns the minimum ng per side:
//   FirstOrder: ng >= 1
//   MUSCL:      ng >= 2
//   WENO5:      ng >= 3
//
// Design note:
// The solver calls reconstructFacesX()/reconstructFacesY() after applyBC() so ghost cells
// contain valid boundary/halo data.

#include "cfg.hpp"
#include "state.hpp"
#include <string>
#include <vector>

namespace recon {

// All reconstruction-related types live in this namespace.

// Scheme: spatial reconstruction order.
//
// FirstOrder:
//   piecewise-constant (robust but diffusive)
// MUSCL:
//   piecewise-linear with slope limiter (2nd order in smooth regions)
// WENO5:
//   5th-order Weighted ENO (Jiang–Shu), non-oscillatory near discontinuities
enum class Scheme {
    FirstOrder,  // piecewise constant
    MUSCL,       // piecewise linear (2nd order)
    WENO5        // 5th-order WENO (Jiang–Shu)
};

// Ghost-cell requirement (per side) for each scheme.
// - FirstOrder: ng >= 1
// - MUSCL:      ng >= 2
// - WENO5:      ng >= 3
inline int requiredGhostCells(Scheme s) {
    switch (s) {
        case Scheme::FirstOrder: return 1;
        case Scheme::MUSCL:      return 2;
        case Scheme::WENO5:      return 3;
    }
    return 1;
}

inline bool isHighOrder(Scheme s) {
    return s == Scheme::MUSCL || s == Scheme::WENO5;
}

// Limiter: slope limiter used by MUSCL.
//
// None disables limiting (can overshoot near discontinuities).
// Supported choices are intentionally kept small:
//   Minmod  : most diffusive but robust
//   VanLeer : less diffusive and smooth in regular regions
enum class Limiter {
    None,
    Minmod,
    VanLeer
};

// Options
// -------
// A single bundle of user-selectable reconstruction settings.
// Populated from cfg via readOptions(cfg).
struct Options {
    Scheme scheme = Scheme::FirstOrder;
    // Limiter for MUSCL (ignored for FirstOrder/WENO5).
    Limiter limiter = Limiter::VanLeer;

    // Admissibility / repair controls.
    // If enabled, reconstruction will call the centralized state-layer repair
    // routine on reconstructed face states.
    bool positivityFix = true;
    // If a reconstructed face state is not admissible, retry with a lower-order scheme.
    bool enableFallback = true;

    // Numerical tolerances.
    // Small epsilon used in WENO weights and positivity-related repairing.
    double eps = 1e-12;

    // Small admissibility floors used by the centralized state-layer checks.
    double rhoMin = 1e-12;
    double pMin = 1e-12;

    // Characteristic reconstruction is fixed to conservative variables
    // with Roe-linearized eigenvectors.

    // Convenience: required ghost cells for the chosen scheme.
    int requiredNg() const { return recon::requiredGhostCells(scheme); }

    // Convert reconstruction-local thresholds into the shared state-layer limits.
    StateLimits stateLimits() const {
        StateLimits limits{};
        limits.eps = eps;
        limits.rhoMin = rhoMin;
        limits.pMin = pMin;
        return limits;
    }
};

// Reconstruction statistics
// -------------------------
// Lightweight counters collected during face reconstruction.
// These counters are intended to be accumulated by the caller per reconstruction pass.
struct ReconstructionStats {
    int fallbackFaceCount = 0;
    int repairedStateCount = 0;
    int failedRepairCount = 0;

    void clear() {
        fallbackFaceCount = 0;
        repairedStateCount = 0;
        failedRepairCount = 0;
    }

    void accumulate(const ReconstructionStats& other) {
        fallbackFaceCount += other.fallbackFaceCount;
        repairedStateCount += other.repairedStateCount;
        failedRepairCount += other.failedRepairCount;
    }
};

// Read reconstruction options from cfg.
//
// Expected keys (strict strings; invalid values should trigger clear errors):
//   reconstruction.scheme: firstOrder | muscl | weno5
//   reconstruction.limiter: none | minmod | vanleer
//   reconstruction.positivityFix: true|false
//   reconstruction.enableFallback: true|false
//   reconstruction.eps: small positive number (e.g. 1e-12)
//   reconstruction.rhoMin: small positive density floor passed into state-layer checks
//   reconstruction.pMin: small positive pressure floor passed into state-layer checks
//
//   WENO5 uses the standard Jiang–Shu nonlinear weights with exponent fixed at p = 2
// Characteristic reconstruction is fixed to conservative variables
// with Roe linearization; no characteristic-mode cfg keys are exposed.
// Limiter support is intentionally restricted to the small set above.
Options readOptions(const Cfg& cfg);

// Reconstruction2D
// ----------------
// Characteristic-space reconstruction only; admissibility/repair are delegated to state.hpp/state.cpp.
// Fixed to conservative-characteristic projection with Roe linearization.
// Build left/right face states in both x- and y-directions.
//
// X-faces (vertical): (nx+1)*ny
//   i = 0..nx, j = 0..ny-1
// Y-faces (horizontal): nx*(ny+1)
//   i = 0..nx-1, j = 0..ny
//
// The input U is cell-centered and includes ghosts.
class Reconstruction2D {
public:
    explicit Reconstruction2D(const Cfg& cfg);

    const Options& options() const { return opt_; }

    void reconstructFacesX(const std::vector<Vec4>& U,
                           int nx, int ny, int ng,
                           double gamma,
                           std::vector<Vec4>& ULx,
                           std::vector<Vec4>& URx,
                           ReconstructionStats* stats = nullptr) const;

    void reconstructFacesY(const std::vector<Vec4>& U,
                           int nx, int ny, int ng,
                           double gamma,
                           std::vector<Vec4>& ULy,
                           std::vector<Vec4>& URy,
                           ReconstructionStats* stats = nullptr) const;

private:
    Options opt_;
};

} // namespace recon
