# EulerSolver_CFD_Training

A modular C++ finite-volume solver for the 2D compressible Euler equations on structured Cartesian grids, with MPI domain decomposition, configurable numerical methods, and case-driven initialization.

## Overview

This repository is a training-oriented and research-oriented CFD codebase focused on the 2D Euler equations.

The current executable is a 2D-only solver. It advances cell-centered conservative variables on a uniform Cartesian mesh using explicit time integration, face reconstruction, configurable numerical fluxes, and MPI halo exchange on a 2D Cartesian process layout.

The project is intended for:
- solver training and code refactoring practice
- numerical-method testing on structured grids
- incremental development toward a more capable research solver

At its current stage, the code should be viewed as a modular prototype rather than a production CFD package.

## Current Code Status

The current codebase supports the following main capabilities.

- 2D compressible Euler equations only
- cell-centered finite-volume discretization on structured Cartesian grids
- MPI-based 2D domain decomposition and halo exchange
- explicit time integration
- characteristic-based reconstruction only
- configurable numerical fluxes
- configurable boundary conditions on all four sides
- setFields-style background-plus-region initialization
- merged single-file VTK output for serial and MPI runs
- selectable VTK `POINT_DATA` or `CELL_DATA` output

The legacy 1D pathway has been removed from the active build and runtime path.

## Governing Model and Data Layout

The solver advances the conservative state

- rho
- rho*u
- rho*v
- E

for the 2D ideal-gas Euler equations.

The mesh is uniform over a rectangular domain defined by:

- nx, ny
- x0, x1
- y0, y1

Each MPI rank stores its local interior block plus ghost cells.

## Numerical Methods

### Flux functions

The current code supports the following flux names in configuration files.

- rusanov
- hllc
- ausm
- godunov

Notes:
- `godunov` is the current exact-directional Godunov option.
- Older names such as `godunovExact` are no longer the active configuration name.

### Time integrators

The current code supports the following explicit time integrators.

- euler
- rk2
- ssprk3
- rk4

### Reconstruction

The current reconstruction module supports the following schemes.

- firstOrder
- muscl
- weno5

The current limiter options are:

- none
- minmod
- vanLeer

Important notes:
- normal MUSCL/WENO reconstruction is characteristic-based
- characteristic eigenvectors are evaluated at the arithmetic mean of the adjacent conservative states
- `reconstruction.variables` is kept only as a compatibility key
- if `reconstruction.variables` is provided, it should be `characteristic`
- older limiter names such as `mc` and `superbee` are not supported by the current implementation

`reconstruction.eps` controls the small regularization parameter.
Reconstruction uses the selected scheme directly, without positivity repair or automatic first-order fallback.
The solver does not apply a positivity-preserving flux limiter.

For `weno5`, normal reconstruction produces transverse face averages. A second,
componentwise WENO-JS reconstruction converts five such averages to point states
at four transverse Gauss nodes. Both directions share the same three quadratic
substencils, Jiang-Shu smoothness indicators and nonlinear-weight calculation.
The transverse candidate values and optimal linear weights depend on the Gauss
node. Four-point Gauss quadrature keeps these linear weights positive, avoiding
negative-weight splitting at the midpoint of three-point quadrature.
The solver evaluates the selected Riemann flux at each node and averages the
fluxes with weights approximately `0.17392742, 0.32607258, 0.32607258, 0.17392742`.
`reconstruction.eps` regularizes both normal and transverse WENO-JS weights.
First-order and MUSCL retain their original single-flux paths. WENO5 still needs
three ghost layers; its tensor stencil also needs valid diagonal ghost cells.
MPI exchange therefore completes x halos before sending full-width y strips.

This is a fifth-order spatial design for smooth two-dimensional solutions on
uniform Cartesian meshes. WENO-JS weights can reduce accuracy near smooth
critical points, and discontinuities reduce the local order. Physical boundary
closures remain extrapolation/reflection/inlet/outlet treatments, without a
fifth-order boundary accuracy guarantee. RK2 remains second order in time.
No positivity limiter has been added by this spatial-accuracy change.

Run the semi-discrete spatial regression (MPI compiler required):

```bash
bash tests/run_spatial_accuracy.sh       # one process
bash tests/run_spatial_accuracy.sh 2 2   # four processes
```

The test integrates an analytic nonlinear periodic state into cell averages,
compares the numerical residual with independently integrated exact face fluxes,
and checks convergence on 16, 32, 64 and 128 squared meshes. It also checks constant
preservation for every reconstruction scheme, transverse polynomial reconstruction
at all four Gauss nodes in both directions, periodic diagonal halos, and mixed
physical/MPI corners for extrapolation and slip walls. The smooth-state mean
absolute residual errors were `2.774e-3`, `9.092e-5`, `3.305e-6` on the last three
meshes (observed orders 4.93 and 4.78), matching between one and four processes.

## Boundary Conditions

The current boundary-condition module supports the following boundary names.

- zeroGradient
- slipWall
- symmetry
- inlet
- outlet
- internal

Notes:
- `internal` is used for MPI subdomain interfaces
- periodic or cyclic boundary conditions are not supported in the current code
- `symmetry` is currently treated with the same reflection logic as `slipWall`

For inlet boundaries, primitive values must be provided.
For outlet boundaries, outlet pressure must be provided.

## Initialization

Initial fields are defined through the `setFields` configuration keys.

The solver initializes the field by:

1. filling the full domain with a background primitive state
2. overwriting rectangular regions in sequence

Currently supported setFields features include:

- background primitive state through `setFields.bg.*`
- rectangular regions through `setFields.regionN.*`
- optional shock-based region construction using `shockMach` and `shockDir`

If regions overlap, later regions overwrite earlier ones.

A Sod-x initial condition can be represented by using the right state as the background field and the left state as one rectangular override region.

## Output

The current output module writes:

- ASCII legacy VTK `.vtk` files for flow-field visualization
- a single merged output file even in MPI runs

Current VTK output fields are:

- rho
- velocity
- p
- rho_u
- rho_v
- E

The VTK writer uses a rectilinear grid and can write either:

- `POINT_DATA`, obtained by averaging surrounding cell-centered values to grid points
- `CELL_DATA`, written directly from finite-volume cell-centered values

`POINT_DATA` is useful for smoother visualization in ParaView, while `CELL_DATA` is better for exact cell-wise inspection of discontinuities, conservation behavior, and diagnostics.

Typical output control keys are:

- `outPrefix`
- `outputEvery`
- `outputTimeInterval`
- `writeFinal`
- `output.vtkDataOutputType`

`outputTimeInterval` defaults to `0` (disabled). A positive, finite value selects
fixed physical-time output and overrides `outputEvery`. For example:

```ini
outputTimeInterval = 0.01
writeFinal = true
```

This writes snapshots at `t = 0, 0.01, 0.02, ...` up to `finalTime`. The solver
still computes each step from the global CFL condition, shortening a step when
needed to land on the next output time or `finalTime`. Output targets are computed
from the output index times the interval to avoid cumulative scheduling drift.
An interval smaller than the CFL step increases the number of integration steps.
Negative or non-finite intervals are rejected.

Scheduled filenames contain only the prefix and physical time, for example
`implosion_t=0.010000.vtk`; iteration counts are omitted. Final files retain
the separate name `implosion_final_t=0.300000.vtk`.

With `outputTimeInterval = 0`, the existing `outputEvery` behavior is preserved:
positive values write every N steps (including step 0), and non-positive values
disable scheduled snapshots. `writeFinal` independently writes a final file,
including when the final time already has a scheduled snapshot. If `finalTime`
falls between scheduled times, only `writeFinal` requests that last snapshot.

## Repository Structure

```text
EulerSolver_CFD_Training/
├── include/          header files
├── src/              source files
├── cases/            example configuration files
├── build_files/      object files generated by make.sh
├── make.sh           build script
├── README.md         project description and usage notes
└── LICENSE           MIT license
```

## Main Source Modules

- `src/main.cpp`              MPI initialization, case loading, 2D solver launch
- `src/solver.cpp`            2D solver workflow, time loop, CFL, output control
- `src/flux.cpp`              numerical flux implementations
- `src/reconstruction.cpp`    characteristic reconstruction and admissibility control
- `src/time_integrator.cpp`   explicit time integrators
- `src/boundary.cpp`          boundary-condition parsing and ghost-cell filling
- `src/setFields.cpp`         background-plus-region initial-field setup
- `src/io.cpp`                VTK output and MPI gather/write
- `src/mpi_parallel.cpp`      Cartesian MPI topology and halo exchange
- `include/constants.hpp`     fixed dimension and serialization conventions
- `include/structures.hpp`    named states, state algebra, packing, and grid metadata
- `src/thermodynamics.cpp`    ideal-gas EOS closure
- `src/common_functions.cpp` common flow-state operations and text/numeric utilities
- `src/read_parameters.cpp`   configuration parsing and centralized control reading

## Requirements

To build and run the current code, you need:

- a C++17-capable compiler
- an MPI implementation such as OpenMPI or MPICH
- a platform with a usable MPI compile and runtime environment

Recommended platforms:

- macOS
- Linux

Notes:
- the current source tree includes `#include <mpi.h>`, so MPI headers and libraries are required for a successful build
- the build script prefers `mpicxx`; otherwise it tries to discover MPI compile and link flags automatically

## Build

Make the build script executable and build the solver.

```bash
chmod +x make.sh
./make.sh
```

Supported build modes:

```bash
./make.sh
./make.sh release
./make.sh debug
./make.sh clean
```

To force a compiler:

```bash
CXX=clang++ ./make.sh
```

To hint an MPI installation prefix:

```bash
PREFIX=/path/to/mpi ./make.sh
```

By default, the build produces:

```bash
./solver
```

## Run

The executable requires an explicit case file path.

```bash
./solver cases/case2d.cfg
```

If no case file is provided, or extra arguments are supplied, the solver prints
`Usage: ./solver <case.cfg>` and exits with a nonzero status. There is no default case file.

### Important note about MPI decomposition

Each case file defines its MPI decomposition through:

```text
mpi.px = ...
mpi.py = ...
```

The product `mpi.px * mpi.py` must match the number of MPI processes used at runtime.

For a serial run, use:

```text
mpi.px = 1
mpi.py = 1
```

and then run:

```bash
./solver cases/case2d.cfg
```

### General MPI rule

The following condition must hold:

```text
mpi.px * mpi.py = number of MPI processes
```

If this condition is not satisfied, the solver will stop during MPI topology setup.

## Example Cases

The repository currently includes the following example case files.

- `cases/case2d.cfg`
- `cases/sod.cfg`
- `cases/lax.cfg`
- `cases/cfg_template.cfg`

### `cases/case2d.cfg`

A 2D test case configured through `setFields` rectangular regions.

### `cases/sod.cfg`

A Sod-x test represented by a background right state and one left-state rectangular override region.

### `cases/lax.cfg`

A Lax-type 1D-in-x Riemann test represented through the same `setFields` background-plus-region syntax.

### `cases/cfg_template.cfg`

A template configuration file documenting the current supported keys and recommended usage style.

## Configuration Guide

The solver uses plain-text key-value configuration files.

A minimal structure typically includes:

```text
dim = 2
nx = 400
ny = 400
ng = 4
x0 = 0.0
x1 = 1.0
y0 = 0.0
y1 = 1.0
gamma = 1.4
cfl = 0.05
finalTime = 0.30
flux = hllc
timeIntegrator = rk4
bc = zeroGradient
mpi.px = 1
mpi.py = 1
outPrefix = solution
outputEvery = 100
writeFinal = true
output.vtkDataOutputType = point
```

### Core keys

- `dim` is an optional compatibility key; if supplied, it must be `2`. The solver always uses two dimensions.
- `nx`, `ny`, `ng`
- `x0`, `x1`, `y0`, `y1`
- `gamma`, `cfl`, `finalTime`
- `flux`
- `timeIntegrator`
- `bc` and optional side-specific `bc.left`, `bc.right`, `bc.bottom`, `bc.top`
- `mpi.px`, `mpi.py`
- output keys

### Boundary-condition keys

Global default:

```text
bc = zeroGradient
```

Side overrides:

```text
bc.left   = inlet
bc.right  = outlet
bc.bottom = slipWall
bc.top    = symmetry
```

Inlet primitive values:

```text
bc.inlet.rho = 1.0
bc.inlet.u   = 0.0
bc.inlet.v   = 0.0
bc.inlet.p   = 1.0
```

Side-specific inlet values are also supported, for example:

```text
bc.left.inlet.rho = 1.0
bc.left.inlet.u   = 1.0
bc.left.inlet.v   = 0.0
bc.left.inlet.p   = 1.0
```

Outlet pressure:

```text
bc.outlet.p = 1.0
```

Side-specific outlet pressure is also supported.

### Reconstruction keys

```text
reconstruction.scheme = weno5
reconstruction.limiter = vanLeer
reconstruction.eps = 1e-12
reconstruction.variables = characteristic
```

### setFields example: Sod-x

```text
setFields.bg.rho = 0.125
setFields.bg.u   = 0.0
setFields.bg.v   = 0.0
setFields.bg.p   = 0.1

setFields.nRegions = 1

setFields.region1.xMin = 0.0
setFields.region1.xMax = 0.5
setFields.region1.yMin = 0.0
setFields.region1.yMax = 1.0
setFields.region1.rho  = 1.0
setFields.region1.u    = 0.0
setFields.region1.v    = 0.0
setFields.region1.p    = 1.0
setFields.region1.shockMach = -1.0
setFields.region1.shockDir  = +x
```

This represents a Sod-x problem by using the right state as the background and overwriting the left half of the domain.

## Current Limitations

At the current stage, the code has the following practical limitations.

- 2D only
- structured Cartesian grids only
- MPI is required at build time
- no periodic boundary support
- output format is currently legacy ASCII VTK
- this is still an actively evolving training and research prototype

## Development Direction

The current code structure is intended to support future work such as:

- further solver-architecture refinement
- stronger verification and benchmark coverage
- additional output variables
- extended setFields options for more complex initial and boundary-driven test problems
- more advanced shock and detonation-oriented test problems
- continued numerical-method development on the current modular framework

## License

This project is licensed under the MIT License.
See `LICENSE` for details.

## Acknowledgment

This repository is primarily intended for personal solver training, numerical-method development, and research-oriented experimentation in compressible CFD.

## Failure handling

Non-finite states, non-positive density or pressure, invalid sound speeds,
invalid CFL time steps, degenerate flux calculations, and failed Godunov
pressure iterations raise errors. The solver validates interior RK inputs,
reconstructed face states, numerical fluxes, and the completed step before
advancing the clock. It does not clip states, substitute wave speeds, retry
steps, or fall back to another reconstruction or flux.

The failing rank reports the error and aborts the MPI job. Numeric configuration
values must be finite and fully parsed; malformed boolean values are rejected.
The Godunov solver also rejects non-positive PVRS estimates or Newton iterates
rather than clipping them; some otherwise admissible Riemann problems may
therefore terminate with a numerical-solver error.

Build and run the failure-handling regression checks with:

```bash
./make.sh debug
python3 tests/test_architecture.py
# Optional two-rank checks, including failure on one rank only:
python3 tests/test_architecture.py --mpi
```

## State and module architecture

The solver stores a contiguous array of `Conservative_vector` objects. Each cell
has named `specific_mass`, `specific_x_momentum`, `specific_y_momentum`, and
`specific_energy` members, storing rho, rho*u, rho*v, and rho*E respectively.
Despite the `specific_*` names, these conserved quantities are per unit volume;
`specific_energy` is total energy density, not energy per unit mass.
`Primitive_vector` carries density, x/y velocity and pressure. `Flux_vector`
stores `mass_flux`, `x_momentum_flux`, `y_momentum_flux` and `energy_flux` in the
selected coordinate direction. `EvaluatedState` retains primitive quantities,
sound speed and total enthalpy per unit mass.

Physical and numerical flux functions return `ReturnCodes` and write to a
`Flux_vector` output reference only on success. Rusanov and HLLC explicitly combine the physical flux with
wave-speed-scaled conservative jumps. The solver forms the conservative residual
from the corresponding flux components; the integrators advance conservative
states. There are no implicit conversions between state and flux types.

`structures.hpp` owns these shared types, componentwise state algebra, explicit
packing functions, and `GridBlock2DInfo`. State structs are never reinterpreted
as arrays of doubles: halo exchange and VTK gathering explicitly pack/unpack
in density, x-momentum, y-momentum, total-energy order. Mathematical arrays
remain local to characteristic projection and WENO reconstruction.

`thermodynamics` implements the constant-gamma EOS closure: pressure/internal-
energy-density conversion and sound speed from density and pressure.
Following MovingBoundaries_chemistry, `common_functions` is the shared flow
operation layer. Its global functions include `calc_conservative_variables`,
`calc_primitive_variables`, `calc_sound_velocity`, `calc_flow_variables`, and
`calc_physical_flux` (x/y selected by direction). It delegates EOS calculations
to `thermodynamics`; the `trim_string`, `to_lowercase`, and `all_values_finite` functions provide text operations
and finite-value predicates. Flow functions return `ReturnCodes` and assign their
output reference only on success. The enum is defined in `constants.hpp`; callers must explicitly check every computational return code. Callers use
`require_success` from `common_functions` to report the operation and numeric error code and raise a
fatal error for MPI termination. There are no repaired states,
zero sound-speed substitutes, or silent defaults. Cached flux properties must
come from the same conservative state.

`common_functions` also provides `is_finite_state` for individual conservative
states and residuals, `is_finite_flux` for fluxes, and `require_success` for
converting failed return codes into exceptions. State conversions check physical validity at the calculation
site. `flux.cpp` contains Riemann/numerical-flux algorithms and reuses the common
physical flux functions.
`constants.hpp` contains fixed conventions, not run settings.
Parameter structures and their supporting enums are defined in `structures.hpp`;
face workspaces remain with their owning modules. Functions and variables use snake_case. The shared
vector types use the reference-project names `Primitive_vector`,
`Conservative_vector` and `Flux_vector`; other type names retain PascalCase.
Configuration keys and VTK field names remain unchanged.

Compile and run the standalone state/serialization checks:

```bash
c++ -std=c++17 -Wall -Wextra -pedantic-errors -Iinclude tests/test_state_types.cpp src/structures.cpp src/thermodynamics.cpp src/common_functions.cpp -o /tmp/euler_state_types_test
/tmp/euler_state_types_test
```

The architecture regression checks exercise all four fluxes, three
reconstruction schemes, and four time integrators, plus boundary conditions,
initialization, and strict failures. An optional pre-refactor executable can be
compared with `python3 tests/test_architecture.py --reference /path/to/old_solver`.
Comparisons permit roundoff differences (relative tolerance 1e-12, absolute
1e-13). `--mpi` additionally checks serial/four-rank VTK agreement for uneven
subdomains and abort behavior when only one rank fails.

Common flow API example:

```cpp
Conservative_vector conservative{};
const ReturnCodes code = calc_conservative_variables(primitive, gamma, conservative);
require_success(code, "initial state conversion");
```

Text transformations return strings and finite-value predicates return booleans;
they do not use computational error codes. Expected EOS numerical exceptions are
translated into return codes inside the common flow layer. Failure never commits
a partial output or authorizes continued time advancement.

The common flow functions use the same calling style as
MovingBoundaries_chemistry: an unscoped `enum ReturnCodes` in `constants.hpp`,
uppercase values such as `SUCCESS`, `NEGATIVE_PRESSURE`, and
`INVALID_SOUND_VELOCITY`, and standalone functions with output references.
Check against `SUCCESS` before using an output. The historical
`NEGATIVE_DENSITY`/`NEGATIVE_PRESSURE` names reject zero as well as negative
values. Text and finite-value utilities are also standalone functions.

## Implementation style

The implementation prioritizes visible calculations and explicit failure paths.
Flux functions and time-integrator interfaces operate on the current 2D
conservative state; no dimensional/state templates or explicit instantiation
machinery are used. MUSCL and WENO stencil loading and reconstruction use direct branches and
loops. Common flow functions use standalone calls, output references, and an
explicit `ReturnCodes` check before committing outputs. EOS exception translation
is scoped to the individual EOS call.

`std::vector`, `std::string`, and smart pointers manage storage and ownership.
The short RHS callback connects the time integrator to stage boundary/halo
updates without giving the integrator knowledge of the mesh or MPI. Fixed
constants and simple state algebra remain where they make formulas clearer.
No `[[nodiscard]]` attributes or C++17 if-initializers are required.

## Namespace policy

Project interfaces use global functions and descriptive types such as
`ReconstructionOptions`, `ReconstructionScheme`, and `BoundaryConditions2D`.
There are no project-defined namespaces or namespace aliases. Implementation-only
helper functions use file-local `static` linkage. Standard-library names keep
explicit `std::` prefixes; the code does not import `std` into global scope.

## Source formatting

All C++ headers and sources follow `.clang-format`, modeled on
MovingBoundaries_chemistry's gas_thermodynamics module: four-space indentation,
braces on separate lines, spaces inside non-empty parentheses, pointer/reference
symbols beside the variable name, and four-space continuation indentation.
Function return types and names stay on the same line; long parameter lists
continue after commas, with a 120-column target. Formatting preserves existing
statement blocks, include order, string literals, and computation.

Separate validation, local preparation, calculation, and output assignment with
blank lines. Keep return-code calls next to their checks, and keep related
formula or matrix entries together. Use two blank lines between top-level
definitions; declarations inside a class or header use a single blank line.
The formatter preserves these logical groups.

```bash
clang-format -i include/*.hpp src/*.cpp
```

## Shared constants

`constants.hpp` defines the spatial dimension, conservative component count,
coordinate directions (`x_direction`, `y_direction`), serialization component
indices, and `ReturnCodes`. State packing and conservative eigenvectors use the
same component order: density, x-momentum density, y-momentum density, total energy
density. Direction arguments still accept integers so invalid values can follow
the existing error paths.

Algorithm settings stay with their modules: Godunov iteration controls in
`flux.cpp`; the default WENO epsilon accompanies `ReconstructionOptions` in
`structures.hpp`. WENO epsilon is
a weight regularization parameter. Physical-state checks remain in the flow
calculations and reject invalid inputs without repairing them. MPI tags remain
in the MPI module. Gamma, CFL and end time remain case
settings. Reconstruction stencil sizes count cells and are independent of the
conservative component count.

## Numerical flux return codes

`flux.hpp` declares `calc_rusanov_flux`, `calc_hllc_flux`, `calc_ausm_flux` and
`calc_godunov_flux`. Each accepts two conservative states, direction, gamma and a
flux output reference. All four return `ReturnCodes`; failure leaves every output
component unchanged. `FluxScheme` and `calc_numerical_flux` select the calculation
with an explicit switch, replacing the former flux class hierarchy and factory.
The configuration names remain rusanov, hllc, ausm and godunov. Unknown names
remain configuration errors.

Expected numerical failures propagate from the calculation site without broad
exception catches. Additional codes identify invalid wave speeds, invalid star
states, unsupported vacuum, invalid pressure estimates, invalid Newton iterations,
pressure-iteration nonconvergence and invalid scheme values. All internal numerical helpers use
`static void` and output references. Helpers that can fail additionally write a
`ReturnCodes &code` status, which callers check before using numerical outputs.
Numerical outputs are committed only on success. No pressure repair or alternate
flux is applied after failure. The solver reports the selected scheme, numeric
code and global face coordinates; its existing step/stage reporting and MPI-abort
path remain in place.


## Centralized parameter reading

`read_parameters(filename, parameters, error)` in `read_parameters.cpp` reads the case once
and directly fills `CaseParameters`, grouping mesh, solver, reconstruction, boundary,
and initialization settings. All configuration key lookups and their defaults
live in this module. `main` uses the dimension and MPI layout and passes the
parsed settings to `Solver`. Reconstruction, boundary filling and setFields
consume typed settings without accessing the configuration parser.

Numerical operations and checks that depend on the local MPI mesh remain in
their owning modules. Omitted initialization-region bounds still use the local
domain bounds. Side-specific inlet values still select a complete inlet state,
rather than mixing missing components with the global inlet state.

Build with `bash make.sh` and check centralized parameter reading with
`python3 tests/test_parameters.py`. To compare the five valid cases against an
executable saved before this refactor, add `--reference /path/to/old_solver`;
their VTK files must match byte for byte. These checks run in serial.

All parameter structures (`GridParameters`, `SolverParameters`, `CaseParameters`,
`ReconstructionOptions`, `BoundarySide2D`, `BoundaryConditions2D`,
`SetFieldsRegion`, and `SetFieldsOptions`) and their supporting enums live in
`include/structures.hpp`. The reader header declares configuration-reading
interfaces; module headers declare their computational interfaces.

The reading entry point returns `ReturnCodes` and writes a diagnostic string
containing the filename, line (when available), key and reason. Internal reading
helpers are file-local `void` functions using output references. There is no
public parser class. Parameters and initialization regions are filled directly;
on failure the structure may be partially assigned and callers must stop without
using it. No temporary parameter structure or rollback is used.

`structures.hpp` contains shared type definitions and function declarations.
State algebra, serialization, structure member functions and reconstruction
queries are implemented in `src/structures.cpp`, which must be linked in
standalone checks that use those functions. The header has no inline function
definitions; `default_weno_epsilon` is a header-local `constexpr` value.
