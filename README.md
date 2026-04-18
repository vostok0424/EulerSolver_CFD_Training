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
- two initialization paths: built-in IC or setFields-style region initialization
- merged single-file VTK output for serial and MPI runs
- optional CSV state-diagnostics output

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
- vanleer

Important notes:
- reconstruction is now characteristic-based only
- `reconstruction.variables` is kept only as a compatibility key
- if `reconstruction.variables` is provided, it should be `characteristic`
- older limiter names such as `mc` and `superbee` are not supported by the current implementation

Additional reconstruction controls currently used by the solver include:

- `reconstruction.enableFallback`
- `reconstruction.positivityFix`
- `reconstruction.eps`
- `reconstruction.rhoMin`
- `reconstruction.pMin`

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

## Initialization Paths

The current code provides two initialization routes.

### 1. Built-in IC path

Use:

- `setFields.use = false`
- `ic = sodx`

The currently implemented built-in IC is:

- `sodx`

Relevant keys are:

- `ic.sodx.xMid`
- `ic.sodx.rhoL`, `ic.sodx.uL`, `ic.sodx.vL`, `ic.sodx.pL`
- `ic.sodx.rhoR`, `ic.sodx.uR`, `ic.sodx.vR`, `ic.sodx.pR`

### 2. setFields path

Use:

- `setFields.use = true`

This path initializes the field by:

1. filling the full domain with a background primitive state
2. overwriting rectangular regions in sequence

Currently supported setFields features include:

- background primitive state through `setFields.bg.*`
- rectangular regions through `setFields.regionN.*`
- optional shock-based region construction using `shockMach` and `shockDir`

If regions overlap, later regions overwrite earlier ones.

## Output

The current output module writes:

- ASCII legacy VTK `.vtk` files for flow-field visualization
- a single merged output file even in MPI runs
- optional CSV state diagnostics

Current VTK output fields are:

- rho
- velocity
- p
- rho_u
- rho_v
- E

The VTK writer uses a rectilinear grid and writes point data obtained from surrounding cell averages.

Typical output control keys are:

- `outPrefix`
- `outputEvery`
- `writeFinal`

Typical diagnostics keys are:

- `stateDiagnostics.enable`
- `stateDiagnostics.csv`

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
- `src/ic.cpp`                built-in initial conditions
- `src/setFields.cpp`         region-based initialization
- `src/io.cpp`                VTK output and MPI gather/write
- `src/mpi_parallel.cpp`      Cartesian MPI topology and halo exchange
- `src/state.cpp`             state conversion, admissibility checks, diagnostics
- `src/cfg.cpp`               key-value configuration parser

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

The executable accepts an optional case file path.

```bash
./solver cases/case2d_sod.cfg
```

If no configuration file is provided, the solver defaults to:

```bash
cases/case2d_sod.cfg
```

### Important note about the default case

The current default case file `cases/case2d_sod.cfg` uses:

```text
mpi.px = 4
mpi.py = 2
```

So it should be run with 8 MPI processes unless you edit the decomposition.

Example:

```bash
mpirun -np 8 ./solver cases/case2d_sod.cfg
```

If you want a serial run, change the case file to:

```text
mpi.px = 1
mpi.py = 1
```

and then run:

```bash
./solver cases/case2d_sod.cfg
```

### General MPI rule

The following condition must hold:

```text
mpi.px * mpi.py = number of MPI processes
```

If this condition is not satisfied, the solver will stop during MPI topology setup.

## Example Cases

The repository currently includes the following example case files.

- `cases/case2d_sod.cfg`
- `cases/quad_riemann.cfg`
- `cases/cfg_template.cfg`

### `cases/case2d_sod.cfg`

A narrow-channel 2D Sod-type test configured through `setFields`.

### `cases/quad_riemann.cfg`

A four-region 2D Riemann-type problem configured through rectangular regions.

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
```

### Core keys

- `dim` must be `2`
- `nx`, `ny`, `ng`
- `x0`, `x1`, `y0`, `y1`
- `gamma`, `cfl`, `finalTime`
- `flux`
- `timeIntegrator`
- `bc` and optional side-specific `bc.left`, `bc.right`, `bc.bottom`, `bc.top`
- `mpi.px`, `mpi.py`
- output and diagnostics keys

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
reconstruction.limiter = vanleer
reconstruction.enableFallback = true
reconstruction.positivityFix = true
reconstruction.eps = 1e-12
reconstruction.rhoMin = 1e-12
reconstruction.pMin = 1e-12
reconstruction.variables = characteristic
```

### Built-in IC example

```text
setFields.use = false
ic = sodx
ic.sodx.xMid = 0.5
ic.sodx.rhoL = 1.0
ic.sodx.uL   = 0.0
ic.sodx.vL   = 0.0
ic.sodx.pL   = 1.0
ic.sodx.rhoR = 0.125
ic.sodx.uR   = 0.0
ic.sodx.vR   = 0.0
ic.sodx.pR   = 0.1
```

### setFields example

```text
setFields.use = true
setFields.bg.rho = 1.0
setFields.bg.u   = 0.0
setFields.bg.v   = 0.0
setFields.bg.p   = 1.0
setFields.nRegions = 1
setFields.region1.xMin = 0.0
setFields.region1.xMax = 0.5
setFields.region1.yMin = 0.0
setFields.region1.yMax = 1.0
setFields.region1.rho  = 2.0
setFields.region1.u    = 0.0
setFields.region1.v    = 0.0
setFields.region1.p    = 2.0
setFields.region1.shockMach = -1.0
setFields.region1.shockDir  = +x
```

## Current Limitations

At the current stage, the code has the following practical limitations.

- 2D only
- structured Cartesian grids only
- MPI is required at build time
- no periodic boundary support
- built-in IC support is currently minimal
- output format is currently legacy ASCII VTK
- this is still an actively evolving training and research prototype

## Development Direction

The current code structure is intended to support future work such as:

- further solver-architecture refinement
- stronger verification and benchmark coverage
- improved diagnostics and output variables
- additional initial-condition and boundary-condition options
- more advanced shock and detonation-oriented test problems
- continued numerical-method development on the current modular framework

## License

This project is licensed under the MIT License.
See `LICENSE` for details.

## Acknowledgment

This repository is primarily intended for personal solver training, numerical-method development, and research-oriented experimentation in compressible CFD.
