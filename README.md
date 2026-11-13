# EulerSolver_CFD_Training

A C++17 finite-volume solver for the 2D ideal-gas Euler equations on uniform
Cartesian grids, with MPI domain decomposition. This is a training and research
prototype, limited to rectangular structured meshes and non-periodic boundaries.

## Build and run

Requires a C++17 compiler and OpenMPI or MPICH. The build script prefers `mpicxx`;
use `CXX` to select a compiler or `PREFIX` to specify an MPI installation.

```bash
bash make.sh release       # default: -O2, produces ./solver
bash make.sh debug         # -O0 -g
bash make.sh clean         # remove build artifacts

# Select a case explicitly; its mpi.px * mpi.py must equal the process count.
mpirun -np 8 ./solver cases/case2d.cfg
# For a case configured with mpi.px=1 and mpi.py=1:
./solver /path/to/serial.cfg
```

Examples live in [cases/](cases/): `case2d.cfg`, `sod.cfg`, `lax.cfg`,
`implosion.cfg` and the annotated [cfg_template.cfg](cases/cfg_template.cfg).
Each rank's local dimensions must be at least the ghost width `ng`.

## Numerical methods

| Setting | Supported values |
|---|---|
| `flux` | `rusanov`, `hllc`, `ausm`, `godunov` |
| `timeIntegrator` | `euler`, `rk2`, `ssprk3`, `rk4` |
| `reconstruction.scheme` | `firstOrder`, `muscl`, `weno5` |
| `reconstruction.limiter` | `none`, `minmod`, `vanLeer` |
| Boundary type | `zeroGradient`, `slipWall`, `symmetry`, `inlet`, `outlet` |

MUSCL and normal WENO reconstruction use characteristic projection at the
arithmetic mean of adjacent conservative states. WENO5 uses a transverse
componentwise WENO-JS reconstruction and four-point Gauss face quadrature;
`reconstruction.eps` regularizes its weights. WENO5 requires at least three
ghost layers and valid diagonal halos. MPI exchanges x before y to propagate
corner data. `firstOrder` and MUSCL need one and two ghost layers respectively.

WENO5 targets fifth-order spatial accuracy for smooth solutions; critical
points, discontinuities and physical boundary closures can reduce the order.
Time accuracy depends on the chosen integrator. `symmetry` uses slip-wall
reflection; `internal` is reserved for MPI interfaces. Legacy flux names such as
`godunovExact` and limiters `mc`/`superbee` are unsupported.

## Case configuration

Cases use `key = value` lines with `#` comments. This serial Sod example can be
saved as a `.cfg` file and passed to `./solver`:

```ini
nx = 200
ny = 8
ng = 3
x0 = 0.0
x1 = 1.0
y0 = 0.0
y1 = 0.04
gamma = 1.4
cfl = 0.05
finalTime = 0.2
flux = hllc
timeIntegrator = ssprk3
reconstruction.scheme = weno5
reconstruction.limiter = vanLeer
reconstruction.eps = 1e-12
bc = zeroGradient
mpi.px = 1
mpi.py = 1

setFields.bg.rho = 0.125
setFields.bg.u = 0.0
setFields.bg.v = 0.0
setFields.bg.p = 0.1
setFields.nRegions = 1
setFields.region1.xMin = 0.0
setFields.region1.xMax = 0.5
setFields.region1.yMin = 0.0
setFields.region1.yMax = 0.04
setFields.region1.rho = 1.0
setFields.region1.u = 0.0
setFields.region1.v = 0.0
setFields.region1.p = 1.0

outPrefix = sod
outputEvery = 0
outputTimeInterval = 0.01
writeFinal = true
output.vtkDataOutputType = cell
```

Initialization fills a background state, then applies rectangular regions in
order; later regions overwrite earlier ones. Regions also support `shockMach`
and `shockDir`. Specify region bounds explicitly for consistent MPI setup.

Override individual boundaries with `bc.left`, `bc.right`, `bc.bottom` and
`bc.top`. Inlets use `bc.inlet.rho/u/v/p`; outlets use `bc.outlet.p`. Side-specific
keys such as `bc.left.inlet.rho` or `bc.right.outlet.p` are also supported;
side-specific inlet settings select a complete state. Optional `dim` must be 2;
`reconstruction.variables`, if supplied, must be `characteristic`.

## Output and failures

Output is merged ASCII rectilinear VTK in `solution/`, with `rho`, `velocity`,
`p`, `rho_u`, `rho_v` and `E`. `output.vtkDataOutputType` selects `cell` (finite-volume
values) or `point` (averaged visualization values).

A positive `outputTimeInterval` overrides `outputEvery`, writes from time zero,
and shortens CFL steps to reach scheduled times. Otherwise a positive
`outputEvery` writes every N steps, including step zero. `writeFinal` independently
controls the final snapshot. Filenames are `<prefix>_t=0.010000.vtk` and
`<prefix>_final_t=0.200000.vtk`.

Invalid configurations, non-finite values, non-positive density/pressure and
numerical failures abort the MPI job with operation/location diagnostics. The
solver does not clip states, retry steps, apply a positivity limiter or fall
back to another scheme. The Godunov pressure iteration can reject otherwise
admissible Riemann problems; failures are reported rather than repaired.

## Tests and performance

See [test/README.md](test/README.md) for Riemann symmetry/baseline comparisons,
MPI runs, performance timing and result logs. A small Sod regression can be run
with:

```bash
python3 test/run_tests.py prepare --case '*/*/*/1_sod/*' --cfl 0.05
python3 test/run_tests.py run --case '*/*/*/1_sod/*' --keep-going
python3 test/compare_results.py --case '*/*/*/1_sod/*'
```

Symmetry comparisons check directional consistency, not accuracy against an
analytic solution. The benchmark measures complete process wall time, including
MPI startup; it does not provide module timings or peak memory usage.

## Architecture and coding conventions

| Files | Responsibility |
|---|---|
| `main.cpp`, `solver.cpp`, `time_integrator.cpp` | MPI startup, solver workflow and time advancement |
| `flux.cpp`, `reconstruction.cpp`, `boundary.cpp` | Numerical fluxes, face states and ghost boundaries |
| `mpi_parallel.cpp`, `io.cpp`, `setFields.cpp` | Halo exchange, VTK output and initialization |
| `thermodynamics.cpp`, `common_functions.cpp` | Ideal-gas EOS, flow operations and common utilities |
| `structures.hpp/.cpp`, `constants.hpp` | Shared types, state algebra, serialization and fixed conventions |
| `read_parameters.cpp` | Centralized configuration keys/defaults and typed parameter loading |

The contiguous `Conservative_vector` array stores rho, rho*u, rho*v and total
energy density rho*E; its historical `specific_*` names denote quantities per
volume. `Primitive_vector` stores rho/u/v/p, `EvaluatedState` adds sound speed
and specific total enthalpy, and `Flux_vector` stores directional fluxes.
Explicit packing preserves component order without relying on struct layout.

The following conventions apply to C++ headers, sources and tests. Formatting
follows the LLVM baseline with the layout of MovingBoundaries_chemistry's
gas_thermodynamics module:

- Use C++17, `snake_case` functions/local variables, existing shared-vector and
  PascalCase type names, and `UPPER_CASE` return-code enumerators. Preserve case
  keys and VTK field names. Qualify standard names with `std::`; add no project
  namespaces, namespace aliases or global using-directives. File-local helpers
  use `static` linkage.
- Keep shared types and typed parameters in `structures.hpp`, their member and
  utility implementations in `structures.cpp`, and module interfaces/algorithms
  in matching headers/sources. Avoid inline definitions in the shared structures
  header. Read keys/defaults only in `read_parameters.cpp`; pass typed settings
  to modules and keep local-grid checks with `Solver`. Place the reading entry
  point before helpers, grouped by parsing, basic values and module options.
- Prefer direct formulas, ordinary structs and explicit control flow, with
  `std::vector`, `std::string` and smart pointers for ownership. Add templates or
  abstraction layers only when the numerical code benefits.
- Check computational `ReturnCodes` against `SUCCESS` before using outputs.
  Numerical helpers use `void` with output references where appropriate; status
  functions use `ReturnCodes`. Commit numerical outputs only on success; retain
  diagnostics and the MPI fatal path. Parameter reading writes directly to its
  caller's structure, which must not be used after a failed, partial read.
- Document functions in English: a concise purpose, each actual parameter name
  with `(in)`, `(out)` or `(in/out)`, then relevant return/failure behavior.
  Include callback/buffer data flow, assumptions, units and array layout. Keep
  declarations and definitions consistent; update comments with interfaces and
  avoid merely restating individual statements.
- Use four spaces, no tabs, four-space continuation indentation and a 120-column
  target. Access labels sit two spaces left of members; `case` labels align with
  `switch`. Use Allman braces and expand short functions, blocks, conditionals,
  loops, lambdas and enums across lines.
- Keep return types with function names and scope-resolution expressions intact.
  Wrap argument/parameter lists after commas with continuation indentation,
  without aligning to opening parentheses or forcing every declaration parameter
  onto the next line. Pack multiple arguments on a line when they fit.
- Attach pointer/reference symbols to variable names: `double *output`,
  `const Conservative_vector &state`. Use `if ( condition )`, `evaluate( state )`
  and `run()`; omit padding inside C-style cast parentheses.
- Separate validation, preparation, calculation and output assignment with blanks;
  keep return-code calls beside checks and related formulas together. Use two
  blank lines between top-level definitions, one between class/header declarations,
  and at most two consecutive blank lines; preserve deliberate grouping.
- Preserve include/using order, comment layout, string literals and computation.
  Formatting must not reflow comments, split strings, insert/remove braces,
  parentheses or semicolons, or alter namespace comments.

## License

[MIT](LICENSE).
