#!/usr/bin/env bash
# make.sh — portable build script for EulerSolver_CFD_Training
#
# Modes:
#   ./make.sh            -> release build (default), output: ./solver
#   ./make.sh release    -> release build, output: ./solver
#   ./make.sh debug      -> debug build, output: ./solver
#   ./make.sh clean      -> remove build artifacts
#
# Optional environment variables:
#   CXX=clang++          -> force compiler
#   PREFIX=/path/to/mpi  -> hint MPI prefix if auto-detection fails

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="solver"
MODE="${1:-release}"
PREFIX="${PREFIX:-}"

case "$MODE" in
  release|debug|clean)
    ;;
  *)
    echo "Error: unsupported mode: $MODE" >&2
    echo "Usage: ./make.sh [release|debug|clean]" >&2
    exit 1
    ;;
esac

BUILD_DIR="build_files"

if [[ "$MODE" == "clean" ]]; then
  echo "Cleaning build artifacts..."
  rm -rf "$BUILD_DIR"
  rm -f "$TARGET"
  echo "Done."
  exit 0
fi

# Base flags
if [[ "$MODE" == "debug" ]]; then
  CXXFLAGS_BASE="-std=c++17 -O0 -g -Iinclude -Wall -Wextra"
  BUILD_NAME="Debug"
else
  CXXFLAGS_BASE="-std=c++17 -O2 -Iinclude -Wall -Wextra"
  BUILD_NAME="Release"
fi

# Choose compiler driver
if [[ -z "${CXX:-}" ]]; then
  if command -v mpicxx >/dev/null 2>&1; then
    CXX="$(command -v mpicxx)"
  else
    CXX="c++"
  fi
fi

MPI_COMPILE_FLAGS=""
MPI_LINK_FLAGS=""

# Helper: try to find a usable mpicxx
find_mpicxx() {
  if command -v mpicxx >/dev/null 2>&1; then
    echo "$(command -v mpicxx)"
    return 0
  fi
  if command -v mpic++ >/dev/null 2>&1; then
    echo "$(command -v mpic++)"
    return 0
  fi
  return 1
}

# Helper: check prefix has mpi.h and libmpi
prefix_has_mpi() {
  local p="$1"
  [[ -f "$p/include/mpi.h" ]] || return 1
  [[ -f "$p/lib/libmpi.dylib" || -f "$p/lib/libmpi.so" || -f "$p/lib/libmpi.a" ]] || return 1
  return 0
}

# If compiler is not mpicxx, attempt to add MPI flags.
# Priority:
#  1) mpicxx --showme:* if mpicxx exists
#  2) PREFIX if provided
#  3) common prefixes
if [[ "$(basename "$CXX")" != "mpicxx" && "$(basename "$CXX")" != "mpic++" ]]; then
  if MPICXX_PATH="$(find_mpicxx 2>/dev/null || true)"; [[ -n "$MPICXX_PATH" ]]; then
    MPI_COMPILE_FLAGS="$("$MPICXX_PATH" --showme:compile 2>/dev/null || true)"
    MPI_LINK_FLAGS="$("$MPICXX_PATH" --showme:link 2>/dev/null || true)"
    if [[ -z "$MPI_COMPILE_FLAGS$MPI_LINK_FLAGS" ]]; then
      MPI_LINK_FLAGS="$("$MPICXX_PATH" --showme 2>/dev/null || true)"
    fi
  else
    CANDIDATES=()
    [[ -n "$PREFIX" ]] && CANDIDATES+=("$PREFIX")
    CANDIDATES+=("/opt/homebrew" "/usr/local" "/opt/local" "/usr/bin")

    for p in "${CANDIDATES[@]}"; do
      if prefix_has_mpi "$p"; then
        MPI_COMPILE_FLAGS="-I${p}/include"
        MPI_LINK_FLAGS="-L${p}/lib -lmpi"
        break
      fi
    done
  fi
fi

echo "Compiler: $CXX"
echo "Mode: $BUILD_NAME"
echo "Target: ./$TARGET"
if [[ -n "$MPI_COMPILE_FLAGS$MPI_LINK_FLAGS" ]]; then
  echo "MPI flags: $MPI_COMPILE_FLAGS $MPI_LINK_FLAGS"
else
  if [[ "$(basename "$CXX")" != "mpicxx" && "$(basename "$CXX")" != "mpic++" ]]; then
    echo "Warning: MPI flags not detected. If MPI build is needed, install OpenMPI/MPICH or set PREFIX=/path/to/mpi" >&2
  fi
fi

# Collect sources
shopt -s nullglob
SRC=(src/*.cpp)

# Support split state module if placed in project root
if [[ -f "state.cpp" ]]; then
  SRC+=("state.cpp")
fi

if (( ${#SRC[@]} == 0 )); then
  echo "Error: no source files found in ./src/*.cpp or ./state.cpp" >&2
  exit 1
fi

# Sanity check for split state module
if [[ -f "state.cpp" || -f "src/state.cpp" ]]; then
  if [[ ! -f "state.hpp" && ! -f "include/state.hpp" ]]; then
    echo "Error: state.cpp exists, but state.hpp was not found in project root or include/." >&2
    exit 1
  fi
fi

mkdir -p "$BUILD_DIR"

OBJ=()
for f in "${SRC[@]}"; do
  OBJ+=("$BUILD_DIR/$(basename "${f%.cpp}.o")")
done

echo "[1/2] Compiling objects..."
for i in "${!SRC[@]}"; do
  echo "  ${SRC[$i]} -> ${OBJ[$i]}"
  if [[ "${SRC[$i]}" == "state.cpp" || "${SRC[$i]}" == "src/state.cpp" ]]; then
    echo "    building split state module"
  fi
  "$CXX" $CXXFLAGS_BASE $MPI_COMPILE_FLAGS -c "${SRC[$i]}" -o "${OBJ[$i]}"
done

echo "[2/2] Linking -> ./$TARGET"
"$CXX" "${OBJ[@]}" -o "$TARGET" $MPI_LINK_FLAGS

echo ""
echo "============================"
echo "Compile complete - Go for simulations."
echo "============================"
echo "Run serial: ./$TARGET cases/<configuration_file_name>.cfg"
echo "Run parallel: mpirun -np 4 ./$TARGET cases/<configuration_file_name>.cfg (ensure px*py == np)"
