# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a fork of [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb) (Academy Software Foundation). OpenVDB is a C++ library implementing a hierarchical (B+tree-like) sparse volumetric data structure plus a large suite of tools for storing and manipulating sparse 3D volume data. The repository is a CMake super-project of several independently-buildable components.

Fork-specific work lives on feature branches (e.g. `mesh-to-sdf`). The main upstream branch is `master`. A notable in-progress addition is `nanovdb/nanovdb/tools/cuda/MeshToGrid.cuh` — a CUDA triangle-mesh → NanoVDB UDF/IndexGrid rasterizer (authored by Efty Sifakis).

## Components

The top-level directory groups loosely-coupled components, each with its own `CMakeLists.txt`:

- **`openvdb/`** — the core C++ library. This is the foundation everything else depends on.
- **`openvdb_ax/`** — OpenVDB AX: a JIT-compiled (LLVM) expression language for processing VDB grids. Depends on core OpenVDB. Has its own lexer/parser grammar under `openvdb_ax/openvdb_ax/grammar/`.
- **`nanovdb/`** — NanoVDB: a header-only, GPU-friendly (CUDA/HIP) read-optimized counterpart to OpenVDB. Can build with or without core OpenVDB (`NANOVDB_USE_OPENVDB`). CUDA code lives in `*.cuh`/`*.cu` files.
- **`openvdb_cmd/`** — command-line tools: `vdb_print`, `vdb_render`, `vdb_lod`, `vdb_view`, `vdb_ax`, `vdb_tool`.
- **`openvdb_houdini/`**, **`openvdb_maya/`**, **`openvdb_wolfram/`** — DCC plugins.
- **`ext/`** — bundled third-party code (imath, vcl SIMD wrapper, doxygen-awesome). See `ext/THIRD-PARTY.md`.

### Core library architecture (`openvdb/openvdb/`)

The central data structure is a fixed-depth tree, assembled from templates in `tree/`:
- `RootNode` → `InternalNode` (two levels by default) → `LeafNode` — a `RootNode<InternalNode<InternalNode<LeafNode>>>` chain forms the `Tree`. `Grid` (`Grid.h`) wraps a `Tree` with a transform and metadata.
- `tree/ValueAccessor.h` provides cached, fast spatially-coherent access; `tree/LeafManager.h` / `NodeManager.h` enable parallel (TBB) traversal.
- `tools/` — the bulk of the algorithms (level sets, filtering, mesh-to-volume, particle rasterization, compositing, etc.). `tools/MeshToVolume.h` is the CPU mesh→SDF path.
- `math/` — vectors, matrices, transforms, quaternions, stencils.
- `points/` — point-data grids (points stored in leaf voxels).
- `io/` — serialization, optional delayed-loading and Blosc/ZLIB compression.
- `Types.h` / `TypeList.h` — the template type lists that drive explicit instantiation.

## Build

OpenVDB is CMake-based. Core dependencies: Boost (iostreams), TBB, Blosc, ZLIB. Optional components pull in more (LLVM for AX, CUDA for NanoVDB, GTest for unit tests, etc.).

```bash
mkdir build && cd build
cmake ..                       # configure core only
make -j$(nproc) && make install
```

Components are toggled at configure time. Most-used options (see top-level `CMakeLists.txt` for the full list):

| Option | Purpose |
| :----- | :------ |
| `-DOPENVDB_BUILD_CORE=ON`        | core library (default ON) |
| `-DOPENVDB_BUILD_AX=ON`          | build OpenVDB AX |
| `-DOPENVDB_BUILD_NANOVDB=ON`     | build NanoVDB |
| `-DNANOVDB_USE_OPENVDB=ON`       | let NanoVDB use core OpenVDB |
| `-DNANOVDB_USE_CUDA=ON`          | enable NanoVDB CUDA code (required for `MeshToGrid.cuh`, `*.cu`) |
| `-DOPENVDB_BUILD_UNITTESTS=ON`   | core unit tests |
| `-DOPENVDB_BUILD_AX_UNITTESTS=ON`| AX unit tests |
| `-DNANOVDB_BUILD_UNITTESTS=ON`   | NanoVDB unit tests |
| `-DOPENVDB_BUILD_BINARIES=ON`    | command-line tools (default ON, but only `vdb_print` unless deps present) |
| `-DOPENVDB_BUILD_PYTHON_MODULE=ON` | `pyopenvdb` Python bindings |
| `-DOPENVDB_ENABLE_ASSERTS=ON`    | enable internal asserts (off by default; turn on while developing) |
| `-DOPENVDB_CXX_STRICT=ON`        | treat compiler warnings strictly (CI uses this — build with it before submitting) |

`ci/build.sh` is the canonical CI build wrapper if you need a reference invocation (it sets deprecated-ABI bypass flags and component lists). Python wheel builds are driven by `pyproject.toml` (scikit-build-core + nanobind).

### ABI

The OpenVDB ABI version is significant: it changes data-structure layout and is gated in `CMakeLists.txt` (`OPENVDB_ABI_VERSION_NUMBER`, `MINIMUM_OPENVDB_ABI_VERSION`, `FUTURE_OPENVDB_ABI_VERSION`). Code branches on `OPENVDB_ABI_VERSION_NUMBER`. Be deliberate when touching anything guarded by it — changing struct layout can break the ABI.

## Tests

Tests use GoogleTest and are registered with CTest. Each component builds its own test executable:

- Core: `vdb_test` → CTest name `vdb_unit_test`
- AX: `vdb_ax_test`
- NanoVDB: `nanovdb_test_nanovdb` (CPU) and `nanovdb_test_cuda` (CUDA, when enabled)

Run via CTest or the executable directly. The executables are standard GTest binaries, so use `--gtest_filter` to run a subset:

```bash
cd build
ctest                                          # run all registered tests
ctest -R vdb_unit_test                         # one CTest entry
./openvdb/openvdb/vdb_test                      # run the core test binary directly
./openvdb/openvdb/vdb_test --gtest_filter='TestMeshToVolume.*'   # one suite
./openvdb/openvdb/vdb_test --gtest_filter='*testName*'           # one case
```

Unit test source files live in each component's `unittest/` (or `test/` for AX) directory, one `Test<Thing>.cc` per area.

## Changelog / pending changes convention

Do **not** edit `CHANGES` directly in a PR. Instead add a `.txt` file under `pendingchanges/` describing the change (these are periodically collapsed into `CHANGES`). This avoids merge conflicts. Follow the format of existing files there (component name, then `New features:` / `Improvements:` / `Bug Fixes:` sections). Never delete `pendingchanges/README` — git would drop the otherwise-empty directory.

## Contribution conventions (upstream)

- All commits must be DCO signed-off (`git commit --signoff`) — upstream requires a `Signed-off-by` line and a signed CLA.
- Follow the [OpenVDB coding style](https://www.openvdb.org/documentation/doxygen/codingStyle.html).
- Source files carry the Apache-2.0 SPDX header (`// Copyright Contributors to the OpenVDB Project` / `// SPDX-License-Identifier: Apache-2.0`).
- CUDA-containing headers use the `.cuh` extension and must only be included from `.cu`/`.cuh` translation units (they contain device code).
