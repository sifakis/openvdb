# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of the [OpenVDB](https://www.openvdb.org) repository. The active development branch (`mesh_to_grid`) is implementing a **GPU-accelerated triangle mesh to NanoVDB grid converter** — specifically `nanovdb::tools::cuda::MeshToGrid<BuildT>` — which rasterizes a triangle surface mesh into a sparse NanoVDB index grid on the device.

The primary new files relative to upstream are:
- `nanovdb/nanovdb/tools/cuda/MeshToGrid.cuh` — the main CUDA tool header (device code)
- `nanovdb/nanovdb/examples/ex_mesh_to_grid_cuda/` — example driver (`.cpp` + `.cu`)

## Building

The NanoVDB sub-project has its own standalone build in `nanovdb/nanovdb/build/`. The existing CMake cache targets CUDA arch `120` with CUDA 12.9 and links against a pre-installed OpenVDB at `~/local`.

To reconfigure and rebuild:
```bash
cd nanovdb/nanovdb/build
cmake .. \
  -DNANOVDB_USE_CUDA=ON \
  -DNANOVDB_USE_OPENVDB=ON \
  -DNANOVDB_USE_TBB=ON \
  -DNANOVDB_BUILD_EXAMPLES=ON \
  -DNANOVDB_BUILD_UNITTESTS=ON \
  -DNANOVDB_ALLOW_FETCHCONTENT=ON \
  -DOpenVDB_ROOT=~/local \
  -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc)
```

To build just the mesh-to-grid example:
```bash
cd nanovdb/nanovdb/build
make -j$(nproc) ex_mesh_to_grid_cuda
```

To run it:
```bash
nanovdb/nanovdb/build/examples/ex_mesh_to_grid_cuda/ex_mesh_to_grid_cuda input.obj [output.vdb] [voxel_size]
```

## Running Tests

```bash
cd nanovdb/nanovdb/build
ctest                         # all tests
ctest -R TestNanoVDB          # specific test
./unittest/TestNanoVDB        # run directly
```

## Architecture

### NanoVDB Tree Structure
NanoVDB uses a fixed 4-level hierarchy: **Root → Upper (4096³) → Lower (512³) → Leaf (8³) → Voxels**. Node dimensions are powers of 8; the root node uses a hash map of tiles. The core data structure is header-only in `nanovdb/nanovdb/NanoVDB.h`.

### MeshToGrid Algorithm (`tools/cuda/MeshToGrid.cuh`)

**Current goal**: enumerate all `(leafNode, triangle)` pairs that intersect (within `mBandWidth` voxels). Tree construction comes after.

**Coordinate convention**: After `transformTriangles()`, all geometry lives in floating-point index space where the cell at integer index `i` is a unit cube centered at `i`, occupying the geometric interval `[i-0.5, i+0.5]`. The correct cell index for a geometric coordinate `x` is therefore `floor(x + 0.5)`.

**Constraints**: Only isotropic linear transforms are supported (uniform scale + rotation + translation, no anisotropic scaling or projective component). The scalar `mBandWidth` expansion is only meaningful in isotropic index space.

**Empty input**: When `triangleCount == 0`, the algorithm should gracefully return an empty grid. Currently it throws — this is a known deferred TODO.

The converter works via a hierarchical top-down subdivision approach on the GPU:

1. **`transformTriangles()`** — Transforms all mesh vertices from world space to NanoVDB index space using `nanovdb::Map::applyInverseMap()`. Result is stored in `mXformedTriangles` (device buffer of `std::array<Vec3f,3>`). Degenerate (zero-area) triangles survive the transform unchanged and are harmless.

2. **`processRootTrianglePairs()`** — Seeds `mBoxTrianglePairsBuffer` with all `(rootTile, triangle)` pairs whose AABBs overlap. Uses a conservative AABB-only test (no SAT) — acceptable because each triangle typically touches only 1 root tile (4096³ voxels). Three-pass scatter pattern:
   - Pass 1: `CountRootBoxesFunctor` — per triangle, count overlapping root tiles.
   - Pass 2: CUB `DeviceScan::InclusiveSum` prefix scan → write offsets + total count.
   - Pass 3: `ScatterRootTrianglePairsFunctor` — scatter `BoxTrianglePair` structs (16B: `Coord origin` + `uint32_t triangleID`) at pre-computed offsets. No atomics needed.

   **Padding logic**: the continuous region with UDF ≤ `mBandWidth` is `[xmin - mBandWidth, xmax + mBandWidth]`. The integer cell indices whose centers fall in that interval are `ceilf(xmin - mBandWidth)` to `floorf(xmax + mBandWidth)`. Those voxel indices are then mapped to root tile indices via `floorf(voxel * invRootDim)`.

3. **`processLeafTrianglePairs()`** — Hierarchical subdivision (3 passes: 4096→512→64→8), refining `mBoxTrianglePairsBuffer` in place (ping-pong). Each pass:
   - `evaluateAndCountSubBoxesKernel` — 1 CTA per parent pair, 512 threads (one per 8³ sub-box). Uses `testTriangleAABB<OnlyUseAABB>` for triangle-AABB intersection. Warp ballot via `__ballot_sync` writes results directly into a `Mask<3>` via `reinterpret_cast` on an `alignas`-qualified `uint32_t[16]` shared array (avoids union constructor issues). `countOn()` gives hit count.
   - AABB-only test used for `childScale >= mSATThreshold` (default 128); full SAT used below.
   - Prefix scan → allocate child pair buffer → `ScatterChildPairsFunctor` (1 thread per parent, iterates set bits) → ping-pong swap.
   - **padding**: `mBandWidth` is passed as padding (tight bound would be `mBandWidth - 0.5`; extra 0.5 is deliberate conservatism).

**Validated on dragon.obj at voxel size 0.0005** (871K triangles, 220K reference leaves):
- Pair counts per pass: 897K (root) → 929K → 1.3M → 8.1M (leaf/triangle pairs)
- 278K unique leaf origins in output vs 220K reference leaves — ~27% conservatism, all reference leaves present ✓
- Evaluate & count dominates timing (~4ms at finest scale); scatter is fast (~0.8ms)

**Next step**: build the NanoVDB `ValueOnIndex` tree from the leaf/triangle pair list.

The narrow-band half-width (`mBandWidth`, default 3.0 voxels) controls the geometric expansion applied during AABB overlap tests.

`CALL_CUBS` is a utility macro wrapping the two-call CUB pattern (null-pointer size query → reallocate → execute) using `mTempDevicePool`.

### Example Driver (`ex_mesh_to_grid_cuda/`)
- `.cpp` file: reads OBJ, builds OpenVDB level set as reference, uploads mesh to device via `thrust::universal_vector`, creates `nanovdb::Map` from the OpenVDB transform, calls `mainMeshToGrid<ValueOnIndex>()`.
- `.cu` file: instantiates `MeshToGrid<BuildT>`, runs `getHandle()`, then downloads and validates the `BoxTrianglePair` output.

### Key Dependencies
- **CCCL (CUB/Thrust)** — fetched via CMake `FetchContent` at configure time into `nanovdb/nanovdb/build/_deps/cccl-src/`
- **OpenVDB** — pre-installed at `~/local`; required for the example and `NANOVDB_USE_OPENVDB=ON` builds
- **CUDA 12.9** at `/usr/local/cuda-12.9`
