// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda_kernels.cu
///
/// @brief CUDA / NanoVDB side of the mesh->SDF example.
///
///        Three passes over an opaque SdfPipeline (the host driver holds only a pointer):
///          buildMeshToSdf()     rasterize -> prune -> CC -> sign -> fill (steps 1-6)
///          validateMeshToSdf()  independent CPU oracles + OpenVDB / analytic cross-checks
///          exportMeshToSdf()    dump the Polyscope visualization (.ccvis + .fill)
///        The connected-components labeling and the mesh->SDF library live in
///        nanovdb/tools/cuda/{ConnectedComponents,MeshToSDF}.cuh. See MeshToSDF_PipelinePlan.md
///        and MeshToSDFDevelopmentPlan.md in this directory for the design notes.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h>

#ifdef NANOVDB_USE_OPENVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>  // meshToLevelSet — independent sign cross-check
#endif

#include <thrust/universal_vector.h>

#include <algorithm>  // std::sort
#include <cmath>      // std::sqrt, std::fabs
#include <cstdint>
#include <cstdlib>    // std::getenv
#include <fstream>    // std::ofstream (visualization export)
#include <iostream>
#include <memory>     // std::unique_ptr (SdfPipeline)
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Must match the aliases in connected_components_cuda.cpp.
using GridHandleT   = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using UDFSidecarT   = nanovdb::cuda::DeviceBuffer;
using IndexSidecarT = nanovdb::cuda::DeviceBuffer;

// Summary of validateMeshToSdf's checks, returned so the in-code analytic self-tests can assert on it.
// (Definition must match in connected_components_cuda.cpp.)
struct SDFResult {
    uint64_t globalComponents            = 0;      // distinct global CC labels
    bool     openvdbChecked              = false;  // OpenVDB cross-check ran (needs NANOVDB_USE_OPENVDB)
    uint64_t confidentSignMismatches     = 0;      // OpenVDB cross-check, beyond-shell (must be 0)
    uint64_t inShellTies                 = 0;      // OpenVDB cross-check, within √3/2-voxel shell
    bool     analyticChecked             = false;  // sphere ground-truth check ran
    uint64_t analyticConfidentMismatches = 0;      // sphere ground-truth, beyond-shell (must be 0)
    uint64_t analyticInShellTies         = 0;      // sphere ground-truth, within √3/2-voxel shell
    bool     invertChecked               = false;  // leaf invert-mask analytic check ran (step 6 chunk A)
    uint64_t invertMismatches            = 0;      // inactive voxels whose invert bit disagrees w/ analytic
    uint64_t invertOnBits                = 0;      // inactive voxels marked interior (always reported)
    bool     coarseInvertChecked         = false;  // lower/upper invert-mask analytic check ran (chunk B)
    uint64_t coarseInvertMismatches      = 0;      // childless tiles whose invert bit disagrees w/ analytic
    uint64_t lowerOnTiles                = 0;      // childless lower tiles marked interior
    uint64_t upperOnTiles                = 0;      // childless upper tiles marked interior
    uint64_t rootInteriorCells           = 0;      // absent root cells marked deep-interior (chunk C)
    bool     fullDomainChecked           = false;  // full-domain signedSignAt sweep ran (chunk C)
    uint64_t fullDomainMismatches        = 0;      // sampled coords whose queried sign disagrees w/ analytic
    uint64_t fullDomainTies              = 0;      // sampled coords within the on-surface tie band
};

// All live device state produced by buildMeshToSdf and consumed (read-only) by validateMeshToSdf /
// exportMeshToSdf. Opaque to the host driver (which only holds an SdfPipeline*), since it embeds the
// CUDA-only pipeline objects. The two objects own device buffers and are built once the derived grid
// exists, hence unique_ptr. (Phase 2 will move this struct to a .cu-only header.)
struct SdfPipeline {
    GridHandleT   orig, derived;   // original (pre-prune) + derived (barrier-pruned) index grids
    UDFSidecarT   udf;             // per-active-voxel unsigned distance (world units)
    IndexSidecarT index;          // per-active-voxel nearest-triangle index
    nanovdb::Map  map;            // index<->world transform
    float         bandWidth = 0.f;
    std::unique_ptr<nanovdb::tools::cuda::ConnectedComponents<nanovdb::ValueOnIndex>> cc;   // step 3 labels
    std::unique_ptr<nanovdb::tools::cuda::MeshToSDF<nanovdb::ValueOnIndex>>           sdf;  // steps 4-6 signs+fill
};

/// @brief Step 1: rasterize the mesh into the index grid + UDF sidecar AND a per-active-voxel
///        NEAREST-TRIANGLE-INDEX sidecar (uint32), needed by step 5
///        (barrier signing). See MeshToGrid::getHandleAndUDFAndIndex. After building, a CPU oracle
///        re-derives each active voxel's distance from its stored triangle and checks it against the
///        UDF, and confirms background / no-hit voxels carry the INVALID (0xFFFFFFFF) index.
/// @return { index-grid handle, UDF sidecar, nearest-triangle-index sidecar } (device buffers)
std::tuple<GridHandleT, UDFSidecarT, IndexSidecarT> computeUDFAndIndex(
    const std::vector<nanovdb::Vec3f>& points,
    const std::vector<nanovdb::Vec3i>& triangles,
    const nanovdb::Map&                map,
    float                              bandWidth)
{
    using BuildT = nanovdb::ValueOnIndex;

    thrust::universal_vector<nanovdb::Vec3f> dPoints(points.begin(), points.end());
    thrust::universal_vector<nanovdb::Vec3i> dTriangles(triangles.begin(), triangles.end());

    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(
        dPoints.data().get(),    uint32_t(dPoints.size()),
        dTriangles.data().get(), uint32_t(dTriangles.size()),
        map);
    converter.setVerbose(1);
    converter.setNarrowBandWidth(bandWidth);

    auto result = converter.getHandleAndUDFAndIndex();
    auto& handle       = std::get<0>(result);
    auto& udfSidecar   = std::get<1>(result);
    auto& indexSidecar = std::get<2>(result);

    // ---- CPU oracle: the stored nearest-triangle index must reproduce the UDF (pipeline step 1) ----
    const float        voxelSize      = float(map.getVoxelSize()[0]);
    const float        bandWidthWorld = bandWidth * voxelSize;
    constexpr uint32_t INVALID        = 0xFFFFFFFFu;

    const uint64_t        n = udfSidecar.size() / sizeof(float);  // activeVoxelCount + 1 (slot 0 = background)
    std::vector<float>    udf(n);
    std::vector<uint32_t> index(n);
    cudaCheck(cudaMemcpy(udf.data(),   udfSidecar.deviceData(),   n * sizeof(float),    cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(index.data(), indexSidecar.deviceData(), n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Byte-copy the device grid blob to a pinned host buffer (NanoVDB is position-independent).
    const uint64_t gridBytes = handle.bufferSize();
    void* hostBlob = nullptr;
    cudaCheck(cudaMallocHost(&hostBlob, gridBytes));
    cudaCheck(cudaMemcpy(hostBlob, handle.deviceData(), gridBytes, cudaMemcpyDeviceToHost));
    const auto* h_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(hostBlob);

    const auto&    tree      = h_grid->tree();
    const uint32_t leafCount = tree.nodeCount(0);
    const auto*    leaves    = tree.getFirstLeaf();
    const float    tol       = 1e-3f * bandWidthWorld + 1e-6f;  // generous vs host/device float divergence

    std::size_t mismatches = 0, hits = 0, misses = 0, oob = 0, shown = 0;

    // Background slot 0 must carry the INVALID index.
    if (index[0] != INVALID) { ++mismatches; std::cerr << "  index[0] (background) != INVALID\n"; }

    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto&         leaf = leaves[li];
        const nanovdb::Coord o   = leaf.origin();
        for (int q = 0; q < 512; ++q) {
            if (!leaf.isActive(uint32_t(q))) continue;
            const uint64_t vIdx = leaf.getValue(uint32_t(q));
            const uint32_t tid  = index[vIdx];
            const float    d    = udf[vIdx];
            if (tid == INVALID) {
                // No-hit (false-positive) voxel: the UDF is clamped to the band width.
                ++misses;
                if (std::fabs(d - bandWidthWorld) > tol) {
                    ++mismatches;
                    if (shown++ < 10)
                        std::cerr << "  no-hit voxel udf=" << d << " != bandWidth=" << bandWidthWorld << "\n";
                }
                continue;
            }
            if (tid >= triangles.size()) { ++oob; ++mismatches; continue; }
            ++hits;
            // Recompute the distance to the stored triangle in index space (mirrors the device).
            const nanovdb::Vec3i& T  = triangles[tid];
            const nanovdb::Vec3f  v0 = map.applyInverseMap(points[T[0]]);
            const nanovdb::Vec3f  v1 = map.applyInverseMap(points[T[1]]);
            const nanovdb::Vec3f  v2 = map.applyInverseMap(points[T[2]]);
            const nanovdb::Vec3f  c(float(o[0] + (q >> 6)), float(o[1] + ((q >> 3) & 7)), float(o[2] + (q & 7)));
            const float dRecomp = std::sqrt(nanovdb::math::pointToTriangleDistSqr(v0, v1, v2, c)) * voxelSize;
            if (std::fabs(dRecomp - d) > tol) {
                ++mismatches;
                if (shown++ < 10)
                    std::cerr << "  voxel @ leaf " << li << " off " << q << ": stored tri " << tid
                              << " dist " << dRecomp << " != udf " << d << "\n";
            }
        }
    }
    cudaCheck(cudaFreeHost(hostBlob));

    std::cout << "UDF nearest-triangle index validation:  " << (mismatches == 0 ? "PASS" : "FAIL")
              << " (" << (n - 1) << " active voxels: " << hits << " hits, " << misses << " no-hit; "
              << mismatches << " mismatches"
              << (oob ? ", " + std::to_string(oob) + " out-of-range" : "") << ")\n";

    return result;
}

/// @brief Print topology diagnostics for a device-resident ValueOnIndex grid, in the
///        style of ex_dilate_nanovdb_cuda. Reads the grid's header/tree fields directly
///        off the device via DeviceGridTraits (small targeted cudaMemcpy's) - no full
///        deviceDownload of the grid buffer is required.
void printGridDiagnostics(const GridHandleT& handle, const std::string& title)
{
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

    const auto* d_grid = handle.deviceGrid<BuildT>();

    const auto     treeData     = Traits::getTreeData(d_grid);
    const uint64_t valueCount   = Traits::getValueCount(d_grid);
    const uint64_t activeVoxels = Traits::getActiveVoxelCount(d_grid);
    const uint64_t gridSize     = Traits::getGridSize(d_grid);
    const auto     bbox         = Traits::getIndexBBox(d_grid, treeData);

    const uint64_t leafNodes  = treeData.mNodeCount[0];
    const uint64_t lowerNodes = treeData.mNodeCount[1];
    const uint64_t upperNodes = treeData.mNodeCount[2];

    std::cout << "============ " << title << " ============\n";
    std::cout << "Allocated values [valueCount()]       : " << valueCount << "\n";
    std::cout << "Active voxels    [activeVoxelCount()] : " << activeVoxels << "\n";
    std::cout << "Index-space bounding box              : ["
              << bbox.min().x() << "," << bbox.min().y() << "," << bbox.min().z() << "] -> ["
              << bbox.max().x() << "," << bbox.max().y() << "," << bbox.max().z() << "]\n";
    std::cout << "Leaf nodes                            : " << leafNodes  << "\n";
    std::cout << "Lower internal nodes                  : " << lowerNodes << "\n";
    std::cout << "Upper internal nodes                  : " << upperNodes << "\n";
    std::cout << "Leaf-level occupancy                  : "
              << (leafNodes ? 100.f * float(activeVoxels) / float(leafNodes * 512) : 0.f)
              << "%\n";
    std::cout << "Memory usage                          : " << gridSize << " bytes\n";
}

// Step 2 (SDF domain): prune the surface/barrier shell into the derived CC-input grid. The barrier
// test, retain mask, and PruneGrid call all live in the MeshToSDF library tool; this is a thin seam.
GridHandleT computeDerivedTopology(const GridHandleT& srcHandle, const UDFSidecarT& udfSidecar,
                                   float voxelSize)
{
    using BuildT = nanovdb::ValueOnIndex;
    nanovdb::tools::cuda::MeshToSDF<BuildT> sdf;
    sdf.setVerbose(1);
    return sdf.computeDerivedTopology(srcHandle.deviceGrid<BuildT>(),
                                      static_cast<const float*>(udfSidecar.deviceData()), voxelSize);
}

namespace {

/// @brief CPU reference for per-leaf 6-connected component counts on a host-resident
///        ValueOnIndex grid. Harvested from TestNanoVDB.cu's LeafConnectedComponents
///        oracle: a union-find local to each 8^3 leaf (parent[n]=n for active voxels,
///        -1 for inactive; union only active +X/+Y/+Z in-leaf neighbors so each undirected
///        face edge is visited once; larger root attaches under the smaller; the component
///        count is the number of surviving roots). Counts are in leaf storage order, to
///        match the device-side deviceLeafComponentCounts().
template <typename GridT>
std::vector<uint16_t> cpuLeafComponentCounts(const GridT* grid)
{
    const auto&    tree      = grid->tree();
    const uint32_t leafCount = tree.nodeCount(0);
    const auto*    leaves    = tree.getFirstLeaf();

    int parent[512] = {};// re-initialized per leaf below
    auto find = [&](int i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    };
    auto unite = [&](int a, int b) {
        const int ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;// attach larger root under smaller root
        else         parent[ra] = rb;
    };

    std::vector<uint16_t> counts(leafCount);
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (int n = 0; n < 512; ++n) parent[n] = leaf.isActive(uint32_t(n)) ? n : -1;
        // x-major offset n = (x<<6)|(y<<3)|z, so +X/+Y/+Z in-leaf neighbors are n+64/n+8/n+1.
        for (int n = 0; n < 512; ++n) {
            if (parent[n] < 0) continue;// inactive
            const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
            if (x < 7 && parent[n + 64] >= 0) unite(n, n + 64);
            if (y < 7 && parent[n +  8] >= 0) unite(n, n +  8);
            if (z < 7 && parent[n +  1] >= 0) unite(n, n +  1);
        }
        uint16_t c = 0;
        for (int n = 0; n < 512; ++n) if (parent[n] == n) ++c;// surviving roots == components
        counts[li] = c;
    }
    return counts;
}

/// @brief CPU reference for the per-component leaf masks and 6 face bitmasks, mirroring
///        cpuLeafComponentCounts(): the same per-leaf 6-connected union-find, then components
///        enumerated in ascending root-label order (which matches the GPU kernel's repeated
///        BlockReduce-min enumeration, so global component slots line up). For each component
///        it builds the Mask<3> footprint and the 6 face bitmasks directly from voxel
///        coordinates -- intentionally NOT mirroring the kernel's shift/0x0101... extraction,
///        so this is an independent check. Bit conventions follow the LeafNeighborTap enum in
///        tools/cuda/ConnectedComponents.cuh:
///          +/-X face: bit y*8+z ;  +/-Y face: bit x*8+z ;  +/-Z face: bit y*8+x.
struct CpuMasksFaces {
    std::vector<nanovdb::Mask<3>> masks;  // K masks, one per leaf-local component (global slot order)
    std::vector<uint64_t>         faces;  // 6*K: face t of component c at faces[6*c + t]
    uint64_t                      K{0};   // total leaf-local components (== deviceLeafComponentOffsets()[leafCount])
};

template <typename GridT>
CpuMasksFaces cpuMasksFaces(const GridT* grid)
{
    namespace cc = nanovdb::tools::cuda;  // for the LeafNeighborTap enumerators (minusX..plusZ)
    const auto&    tree      = grid->tree();
    const uint32_t leafCount = tree.nodeCount(0);
    const auto*    leaves    = tree.getFirstLeaf();

    int parent[512] = {};// re-initialized per leaf below
    auto find = [&](int i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    };
    auto unite = [&](int a, int b) {
        const int ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;// attach larger root under smaller root
        else         parent[ra] = rb;
    };

    CpuMasksFaces out;
    int rootIdx[512];
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (int n = 0; n < 512; ++n) parent[n] = leaf.isActive(uint32_t(n)) ? n : -1;
        for (int n = 0; n < 512; ++n) {
            if (parent[n] < 0) continue;// inactive
            const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
            if (x < 7 && parent[n + 64] >= 0) unite(n, n + 64);
            if (y < 7 && parent[n +  8] >= 0) unite(n, n +  8);
            if (z < 7 && parent[n +  1] >= 0) unite(n, n +  1);
        }
        // Dense local component indices in ascending root-label (== ascending offset) order.
        int localCount = 0;
        for (int n = 0; n < 512; ++n) rootIdx[n] = -1;
        for (int n = 0; n < 512; ++n) if (parent[n] == n) rootIdx[n] = localCount++;

        std::vector<nanovdb::Mask<3>> lmask(localCount);
        for (auto& m : lmask) m.setOff();
        std::vector<uint64_t> lface(std::size_t(6) * localCount, 0);

        for (int n = 0; n < 512; ++n) {
            if (parent[n] < 0) continue;
            const int k = rootIdx[find(n)];
            const int x = n >> 6, y = (n >> 3) & 7, z = n & 7;
            lmask[k].setOn(uint32_t(n));
            uint64_t* f = &lface[6 * k];
            if (x == 0) f[cc::minusX] |= uint64_t(1) << (y * 8 + z);
            if (x == 7) f[cc::plusX ] |= uint64_t(1) << (y * 8 + z);
            if (y == 0) f[cc::minusY] |= uint64_t(1) << (x * 8 + z);
            if (y == 7) f[cc::plusY ] |= uint64_t(1) << (x * 8 + z);
            if (z == 0) f[cc::minusZ] |= uint64_t(1) << (y * 8 + x);
            if (z == 7) f[cc::plusZ ] |= uint64_t(1) << (y * 8 + x);
        }
        out.masks.insert(out.masks.end(), lmask.begin(), lmask.end());
        out.faces.insert(out.faces.end(), lface.begin(), lface.end());
        out.K += uint64_t(localCount);
    }
    return out;
}

/// @brief CPU reference for the cross-leaf connectivity edges (pipeline step 5). Mirrors the device
///        CrossLeafEdge[] output: for each leaf and each of its +X/+Y/+Z neighbor leaves, pairs every
///        local component with every neighbor local component and emits a canonical (a<b) edge over
///        global component slots iff their touching face masks intersect. Uses the already-validated
///        per-component face masks (cmf.faces) and component offsets (compOffsets), and the same host
///        probeLeaf neighbor walk the kernel uses, so the two sides match bit-for-bit. Returned
///        unsorted; the caller sorts both sides and compares as a set.
template <typename GridT>
std::vector<nanovdb::tools::cuda::CrossLeafEdge>
cpuCrossLeafEdges(const GridT* grid, const CpuMasksFaces& cmf, const std::vector<uint64_t>& compOffsets)
{
    namespace cc = nanovdb::tools::cuda;
    const auto&    tree      = grid->tree();
    const uint32_t leafCount = tree.nodeCount(0);
    const auto*    leaves    = tree.getFirstLeaf();

    const int faceL[3] = { cc::plusX,  cc::plusY,  cc::plusZ  };  // this leaf's +axis face
    const int faceN[3] = { cc::minusX, cc::minusY, cc::minusZ };  // neighbor's matching -axis face

    std::vector<cc::CrossLeafEdge> edges;
    for (uint32_t L = 0; L < leafCount; ++L) {
        const uint64_t baseL  = compOffsets[L];
        const int      countL = int(compOffsets[L + 1] - baseL);
        const nanovdb::Coord o = leaves[L].origin();
        for (int axis = 0; axis < 3; ++axis) {
            const nanovdb::Coord no = (axis == 0) ? o.offsetBy(8, 0, 0)
                                    : (axis == 1) ? o.offsetBy(0, 8, 0)
                                                  : o.offsetBy(0, 0, 8);
            const auto* nptr = tree.root().probeLeaf(no);
            if (!nptr) continue;
            const uint64_t N = uint64_t(nptr - leaves);
            const uint64_t baseN  = compOffsets[N];
            const int      countN = int(compOffsets[N + 1] - baseN);
            const int fL = faceL[axis], fN = faceN[axis];
            for (int i = 0; i < countL; ++i)
                for (int j = 0; j < countN; ++j)
                    if (cmf.faces[6 * (baseL + i) + fL] & cmf.faces[6 * (baseN + j) + fN]) {
                        const uint32_t ga = uint32_t(baseL + i), gb = uint32_t(baseN + j);
                        cc::CrossLeafEdge e;
                        e.a = (ga < gb) ? ga : gb;
                        e.b = (ga < gb) ? gb : ga;
                        edges.push_back(e);
                    }
        }
    }
    return edges;
}

/// @brief CPU reference for barrier signing (pipeline step 5), validating MeshToSDF::signBarrier.
///        Non-barrier voxels (signIn != 0) must be carried through unchanged. For each barrier voxel
///        (signIn == 0) the GPU's order-independent outcome is "exterior (+1) iff some exterior (+1,
///        non-INVALID) neighbor's signed dot is > 0, else interior (-1)" — the early-out / two passes
///        in the kernel are only an optimization and cannot change that OR. The oracle scans all 26
///        active neighbors (one host ReadAccessor), reusing the SAME __hostdev__ math, and tracks the
///        MAXIMUM signed dot over the exterior anchors; expected = (maxDot > 0) ? +1 : -1.
///
///        The decision is a sign-of-dot test, so a voxel exactly on the surface tangent plane
///        (|maxDot| within float noise of 0) is genuinely ambiguous. With the double-precision sign
///        test (matching the GPU) host/device agree to ~1e-13, so this band is essentially empty; the
///        tiny guard (|maxDot| <= kDotTieEps) still reports any such exact tie as @a ambiguous rather
///        than a mismatch. Any disagreement OUTSIDE that band is a real mismatch.
template <typename GridT>
std::size_t cpuSignBarrier(const GridT* g,
                           const std::vector<int8_t>&        signIn,    // post-injection snapshot (anchors)
                           const std::vector<int8_t>&        gpuSigned, // GPU result to validate
                           const std::vector<uint32_t>&      index,     // nearest-triangle-index sidecar
                           const std::vector<nanovdb::Vec3f>& points,
                           const std::vector<nanovdb::Vec3i>& triangles,
                           const nanovdb::Map&               map,
                           std::size_t& barrierCount, std::size_t& residualZeros, std::size_t& ambiguous)
{
    constexpr double kDotTieEps = 1e-9;  // |signed dot| below this == exact surface-tangent tie
    const auto&    tree      = g->tree();
    const uint32_t leafCount = tree.nodeCount(0);
    const auto*    leaves    = tree.getFirstLeaf();
    auto           acc       = g->getAccessor();

    std::size_t mism = 0; barrierCount = 0; residualZeros = 0; ambiguous = 0; std::size_t shown = 0;
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto&          leaf   = leaves[li];
        const nanovdb::Coord origin = leaf.origin();
        for (int nn = 0; nn < 512; ++nn) {
            if (!leaf.isActive(uint32_t(nn))) continue;
            const uint64_t qv = leaf.getValue(uint32_t(nn));
            if (gpuSigned[qv] == int8_t(0)) ++residualZeros;

            if (signIn[qv] != int8_t(0)) {                 // non-barrier: must be carried through
                if (gpuSigned[qv] != signIn[qv]) ++mism;
                continue;
            }

            ++barrierCount;
            const int lx = nn >> 6, ly = (nn >> 3) & 7, lz = nn & 7;
            const nanovdb::Vec3d q_xyz(double(origin[0] + lx), double(origin[1] + ly), double(origin[2] + lz));

            // Maximum signed dot over the 26 active exterior anchors (-1 sentinel = no anchor), in
            // double precision to match the GPU's barrierExteriorProof.
            double maxDot = -1.0;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const nanovdb::Coord nijk(origin[0] + lx + dx, origin[1] + ly + dy, origin[2] + lz + dz);
                        if (!acc.isActive(nijk)) continue;
                        const uint64_t nv = acc.getValue(nijk);
                        if (signIn[nv] != int8_t(1)) continue;            // only exterior anchors
                        const uint32_t tid = index[nv]; if (tid == 0xFFFFFFFFu) continue;
                        const nanovdb::Vec3i& T  = triangles[tid];
                        const nanovdb::Vec3f& p0 = points[T[0]];
                        const nanovdb::Vec3f& p1 = points[T[1]];
                        const nanovdb::Vec3f& p2 = points[T[2]];
                        const nanovdb::Vec3d  v0 = map.applyInverseMap(nanovdb::Vec3d(p0[0], p0[1], p0[2]));
                        const nanovdb::Vec3d  v1 = map.applyInverseMap(nanovdb::Vec3d(p1[0], p1[1], p1[2]));
                        const nanovdb::Vec3d  v2 = map.applyInverseMap(nanovdb::Vec3d(p2[0], p2[1], p2[2]));
                        const nanovdb::Vec3d  nxyz(static_cast<double>(nijk[0]), static_cast<double>(nijk[1]), static_cast<double>(nijk[2]));
                        double a, b;
                        const nanovdb::Vec3d cp = nanovdb::math::closestPointOnTriangleToPoint(v0, v1, v2, nxyz, a, b);
                        nanovdb::Vec3d dn = nxyz - cp; dn.normalize();
                        nanovdb::Vec3d dq = q_xyz - cp; dq.normalize();
                        const double d = dn.dot(dq);
                        if (d > maxDot) maxDot = d;
                    }

            const int8_t expected = (maxDot > 0.0) ? int8_t(1) : int8_t(-1);
            if (gpuSigned[qv] == expected) continue;
            if (std::fabs(maxDot) <= kDotTieEps) { ++ambiguous; continue; }  // exact surface-tangent tie

            ++mism;
            if (shown++ < 10)
                std::cerr << "  barrier mismatch @ (" << q_xyz[0] << "," << q_xyz[1] << "," << q_xyz[2]
                          << ") gpu=" << int(gpuSigned[qv]) << " cpu=" << int(expected)
                          << " maxDot=" << maxDot << "\n";
        }
    }
    return mism;
}

} // anonymous namespace

// ---------------------------------------------------------------------------------------------------
// Visualization export (Polyscope Sparse Volume Grid).
//
// Dumps one record per ORIGINAL-grid active voxel — [i,j,k, cc, sign, udf] — to a compact binary the
// companion python viewer (scripts/mesh_to_sdf_viewer.py) reads via np.fromfile. cc is the derived-grid
// connected-component label (barrier voxels, absent from the derived grid, get cc = -1 so they show as
// their own category); sign is the final signed-level-set sign (step 5); udf is the world-space
// unsigned distance. Gated by the CC_EXPORT_VIS env var so normal runs / self-tests are unaffected.
//
// File layout (little-endian):
//   char   magic[8] = "CCVIS001"
//   uint64 N                       (number of records)
//   double voxelSize               (uniform; world units per voxel)
//   double tx, ty, tz              (world position of index origin (0,0,0))
//   N × { int32 i,j,k,cc,sign;  float udf }   (24 bytes each)
// ---------------------------------------------------------------------------------------------------
void exportMeshToSdf(const SdfPipeline* p, const std::string& path)
{
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;
    const auto& origHandle    = p->orig;
    const auto& derivedHandle = p->derived;
    auto&       cc            = *p->cc;
    auto&       sdf           = *p->sdf;
    const auto& udfSidecar    = p->udf;
    const auto& map           = p->map;

    const auto* d_grid = derivedHandle.deviceGrid<BuildT>();
    const auto* o_grid = origHandle.deviceGrid<BuildT>();

    // --- derived grid (host copy) + CC arrays: gives every non-barrier voxel its component label ---
    const uint32_t derLeafCount = Traits::getTreeData(d_grid).mNodeCount[0];
    const uint64_t derBytes     = derivedHandle.bufferSize();
    void* derBlob = nullptr;
    cudaCheck(cudaMallocHost(&derBlob, derBytes));
    cudaCheck(cudaMemcpy(derBlob, derivedHandle.deviceData(), derBytes, cudaMemcpyDeviceToHost));
    const auto* h_der = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(derBlob);

    std::vector<uint64_t> offsets(std::size_t(derLeafCount) + 1);
    cudaCheck(cudaMemcpy(offsets.data(), cc.deviceLeafComponentOffsets(),
                         (std::size_t(derLeafCount) + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const uint64_t K = derLeafCount ? offsets[derLeafCount] : 0;
    std::vector<nanovdb::Mask<3>> masks(K);
    if (K) cudaCheck(cudaMemcpy(masks.data(), cc.deviceLeafComponentMasks(),
                                std::size_t(K) * sizeof(nanovdb::Mask<3>), cudaMemcpyDeviceToHost));
    std::vector<uint64_t> parent(K);
    if (K) cudaCheck(cudaMemcpy(parent.data(), cc.deviceComponentParent(),
                                std::size_t(K) * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    const auto* dFirst = derLeafCount ? h_der->tree().getFirstLeaf() : nullptr;
    auto voxelSlot = [&](uint32_t li, uint32_t n) -> uint64_t {       // voxel -> its global component slot
        for (uint64_t s = offsets[li]; s < offsets[li + 1]; ++s)
            if (masks[s].isOn(n)) return s;
        return offsets[li];
    };

    // --- original grid (host copy) + final signs + UDF: the voxel set we actually export ---
    const uint64_t origActive = Traits::getActiveVoxelCount(o_grid);
    const uint64_t origBytes  = origHandle.bufferSize();
    void* origBlob = nullptr;
    cudaCheck(cudaMallocHost(&origBlob, origBytes));
    cudaCheck(cudaMemcpy(origBlob, origHandle.deviceData(), origBytes, cudaMemcpyDeviceToHost));
    const auto* h_orig = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(origBlob);

    std::vector<int8_t> signs(origActive + 1);
    cudaCheck(cudaMemcpy(signs.data(), sdf.deviceSignedVoxelSign(),
                         std::size_t(origActive + 1) * sizeof(int8_t), cudaMemcpyDeviceToHost));
    std::vector<float> udf(origActive + 1);
    cudaCheck(cudaMemcpy(udf.data(), static_cast<const float*>(udfSidecar.deviceData()),
                         std::size_t(origActive + 1) * sizeof(float), cudaMemcpyDeviceToHost));

    // World transform (this example uses a uniform scale, zero translation, but read it generically).
    const nanovdb::Vec3d w0 = map.applyMap(nanovdb::Vec3d(0.0, 0.0, 0.0));
    const nanovdb::Vec3d wx = map.applyMap(nanovdb::Vec3d(1.0, 0.0, 0.0));
    const double voxelSize  = wx[0] - w0[0];

    std::ofstream out(path, std::ios::binary);
    if (!out) { std::cerr << "CC_EXPORT_VIS: cannot open " << path << " for writing\n"; }
    else {
        const uint32_t origLeafCount = Traits::getTreeData(o_grid).mNodeCount[0];
        const auto*    oFirst        = origLeafCount ? h_orig->tree().getFirstLeaf() : nullptr;

        out.write("CCVIS001", 8);
        const uint64_t N = origActive;
        out.write(reinterpret_cast<const char*>(&N), sizeof(N));
        out.write(reinterpret_cast<const char*>(&voxelSize), sizeof(double));
        for (int a = 0; a < 3; ++a) { const double t = w0[a]; out.write(reinterpret_cast<const char*>(&t), sizeof(double)); }

        uint64_t written = 0, barrier = 0;
        for (uint32_t li = 0; li < origLeafCount; ++li) {
            const auto& oleaf = oFirst[li];
            const nanovdb::Coord o = oleaf.origin();
            // matching derived leaf (same origin); pointer arithmetic gives its CC-array index
            const auto* dleaf = dFirst ? h_der->tree().root().probeLeaf(o) : nullptr;
            const uint32_t dli = dleaf ? uint32_t(dleaf - dFirst) : 0u;
            for (uint32_t n = 0; n < 512; ++n) {
                if (!oleaf.isActive(n)) continue;
                const nanovdb::Coord ijk = o + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
                int32_t cclabel = -1;                                 // -1 = barrier (not in derived grid)
                if (dleaf && dleaf->isActive(n)) cclabel = int32_t(parent[voxelSlot(dli, n)]);
                else ++barrier;
                const uint64_t vi = oleaf.getValue(n);
                const int32_t  ii[5] = { ijk[0], ijk[1], ijk[2], cclabel, int32_t(signs[vi]) };
                const float    d     = udf[vi];
                out.write(reinterpret_cast<const char*>(ii), sizeof(ii));
                out.write(reinterpret_cast<const char*>(&d), sizeof(float));
                ++written;
            }
        }
        std::cout << "CC_EXPORT_VIS: wrote " << written << " voxels (" << barrier
                  << " barrier, cc=-1) to " << path << "  [voxelSize=" << voxelSize << "]\n";
    }

    // --- Step-6 interior fill (deep interior) -> companion "<path>.fill" ---
    // One box per interior region, at its tree level: inactive interior leaf voxels (1^3), childless
    // interior lower (8^3) / upper (128^3) tiles, and absent-root interior tiles (4096^3). Each record
    // is the box's base VOXEL coord + level {0,1,2,3}; the viewer renders each as a cube sized to its
    // level. This is what makes the deep interior — which lives in coarse childless tiles, not active
    // voxels — visible for debugging step 6.
    {
        const uint32_t origLeafCount = Traits::getTreeData(o_grid).mNodeCount[0];
        const uint32_t lowerCount = h_orig->tree().nodeCount(1);
        const uint32_t upperCount = h_orig->tree().nodeCount(2);
        std::vector<nanovdb::Mask<3>> leafInv(origLeafCount);
        std::vector<nanovdb::Mask<4>> lowInv(lowerCount);
        std::vector<nanovdb::Mask<5>> upInv(upperCount);
        if (origLeafCount) cudaCheck(cudaMemcpy(leafInv.data(), sdf.deviceLeafInvertMask(),
                             std::size_t(origLeafCount) * sizeof(nanovdb::Mask<3>), cudaMemcpyDeviceToHost));
        if (lowerCount) cudaCheck(cudaMemcpy(lowInv.data(), sdf.deviceLowerInvertMask(),
                             std::size_t(lowerCount) * sizeof(nanovdb::Mask<4>), cudaMemcpyDeviceToHost));
        if (upperCount) cudaCheck(cudaMemcpy(upInv.data(), sdf.deviceUpperInvertMask(),
                             std::size_t(upperCount) * sizeof(nanovdb::Mask<5>), cudaMemcpyDeviceToHost));
        const nanovdb::Coord rMin = sdf.rootTileMin(), rDims = sdf.rootTileDims();
        const uint64_t rTotal = uint64_t(rDims[0]) * uint64_t(rDims[1]) * uint64_t(rDims[2]);
        std::vector<uint8_t> rootOn(rTotal);
        if (rTotal) cudaCheck(cudaMemcpy(rootOn.data(), sdf.deviceRootInterior(), rTotal, cudaMemcpyDeviceToHost));

        std::vector<int32_t> recs;   // 4 ints per box: baseVoxel x,y,z, level
        const auto* oFirst = origLeafCount ? h_orig->tree().getFirstLeaf() : nullptr;
        for (uint32_t li = 0; li < origLeafCount; ++li) {                        // level 0: 1^3
            const auto& lf = oFirst[li]; const nanovdb::Coord o = lf.origin();
            for (uint32_t n = 0; n < 512; ++n) {
                if (lf.isActive(n) || !leafInv[li].isOn(n)) continue;
                const nanovdb::Coord ijk = o + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
                recs.insert(recs.end(), { ijk[0], ijk[1], ijk[2], 0 });
            }
        }
        const auto* loN = lowerCount ? h_orig->tree().template getFirstNode<1>() : nullptr;
        for (uint32_t ni = 0; ni < lowerCount; ++ni) {                          // level 1: 8^3
            const auto& nd = loN[ni];
            for (uint32_t n = 0; n < 4096; ++n) {
                if (nd.childMask().isOn(n) || !lowInv[ni].isOn(n)) continue;
                const nanovdb::Coord g = nd.offsetToGlobalCoord(n);
                recs.insert(recs.end(), { g[0], g[1], g[2], 1 });
            }
        }
        const auto* upN = upperCount ? h_orig->tree().template getFirstNode<2>() : nullptr;
        for (uint32_t ni = 0; ni < upperCount; ++ni) {                          // level 2: 128^3
            const auto& nd = upN[ni];
            for (uint32_t n = 0; n < 32768; ++n) {
                if (nd.childMask().isOn(n) || !upInv[ni].isOn(n)) continue;
                const nanovdb::Coord g = nd.offsetToGlobalCoord(n);
                recs.insert(recs.end(), { g[0], g[1], g[2], 2 });
            }
        }
        for (int i = 0; i < rDims[0]; ++i)                                        // level 3: 4096^3
            for (int j = 0; j < rDims[1]; ++j)
                for (int k = 0; k < rDims[2]; ++k)
                    if (rootOn[(uint64_t(i) * rDims[1] + j) * rDims[2] + k]) {
                        const nanovdb::Coord t = rMin + nanovdb::Coord(i, j, k);
                        recs.insert(recs.end(), { t[0] * 4096, t[1] * 4096, t[2] * 4096, 3 });
                    }

        const std::string fillPath = std::string(path) + ".fill";
        std::ofstream fout(fillPath, std::ios::binary);
        if (!fout) { std::cerr << "CC_EXPORT_VIS: cannot open " << fillPath << "\n"; }
        else {
            const uint64_t M = recs.size() / 4;
            fout.write("CCFILL01", 8);
            fout.write(reinterpret_cast<const char*>(&M), sizeof(M));
            fout.write(reinterpret_cast<const char*>(&voxelSize), sizeof(double));
            for (int a = 0; a < 3; ++a) { const double t = w0[a]; fout.write(reinterpret_cast<const char*>(&t), sizeof(double)); }
            fout.write(reinterpret_cast<const char*>(recs.data()), recs.size() * sizeof(int32_t));
            std::cout << "CC_EXPORT_VIS: wrote " << M << " interior-fill boxes to " << fillPath << "\n";
        }
    }

    cudaCheck(cudaFreeHost(derBlob));
    cudaCheck(cudaFreeHost(origBlob));
}

// ---------------------------------------------------------------------------------------------------
// BUILD : run the full mesh->SDF pipeline (steps 1-6) in order and return the live device state. No
// validation, no visualization — those are separate passes over the returned SdfPipeline.
// ---------------------------------------------------------------------------------------------------
SdfPipeline* buildMeshToSdf(const std::vector<nanovdb::Vec3f>& points,
                            const std::vector<nanovdb::Vec3i>& triangles,
                            const nanovdb::Map& map, float bandWidth)
{
    using BuildT = nanovdb::ValueOnIndex;
    auto* p = new SdfPipeline;
    p->map = map;
    p->bandWidth = bandWidth;
    const float voxelSize = float(map.getVoxelSize()[0]);

    // STEP 1: rasterize the mesh -> UDF + nearest-triangle-index sidecars (index oracle runs inside).
    std::tie(p->orig, p->udf, p->index) = computeUDFAndIndex(points, triangles, map, bandWidth);
    printGridDiagnostics(p->orig, "Rasterized UDF grid");

    // STEP 2: prune the surface/barrier shell -> derived (CC-input) topology.
    p->derived = computeDerivedTopology(p->orig, p->udf, voxelSize);
    printGridDiagnostics(p->derived, "Derived CC-input grid");
    auto* d_grid = p->derived.deviceGrid<BuildT>();

    // STEP 3: connected components on the derived grid (per-leaf -> cross-leaf edges -> global labels).
    p->cc = std::make_unique<nanovdb::tools::cuda::ConnectedComponents<BuildT>>(d_grid);
    auto& cc = *p->cc;
    cc.setVerbose(1);
    cc.processLeafConnectedComponents(); cudaCheck(cudaDeviceSynchronize());
    cc.processCrossLeafEdges();          cudaCheck(cudaDeviceSynchronize());
    cc.processComponentLabels();         cudaCheck(cudaDeviceSynchronize());

    p->sdf = std::make_unique<nanovdb::tools::cuda::MeshToSDF<BuildT>>();
    auto& sdf = *p->sdf;
    sdf.setVerbose(1);

    // STEP 4: sign the non-barrier voxels from the labeling, then inject the signs onto the original.
    sdf.signNonBarrier(d_grid, cc);                                 cudaCheck(cudaDeviceSynchronize());
    sdf.injectSignsToOriginal(p->orig.deviceGrid<BuildT>(), d_grid); cudaCheck(cudaDeviceSynchronize());

    // STEP 5: sign the barrier voxels in place on the original grid (mirror of OpenVDB's
    // ComputeIntersectingVoxelSign); needs the mesh on the device + the nearest-triangle-index sidecar.
    thrust::universal_vector<nanovdb::Vec3f> dPoints(points.begin(), points.end());
    thrust::universal_vector<nanovdb::Vec3i> dTriangles(triangles.begin(), triangles.end());
    sdf.signBarrier(p->orig.deviceGrid<BuildT>(),
                    static_cast<const uint32_t*>(p->index.deviceData()),
                    dPoints.data().get(), dTriangles.data().get(), map);
    cudaCheck(cudaDeviceSynchronize());

    // STEP 6: complete the level set via per-level invert-mask sidecars (leaf -> lower/upper -> root).
    sdf.fillLeafInvertMask(p->orig.deviceGrid<BuildT>());    cudaCheck(cudaDeviceSynchronize());
    sdf.fillCoarseInvertMasks(p->orig.deviceGrid<BuildT>());  cudaCheck(cudaDeviceSynchronize());
    sdf.fillRootInteriorMask(p->orig.deviceGrid<BuildT>());   cudaCheck(cudaDeviceSynchronize());

    return p;
}

void freeSdfPipeline(SdfPipeline* p) { delete p; }

// ---------------------------------------------------------------------------------------------------
// VALIDATE : independent CPU oracles + OpenVDB / analytic cross-checks over a built pipeline. Reads the
// live device arrays through the pipeline (read-only) and returns the metrics summary.
// ---------------------------------------------------------------------------------------------------
SDFResult validateMeshToSdf(const SdfPipeline* p,
               const std::vector<nanovdb::Vec3f>& points,
               const std::vector<nanovdb::Vec3i>& triangles,
               const double*                      analyticSpheres,   // numSpheres × {Cx,Cy,Cz,R} world, or nullptr
               int                                numAnalyticSpheres,
               const double*                      analyticBoxes,     // numBoxes × {Cx,Cy,Cz,halfExtent} world, or nullptr
               int                                numAnalyticBoxes)
{
    SDFResult result;
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

    const auto&                  origHandle    = p->orig;
    const auto&                  derivedHandle = p->derived;
    const auto&                  indexSidecar  = p->index;
    const auto&                  map           = p->map;
    [[maybe_unused]] const float bandWidth     = p->bandWidth;   // used only by the OpenVDB cross-check
    auto&                        cc            = *p->cc;
    auto&                        sdf           = *p->sdf;

    const auto* d_grid = derivedHandle.deviceGrid<BuildT>();

    const uint32_t leafCount = Traits::getTreeData(d_grid).mNodeCount[0];
    if (leafCount == 0) { std::cout << "CC validation: empty grid, nothing to check\n"; return result; }

    // GPU result -> host.
    std::vector<uint16_t> gpuCounts(leafCount);
    cudaCheck(cudaMemcpy(gpuCounts.data(), cc.deviceLeafComponentCounts(),
                         std::size_t(leafCount) * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // CPU validation. NanoVDB grids are position-independent (relative offsets), so a raw
    // byte copy of the device blob into a (32B-aligned) scratch host buffer is itself a valid
    // host grid. We don't touch the handle (no deviceDownload residue); the scratch is freed
    // immediately after the check.
    const uint64_t gridBytes = derivedHandle.bufferSize();
    void* hostBlob = nullptr;
    cudaCheck(cudaMallocHost(&hostBlob, gridBytes));   // pinned, >=32B aligned
    cudaCheck(cudaMemcpy(hostBlob, derivedHandle.deviceData(), gridBytes, cudaMemcpyDeviceToHost));
    const auto* h_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(hostBlob);

    const std::vector<uint16_t> cpuCounts = cpuLeafComponentCounts(h_grid);

    std::size_t mismatches = 0;
    uint64_t    gpuTotal = 0, cpuTotal = 0;
    for (uint32_t li = 0; li < leafCount; ++li) {
        gpuTotal += gpuCounts[li];
        cpuTotal += cpuCounts[li];
        if (gpuCounts[li] != cpuCounts[li]) {
            if (mismatches < 10) {
                // host active popcount for this leaf
                const auto& hleaf = h_grid->tree().getFirstLeaf()[li];
                int act = 0; for (int q = 0; q < 512; ++q) if (hleaf.isActive(uint32_t(q))) ++act;
                std::cerr << "  CC mismatch @ leaf " << li << ": gpu=" << gpuCounts[li]
                          << " cpu=" << cpuCounts[li] << " (host active voxels=" << act << ")\n";
            }
            ++mismatches;
        }
    }

    std::cout << "CC per-leaf component-count validation: "
              << (mismatches == 0 ? "PASS" : "FAIL") << " ("
              << leafCount << " leaves, " << mismatches << " mismatches; "
              << "total components gpu=" << gpuTotal << " cpu=" << cpuTotal << ")\n";

    // ---- Validate per-component leaf masks and face flags (pipeline step 4) ----
    // Offsets give K (total components) and each leaf's slot range; masks/faces are flat,
    // indexed by global component slot = offsets[leafID] + localIdx.
    std::vector<uint64_t> gpuOffsets(std::size_t(leafCount) + 1);
    cudaCheck(cudaMemcpy(gpuOffsets.data(), cc.deviceLeafComponentOffsets(),
                         (std::size_t(leafCount) + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const uint64_t K = gpuOffsets[leafCount];

    std::vector<nanovdb::Mask<3>> gpuMasks(K);
    cudaCheck(cudaMemcpy(gpuMasks.data(), cc.deviceLeafComponentMasks(),
                         std::size_t(K) * sizeof(nanovdb::Mask<3>), cudaMemcpyDeviceToHost));
    std::vector<uint64_t> gpuFaces(std::size_t(6) * K);
    cudaCheck(cudaMemcpy(gpuFaces.data(), cc.deviceLeafComponentFaceMasks(),
                         std::size_t(6) * K * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    const CpuMasksFaces cmf = cpuMasksFaces(h_grid);

    std::size_t maskMismatch = 0, faceMismatch = 0, shownMF = 0;
    for (uint64_t c = 0; c < K; ++c) {
        bool mbad = false;
        for (int w = 0; w < 8; ++w)
            if (gpuMasks[c].words()[w] != cmf.masks[c].words()[w]) { mbad = true; break; }
        bool fbad = false;
        for (int t = 0; t < 6; ++t)
            if (gpuFaces[6 * c + t] != cmf.faces[6 * c + t]) { fbad = true; break; }
        if (mbad) ++maskMismatch;
        if (fbad) ++faceMismatch;
        if ((mbad || fbad) && shownMF++ < 10) {
            uint32_t li = 0; while (li < leafCount && gpuOffsets[li + 1] <= c) ++li;
            std::cerr << "  CC mask/face mismatch @ component " << c << " (leaf " << li
                      << ", local " << (c - gpuOffsets[li]) << ")"
                      << (mbad ? " [mask]" : "") << (fbad ? " [face]" : "") << "\n";
        }
    }

    std::cout << "CC per-component mask/face validation:  "
              << ((maskMismatch == 0 && faceMismatch == 0) ? "PASS" : "FAIL") << " ("
              << K << " components, " << maskMismatch << " mask mismatches, "
              << faceMismatch << " face mismatches)\n";

    // ---- Validate cross-leaf connectivity edges (pipeline step 5) ----
    // GPU emits edges in atomic-driven (nondeterministic) order, so compare as a SET: sort both
    // sides canonically and compare. Component offsets (gpuOffsets) double as the per-leaf slot
    // ranges the oracle needs.
    using Edge = nanovdb::tools::cuda::CrossLeafEdge;
    const uint64_t E = cc.crossLeafEdgeCount();
    std::vector<Edge> gpuEdges(E);
    if (E) cudaCheck(cudaMemcpy(gpuEdges.data(), cc.deviceCrossLeafEdges(),
                                std::size_t(E) * sizeof(Edge), cudaMemcpyDeviceToHost));

    std::vector<Edge> cpuEdges = cpuCrossLeafEdges(h_grid, cmf, gpuOffsets);

    auto edgeLess = [](const Edge& x, const Edge& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; };
    std::sort(gpuEdges.begin(), gpuEdges.end(), edgeLess);
    std::sort(cpuEdges.begin(), cpuEdges.end(), edgeLess);

    std::size_t edgeMismatch = 0;
    if (gpuEdges.size() != cpuEdges.size()) {
        edgeMismatch = (gpuEdges.size() > cpuEdges.size()) ? gpuEdges.size() - cpuEdges.size()
                                                           : cpuEdges.size() - gpuEdges.size();
        std::cerr << "  CC cross-leaf edge count mismatch: gpu E=" << gpuEdges.size()
                  << " cpu E=" << cpuEdges.size() << "\n";
    } else {
        std::size_t shownE = 0;
        for (std::size_t e = 0; e < gpuEdges.size(); ++e)
            if (gpuEdges[e].a != cpuEdges[e].a || gpuEdges[e].b != cpuEdges[e].b) {
                if (shownE++ < 10)
                    std::cerr << "  CC cross-leaf edge mismatch @ " << e
                              << ": gpu(" << gpuEdges[e].a << "," << gpuEdges[e].b << ")"
                              << " cpu(" << cpuEdges[e].a << "," << cpuEdges[e].b << ")\n";
                ++edgeMismatch;
            }
    }

    std::cout << "CC cross-leaf edge validation:          "
              << (edgeMismatch == 0 ? "PASS" : "FAIL") << " ("
              << E << " edges, " << edgeMismatch << " mismatches)\n";

    // ---- Validate global component labels (pipeline step 6) ----
    // Host union-find over the SAME edge list, same larger->smaller linking, so the flattened
    // representative (each class's minimum slot) matches the GPU elementwise. Path-halving in the
    // host find only shortens paths; it never changes the (minimum) root.
    std::vector<uint64_t> hostParent(K);
    for (uint64_t s = 0; s < K; ++s) hostParent[s] = s;
    auto hFind = [&](uint64_t x) {
        while (hostParent[x] != x) { hostParent[x] = hostParent[hostParent[x]]; x = hostParent[x]; }
        return x;
    };
    for (const Edge& e : gpuEdges) {
        const uint64_t ra = hFind(e.a), rb = hFind(e.b);
        if (ra == rb) continue;
        if (ra < rb) hostParent[rb] = ra; else hostParent[ra] = rb;  // larger root under smaller
    }
    for (uint64_t s = 0; s < K; ++s) hostParent[s] = hFind(s);       // flatten

    std::vector<uint64_t> gpuParent(K);
    cudaCheck(cudaMemcpy(gpuParent.data(), cc.deviceComponentParent(),
                         std::size_t(K) * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    std::size_t labelMismatch = 0, shownL = 0;
    uint64_t    distinct = 0;  // representatives (gpuParent[s]==s) = true global component count
    for (uint64_t s = 0; s < K; ++s) {
        if (gpuParent[s] == s) ++distinct;
        if (gpuParent[s] != hostParent[s]) {
            if (shownL++ < 10)
                std::cerr << "  CC label mismatch @ component " << s
                          << ": gpu=" << gpuParent[s] << " cpu=" << hostParent[s] << "\n";
            ++labelMismatch;
        }
    }

    std::cout << "CC component-label validation:          "
              << (labelMismatch == 0 ? "PASS" : "FAIL") << " ("
              << K << " components, " << distinct << " global labels, "
              << labelMismatch << " mismatches)\n";
    result.globalComponents = distinct;

    // ---- Validate non-barrier voxel signs (pipeline step 4) ----
    // Independently pick the exterior component (the one holding the global min-x active voxel),
    // sign every active voxel (+ exterior / - interior), and compare to the GPU per-voxel buffer.
    // Reuses the already-validated component offsets, masks, and parent array.
    const uint64_t activeCount = Traits::getActiveVoxelCount(d_grid);
    const auto* leaves = h_grid->tree().getFirstLeaf();

    // voxel -> its global component slot (scan the leaf's component masks)
    auto voxelSlot = [&](uint32_t li, uint32_t n) -> uint64_t {
        for (uint64_t s = gpuOffsets[li]; s < gpuOffsets[li + 1]; ++s)
            if (gpuMasks[s].isOn(n)) return s;
        return gpuOffsets[li];
    };

    // exterior representative = parent of the component holding the global minimum-x active voxel
    uint64_t exteriorRepCpu = 0; int32_t bestX = 0; bool found = false;
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (uint32_t n = 0; n < 512; ++n) {
            if (!leaf.isActive(n)) continue;
            const auto ijk = leaf.origin() + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
            if (!found || ijk[0] < bestX) { bestX = ijk[0]; exteriorRepCpu = gpuParent[voxelSlot(li, n)]; found = true; }
        }
    }

    // host per-voxel signs (slot 0 = background +1)
    std::vector<int8_t> cpuSign(activeCount + 1, int8_t(1));
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (uint32_t n = 0; n < 512; ++n) {
            if (!leaf.isActive(n)) continue;
            cpuSign[leaf.getValue(n)] = (gpuParent[voxelSlot(li, n)] == exteriorRepCpu) ? int8_t(1) : int8_t(-1);
        }
    }

    std::vector<int8_t> gpuSign(activeCount + 1);
    cudaCheck(cudaMemcpy(gpuSign.data(), sdf.deviceVoxelSign(),
                         std::size_t(activeCount + 1) * sizeof(int8_t), cudaMemcpyDeviceToHost));

    std::size_t signMismatch = 0; uint64_t nInterior = 0, nExterior = 0;
    for (uint64_t i = 1; i <= activeCount; ++i) {
        if (gpuSign[i] > 0) ++nExterior; else ++nInterior;
        if (gpuSign[i] != cpuSign[i]) ++signMismatch;
    }
    const bool signPass = (signMismatch == 0) && (sdf.exteriorRepresentative() == exteriorRepCpu);
    std::cout << "CC non-barrier sign validation:         " << (signPass ? "PASS" : "FAIL") << " ("
              << activeCount << " voxels, exteriorRep gpu=" << sdf.exteriorRepresentative()
              << " cpu=" << exteriorRepCpu << ", interior=" << nInterior << " exterior=" << nExterior
              << ", " << signMismatch << " mismatches)\n";

    // ---- Validate sign injection (derived -> original grid) ----
    // Every original active voxel must hold: the derived sign if it is non-barrier (present in the
    // derived grid), or the sentinel 0 if it is a barrier voxel (present only in the original).
    const auto*    d_orig     = origHandle.deviceGrid<BuildT>();
    const uint64_t origActive = Traits::getActiveVoxelCount(d_orig);

    const uint64_t origBytes = origHandle.bufferSize();
    void* origBlob = nullptr;
    cudaCheck(cudaMallocHost(&origBlob, origBytes));
    cudaCheck(cudaMemcpy(origBlob, origHandle.deviceData(), origBytes, cudaMemcpyDeviceToHost));
    const auto* h_orig = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(origBlob);

    std::vector<int8_t> gpuOrigSign(origActive + 1);
    cudaCheck(cudaMemcpy(gpuOrigSign.data(), sdf.deviceOriginalVoxelSign(),
                         std::size_t(origActive + 1) * sizeof(int8_t), cudaMemcpyDeviceToHost));

    std::size_t injMismatch = 0; uint64_t nBarrier = 0, nSigned = 0;
    const uint32_t origLeafCount = h_orig->tree().nodeCount(0);
    const auto*    origLeaves    = h_orig->tree().getFirstLeaf();
    for (uint32_t li = 0; li < origLeafCount; ++li) {
        const auto& oleaf = origLeaves[li];
        const auto* dleaf = h_grid->tree().root().probeLeaf(oleaf.origin());  // derived leaf, same origin
        for (uint32_t n = 0; n < 512; ++n) {
            if (!oleaf.isActive(n)) continue;
            int8_t expected;
            if (dleaf && dleaf->isActive(n)) { expected = gpuSign[dleaf->getValue(n)]; ++nSigned; }
            else                             { expected = int8_t(0);                   ++nBarrier; }
            if (gpuOrigSign[oleaf.getValue(n)] != expected) ++injMismatch;
        }
    }
    const bool injPass = (injMismatch == 0) && (gpuOrigSign[0] == int8_t(1));
    std::cout << "CC sign-injection validation:           " << (injPass ? "PASS" : "FAIL") << " ("
              << origActive << " orig voxels, signed(non-barrier)=" << nSigned
              << " barrier(sentinel)=" << nBarrier << ", " << injMismatch << " mismatches)\n";

    // ---- Validate barrier signing (pipeline step 5) ----
    // The post-injection signs (gpuOrigSign: +1/-1 non-barrier, 0 barrier) are the anchor snapshot.
    // Re-run the identical faithful-mirror logic on the host and check the GPU signed array; assert
    // no sentinel-0 remains (slot 0 stays +1).
    std::vector<int8_t> gpuSigned(origActive + 1);
    cudaCheck(cudaMemcpy(gpuSigned.data(), sdf.deviceSignedVoxelSign(),
                         std::size_t(origActive + 1) * sizeof(int8_t), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> hIndex(origActive + 1);
    cudaCheck(cudaMemcpy(hIndex.data(), static_cast<const uint32_t*>(indexSidecar.deviceData()),
                         std::size_t(origActive + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::size_t barrierCount = 0, residualZeros = 0, ambiguous = 0;
    const std::size_t barrierMism = cpuSignBarrier(h_orig, gpuOrigSign, gpuSigned, hIndex,
                                                   points, triangles, map, barrierCount, residualZeros, ambiguous);
    const bool barrierPass = (barrierMism == 0) && (residualZeros == 0) && (gpuSigned[0] == int8_t(1));
    std::cout << "CC barrier sign validation:             " << (barrierPass ? "PASS" : "FAIL") << " ("
              << barrierCount << " barrier voxels, " << barrierMism << " mismatches, "
              << ambiguous << " surface-tangent ties, " << residualZeros << " residual zeros)\n";

#ifdef NANOVDB_USE_OPENVDB
    // ---- Independent cross-check: our final signs vs OpenVDB meshToLevelSet (same mesh + transform) ----
    // Build an OpenVDB narrow-band level set from the identical points/triangles and voxel size, then
    // compare the SIGN of OpenVDB's value at each of our original grid's active index coords against our
    // final per-voxel sign. Conventions match (+ outside / - inside). meshToLevelSet signed-flood-fills,
    // so the sign is defined at every coord (band gives true distance; beyond it, ±background). Voxels
    // essentially on the surface (|value| within a tiny tie band) are reported separately, not counted.
    {
        openvdb::initialize();
        const double voxelSize = map.getVoxelSize()[0];
        openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform(voxelSize);

        std::vector<openvdb::Vec3s> ovPoints(points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
            ovPoints[i] = openvdb::Vec3s(points[i][0], points[i][1], points[i][2]);
        std::vector<openvdb::Vec3I> ovTris(triangles.size());
        for (std::size_t i = 0; i < triangles.size(); ++i)
            ovTris[i] = openvdb::Vec3I(uint32_t(triangles[i][0]), uint32_t(triangles[i][1]), uint32_t(triangles[i][2]));

        // Match our band width (exterior + interior); the signed flood fill makes signs valid everywhere.
        const float halfWidth = std::max(bandWidth, 3.0f);
        openvdb::FloatGrid::Ptr ls =
            openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*xform, ovPoints, ovTris, halfWidth);
        auto ovAcc = ls->getConstAccessor();

        // Sign is inherently method-dependent within the barrier shell (|distance| < √3/2 voxel — the
        // same half-diagonal our pipeline prunes & barrier-signs), so a disagreement there is a tie. A
        // disagreement OUTSIDE the shell, where OpenVDB confidently places the voxel inside/outside, is
        // a real error (e.g. a flipped component) and fails the check.
        const float vs       = float(voxelSize);
        const float shellVox = std::sqrt(0.75f);  // √3/2 ≈ 0.866 voxels

        // Optional Polyscope companion "<CC_EXPORT_VIS>.ovdb": one record per ORIGINAL-grid active
        // voxel, in the SAME leaf/voxel iteration order as the .ccvis dump written by exportMeshToSdf
        // (both walk getFirstLeaf() in leaf order over active voxels 0..511 of byte-identical grid
        // copies), so the viewer associates them positionally. Each record carries OpenVDB's raw
        // level-set value + sign at that voxel plus a precomputed mismatch class, so the viewer can
        // highlight exactly the disagreements this validator reports. Additive: no OpenVDB, no file.
        struct OvdbRec { float value; int32_t sign; int32_t mismatch; };  // mismatch: 0 agree, 1 beyond-shell, 2 in-shell
        const char*          ovdbVisPath = std::getenv("CC_EXPORT_VIS");
        std::vector<OvdbRec> ovdbRecs;
        if (ovdbVisPath) ovdbRecs.reserve(origActive);

        std::size_t realMismatch = 0, shellTie = 0, shownO = 0;
        uint64_t nOutside = 0, nInside = 0; float maxRealVox = 0.0f;
        for (uint32_t li = 0; li < origLeafCount; ++li) {
            const auto&          oleaf = origLeaves[li];
            const nanovdb::Coord o     = oleaf.origin();
            for (uint32_t n = 0; n < 512; ++n) {
                if (!oleaf.isActive(n)) continue;
                const nanovdb::Coord ijk = o + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
                const float  val     = ovAcc.getValue(openvdb::Coord(ijk[0], ijk[1], ijk[2]));
                const int8_t ourSign = gpuSigned[oleaf.getValue(n)];
                const int8_t ovSign  = (val < 0.0f) ? int8_t(-1) : int8_t(1);
                if (ovSign > 0) ++nOutside; else ++nInside;
                const float v = std::fabs(val) / vs;       // distance to surface, in voxels
                int32_t mismatch = 0;                       // 0 = agree
                if (ovSign != ourSign) {
                    if (v < shellVox) {                     // within barrier shell: method-dependent
                        mismatch = 2;
                        ++shellTie;
                    } else {                                // beyond shell: a real disagreement
                        mismatch = 1;
                        ++realMismatch;
                        if (v > maxRealVox) maxRealVox = v;
                        if (shownO++ < 10)
                            std::cerr << "  OpenVDB sign mismatch @ (" << ijk[0] << "," << ijk[1] << ","
                                      << ijk[2] << ") ours=" << int(ourSign) << " openvdb=" << int(ovSign)
                                      << " val=" << val << " (" << v << " vox)\n";
                    }
                }
                if (ovdbVisPath) ovdbRecs.push_back({ val, int32_t(ovSign), mismatch });
            }
        }
        std::cout << "OpenVDB sign cross-check:               " << (realMismatch == 0 ? "PASS" : "FAIL")
                  << " (" << origActive << " orig voxels, outside=" << nOutside << " inside=" << nInside
                  << "; " << realMismatch << " mismatches beyond shell";
        if (realMismatch) std::cout << " (max " << maxRealVox << " vox)";
        std::cout << ", " << shellTie << " in-shell sign ties)\n";
        result.openvdbChecked          = true;
        result.confidentSignMismatches = realMismatch;
        result.inShellTies             = shellTie;

        // Write the .ovdb companion (same directory/basename as the .ccvis dump).
        if (ovdbVisPath) {
            const std::string ovdbPath = std::string(ovdbVisPath) + ".ovdb";
            std::ofstream oout(ovdbPath, std::ios::binary);
            if (!oout) { std::cerr << "CC_EXPORT_VIS: cannot open " << ovdbPath << " for writing\n"; }
            else {
                // Layout: char magic[8]="CCOVDB01"; uint64 N; double voxelSize; N×{float value,int32 sign,int32 mismatch}.
                const uint64_t M = ovdbRecs.size();
                oout.write("CCOVDB01", 8);
                oout.write(reinterpret_cast<const char*>(&M), sizeof(M));
                oout.write(reinterpret_cast<const char*>(&voxelSize), sizeof(double));
                oout.write(reinterpret_cast<const char*>(ovdbRecs.data()), ovdbRecs.size() * sizeof(OvdbRec));
                std::cout << "CC_EXPORT_VIS: wrote " << M << " OpenVDB-comparison records to " << ovdbPath
                          << " (" << realMismatch << " beyond-shell mismatches, " << shellTie
                          << " in-shell ties)\n";
            }
        }
    }
#endif

    // ---- Optional independent ground truth: analytic signed distance to a UNION of spheres/boxes ----
    // A second, OpenVDB-free check (used by the in-code self-tests): a point is inside the union iff it
    // is inside ANY primitive, so the true signed distance is the min over primitives (+ outside /
    // - inside). Spheres use |p-C|-R; boxes use the Chebyshev distance max_i|p_i-C_i| - h, whose SIGN is
    // exact for a box (it underestimates the Euclidean distance outside near corners, which only widens
    // the tie band — the safe direction). Signs are compared in the confident region (|d| >= √3/2
    // voxel); the shell is method-dependent and only reported.
    const bool haveAnalytic = (analyticSpheres && numAnalyticSpheres > 0) ||
                              (analyticBoxes   && numAnalyticBoxes   > 0);
    const double vsWorld = map.getVoxelSize()[0];
    auto analyticDist = [&](double px, double py, double pz) -> double {
        double dist = HUGE_VAL;  // union = min over primitives
        for (int s = 0; s < numAnalyticSpheres; ++s) {
            const double Cx = analyticSpheres[4*s+0], Cy = analyticSpheres[4*s+1],
                         Cz = analyticSpheres[4*s+2], R  = analyticSpheres[4*s+3];
            const double di = std::sqrt((px-Cx)*(px-Cx) + (py-Cy)*(py-Cy) + (pz-Cz)*(pz-Cz)) - R;
            if (di < dist) dist = di;
        }
        for (int b = 0; b < numAnalyticBoxes; ++b) {
            const double ax = std::fabs(px - analyticBoxes[4*b+0]),
                         ay = std::fabs(py - analyticBoxes[4*b+1]),
                         az = std::fabs(pz - analyticBoxes[4*b+2]);
            const double di = std::max(ax, std::max(ay, az)) - analyticBoxes[4*b+3];
            if (di < dist) dist = di;
        }
        return dist;
    };

    if (haveAnalytic) {
        const double vs         = vsWorld;
        const double shellWorld = std::sqrt(0.75) * vs;
        std::size_t aMis = 0, aTie = 0, shownA = 0; double maxAVox = 0.0;
        for (uint32_t li = 0; li < origLeafCount; ++li) {
            const auto&          oleaf = origLeaves[li];
            const nanovdb::Coord o     = oleaf.origin();
            for (uint32_t n = 0; n < 512; ++n) {
                if (!oleaf.isActive(n)) continue;
                const nanovdb::Coord ijk = o + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
                const double dist = analyticDist(double(ijk[0]) * vs, double(ijk[1]) * vs, double(ijk[2]) * vs);
                const int8_t truth   = (dist < 0.0) ? int8_t(-1) : int8_t(1);
                const int8_t ourSign = gpuSigned[oleaf.getValue(n)];
                if (truth == ourSign) continue;
                if (std::fabs(dist) < shellWorld) { ++aTie; continue; }
                ++aMis; const double v = std::fabs(dist) / vs; if (v > maxAVox) maxAVox = v;
                if (shownA++ < 10)
                    std::cerr << "  analytic sign mismatch @ (" << ijk[0] << "," << ijk[1] << "," << ijk[2]
                              << ") ours=" << int(ourSign) << " truth=" << int(truth)
                              << " d=" << dist << " (" << v << " vox)\n";
            }
        }
        result.analyticChecked             = true;
        result.analyticConfidentMismatches = aMis;
        result.analyticInShellTies         = aTie;
        std::cout << "Analytic union sign check:              " << (aMis == 0 ? "PASS" : "FAIL")
                  << " (" << origActive << " orig voxels, " << aMis << " mismatches beyond shell";
        if (aMis) std::cout << " (max " << maxAVox << " vox)";
        std::cout << ", " << aTie << " in-shell ties)\n";
    }

    // ---- Validate the leaf invert mask (pipeline step 6, chunk A) ----
    // The invert mask signs INACTIVE voxels inside materialized leaves (bit ON = interior). Inactive
    // voxels sit beyond the narrow band (> bandWidth voxels from the surface), so the analytic union
    // ground truth is unambiguous — no shell tie band needed: assert bit == (union dist < 0) for every
    // inactive voxel of every materialized leaf. Deep interior beyond materialized leaves is a later
    // chunk and is NOT checked here. Without analytic geometry, only the ON-bit count is reported.
    {
        std::vector<nanovdb::Mask<3>> invMasks(origLeafCount);
        cudaCheck(cudaMemcpy(invMasks.data(), sdf.deviceLeafInvertMask(),
                             std::size_t(origLeafCount) * sizeof(nanovdb::Mask<3>), cudaMemcpyDeviceToHost));

        const double vs = vsWorld;
        uint64_t nInactive = 0, nOn = 0, activeOnBits = 0;
        std::size_t invMis = 0, shownI = 0;
        for (uint32_t li = 0; li < origLeafCount; ++li) {
            const auto&          oleaf = origLeaves[li];
            const nanovdb::Coord o     = oleaf.origin();
            for (uint32_t n = 0; n < 512; ++n) {
                const bool on = invMasks[li].isOn(n);
                if (oleaf.isActive(n)) { if (on) ++activeOnBits; continue; }  // active bits must stay OFF
                ++nInactive;
                if (on) ++nOn;
                if (!haveAnalytic) continue;
                const nanovdb::Coord ijk = o + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
                const double dist = analyticDist(double(ijk[0]) * vs, double(ijk[1]) * vs, double(ijk[2]) * vs);
                const bool truth = dist < 0.0;  // analytic interior
                if (on != truth) {
                    ++invMis;
                    if (shownI++ < 10)
                        std::cerr << "  invert-mask mismatch @ (" << ijk[0] << "," << ijk[1] << "," << ijk[2]
                                  << ") bit=" << int(on) << " truth=" << int(truth)
                                  << " d=" << dist << " (" << dist / vs << " vox)\n";
                }
            }
        }

        result.invertOnBits = nOn;
        if (haveAnalytic) {
            result.invertChecked    = true;
            result.invertMismatches = invMis + activeOnBits;  // stray active bits are also defects
            std::cout << "Leaf invert-mask validation:            "
                      << ((invMis == 0 && activeOnBits == 0) ? "PASS" : "FAIL") << " ("
                      << nInactive << " inactive voxels in " << origLeafCount << " leaves, "
                      << nOn << " marked interior, " << invMis << " mismatches, "
                      << activeOnBits << " stray active bits)\n";
        } else {
            std::cout << "Leaf invert-mask report:                " << nOn << " / " << nInactive
                      << " inactive voxels marked interior (" << origLeafCount << " leaves"
                      << (activeOnBits ? ", " + std::to_string(activeOnBits) + " STRAY ACTIVE BITS" : "")
                      << ")\n";
        }
    }

    // ---- Validate the coarse (lower/upper) invert masks (pipeline step 6, chunk B) ----
    // A childless tile is uniform-sign (a surface crossing it would force refinement), so its bit must
    // equal the analytic sign at the TILE CENTER: bit == (analyticDist(center) < 0). Checked for every
    // childless child slot of every materialized lower (8^3-voxel tiles) and upper (128^3-voxel tiles)
    // node; refined slots carry no bit. Stray bits on refined slots are also defects. Root-level tiles
    // are chunk C. Without analytic geometry, only ON-tile counts are reported.
    {
        const uint32_t lowerCount = h_orig->tree().nodeCount(1);
        const uint32_t upperCount = h_orig->tree().nodeCount(2);
        std::vector<nanovdb::Mask<4>> lowInv(lowerCount);
        std::vector<nanovdb::Mask<5>> upInv(upperCount);
        if (lowerCount) cudaCheck(cudaMemcpy(lowInv.data(), sdf.deviceLowerInvertMask(),
                                  std::size_t(lowerCount) * sizeof(nanovdb::Mask<4>), cudaMemcpyDeviceToHost));
        if (upperCount) cudaCheck(cudaMemcpy(upInv.data(), sdf.deviceUpperInvertMask(),
                                  std::size_t(upperCount) * sizeof(nanovdb::Mask<5>), cudaMemcpyDeviceToHost));

        uint64_t lChildless = 0, lOn = 0, uChildless = 0, uOn = 0, strayRefined = 0;
        std::size_t coarseMis = 0, shownC = 0;
        // One pass per level; the two levels differ only in node type / slot count / tile dim.
        auto checkLevel = [&](const auto* nodes, uint32_t count, const auto* inv, int slots,
                              int tileDim, uint64_t& childless, uint64_t& onCount, const char* lvl) {
            for (uint32_t ni = 0; ni < count; ++ni) {
                const auto& node = nodes[ni];
                for (int n = 0; n < slots; ++n) {
                    const bool on = inv[ni].isOn(uint32_t(n));
                    if (node.childMask().isOn(uint32_t(n))) { if (on) ++strayRefined; continue; }
                    ++childless;
                    if (on) ++onCount;
                    if (!haveAnalytic) continue;
                    const nanovdb::Coord g = node.offsetToGlobalCoord(uint32_t(n));  // tile origin (voxels)
                    const double half = 0.5 * double(tileDim - 1);
                    const double dist = analyticDist((double(g[0]) + half) * vsWorld,
                                                     (double(g[1]) + half) * vsWorld,
                                                     (double(g[2]) + half) * vsWorld);
                    const bool truth = dist < 0.0;
                    if (on != truth) {
                        ++coarseMis;
                        if (shownC++ < 10)
                            std::cerr << "  coarse invert mismatch @ " << lvl << " tile (" << g[0] << ","
                                      << g[1] << "," << g[2] << ")+" << tileDim << ": bit=" << int(on)
                                      << " truth=" << int(truth) << " d=" << dist / vsWorld << " vox\n";
                    }
                }
            }
        };
        checkLevel(h_orig->tree().template getFirstNode<1>(), lowerCount, lowInv.data(), 4096,   8, lChildless, lOn, "lower");
        checkLevel(h_orig->tree().template getFirstNode<2>(), upperCount, upInv.data(), 32768, 128, uChildless, uOn, "upper");

        result.lowerOnTiles = lOn;
        result.upperOnTiles = uOn;
        if (haveAnalytic) {
            result.coarseInvertChecked    = true;
            result.coarseInvertMismatches = coarseMis + strayRefined;
            std::cout << "Coarse invert-mask validation:          "
                      << ((coarseMis == 0 && strayRefined == 0) ? "PASS" : "FAIL") << " ("
                      << lChildless << " lower + " << uChildless << " upper childless tiles, "
                      << lOn << "+" << uOn << " marked interior, " << coarseMis << " mismatches, "
                      << strayRefined << " stray refined bits)\n";
        } else {
            std::cout << "Coarse invert-mask report:              lower " << lOn << " / " << lChildless
                      << ", upper " << uOn << " / " << uChildless << " childless tiles marked interior ("
                      << lowerCount << "+" << upperCount << " nodes"
                      << (strayRefined ? ", " + std::to_string(strayRefined) + " STRAY REFINED BITS" : "")
                      << ")\n";
        }
    }

    // ---- Chunk C: root-interior sidecar + FULL-DOMAIN sign query validation ----
    // signedSignAt composes the whole level set: active signs + leaf/lower/upper invert bits + the
    // root-interior sidecar. With analytic geometry, sample a coarse 3D lattice spanning deep interior,
    // band, far exterior AND beyond the root-cell array (out-of-range => exterior), and compare every
    // queried sign against the analytic one; points within a thin on-surface band are ties (a sampled
    // coord there may be a legitimately ±-signed surface voxel).
    {
        const nanovdb::Coord rMin  = sdf.rootTileMin();
        const nanovdb::Coord rDims = sdf.rootTileDims();
        const uint64_t rTotal = uint64_t(rDims[0]) * uint64_t(rDims[1]) * uint64_t(rDims[2]);
        std::vector<uint8_t> rootOn(rTotal);
        if (rTotal) cudaCheck(cudaMemcpy(rootOn.data(), sdf.deviceRootInterior(), rTotal, cudaMemcpyDeviceToHost));
        uint64_t rootInterior = 0;
        for (uint64_t c = 0; c < rTotal; ++c) rootInterior += rootOn[c];
        result.rootInteriorCells = rootInterior;
        std::cout << "Root-interior sidecar:                  " << rDims[0] << "x" << rDims[1] << "x"
                  << rDims[2] << " cells @ tileMin (" << rMin[0] << "," << rMin[1] << "," << rMin[2]
                  << "), " << rootInterior << " deep-interior\n";

        if (haveAnalytic) {
            // Re-download the invert sidecars (the chunk-A/B validation copies were scoped).
            const uint32_t lowerCount = h_orig->tree().nodeCount(1);
            const uint32_t upperCount = h_orig->tree().nodeCount(2);
            std::vector<nanovdb::Mask<3>> leafInv(origLeafCount);
            std::vector<nanovdb::Mask<4>> lowInv(lowerCount);
            std::vector<nanovdb::Mask<5>> upInv(upperCount);
            cudaCheck(cudaMemcpy(leafInv.data(), sdf.deviceLeafInvertMask(),
                                 std::size_t(origLeafCount) * sizeof(nanovdb::Mask<3>), cudaMemcpyDeviceToHost));
            if (lowerCount) cudaCheck(cudaMemcpy(lowInv.data(), sdf.deviceLowerInvertMask(),
                                 std::size_t(lowerCount) * sizeof(nanovdb::Mask<4>), cudaMemcpyDeviceToHost));
            if (upperCount) cudaCheck(cudaMemcpy(upInv.data(), sdf.deviceUpperInvertMask(),
                                 std::size_t(upperCount) * sizeof(nanovdb::Mask<5>), cudaMemcpyDeviceToHost));

            // Sample lattice: grid bbox expanded by 4200 voxels (crosses into absent root regions and
            // beyond the root-cell array on every side), ~41 samples per axis.
            const auto& hbbox = h_orig->indexBBox();
            const int margin = 4200, steps = 41;
            uint64_t nSampled = 0, ties = 0; std::size_t fdMis = 0, shownF = 0;
            for (int si = 0; si < steps; ++si)
                for (int sj = 0; sj < steps; ++sj)
                    for (int sk = 0; sk < steps; ++sk) {
                        const int lo[3] = { hbbox.min()[0] - margin, hbbox.min()[1] - margin, hbbox.min()[2] - margin };
                        const int hi[3] = { hbbox.max()[0] + margin, hbbox.max()[1] + margin, hbbox.max()[2] + margin };
                        const nanovdb::Coord ijk(lo[0] + int(int64_t(hi[0] - lo[0]) * si / (steps - 1)),
                                                 lo[1] + int(int64_t(hi[1] - lo[1]) * sj / (steps - 1)),
                                                 lo[2] + int(int64_t(hi[2] - lo[2]) * sk / (steps - 1)));
                        ++nSampled;
                        const double dist = analyticDist(double(ijk[0]) * vsWorld, double(ijk[1]) * vsWorld,
                                                         double(ijk[2]) * vsWorld);
                        if (std::fabs(dist) < std::sqrt(0.75) * vsWorld) { ++ties; continue; }
                        const int8_t truth = (dist < 0.0) ? int8_t(-1) : int8_t(1);
                        const int8_t ours  = nanovdb::tools::cuda::sdf_detail::signedSignAt(
                            *h_orig, ijk, gpuSigned.data(), leafInv.data(), lowInv.data(), upInv.data(),
                            rootOn.data(), rMin, rDims);
                        if (ours != truth) {
                            ++fdMis;
                            if (shownF++ < 10)
                                std::cerr << "  full-domain mismatch @ (" << ijk[0] << "," << ijk[1] << ","
                                          << ijk[2] << ") ours=" << int(ours) << " truth=" << int(truth)
                                          << " d=" << dist / vsWorld << " vox\n";
                        }
                    }
            result.fullDomainChecked    = true;
            result.fullDomainMismatches = fdMis;
            result.fullDomainTies       = ties;
            std::cout << "Full-domain sign query validation:      " << (fdMis == 0 ? "PASS" : "FAIL")
                      << " (" << nSampled << " sampled coords, " << fdMis << " mismatches, "
                      << ties << " on-surface ties)\n";
        }
    }

    cudaCheck(cudaFreeHost(origBlob));

    cudaCheck(cudaFreeHost(hostBlob));
    return result;
}

/// @brief Synthetic unit test of the chunk-C root-interior flood (RootInteriorFloodFunctor). An
///        interior ABSENT root cell needs an object >4096 voxels thick — no practically-rasterizable
///        mesh reaches that — so the seed gate / multi-seed / wall-blocking logic is exercised here
///        with fabricated inputs instead. Layout: 7×5×5 cells, full wall planes at i=2 and i=4 carve
///        three disconnected non-wall regions A={i:0,1}, B={i:3}, C={i:5,6}:
///          - A is seeded interior at one cell           -> the flood must fill ALL of A;
///          - B is seeded interior at one cell           -> fills all of B (multi-seed: disconnected);
///          - C gets MIXED evidence (sawInt+sawExt)      -> gate rejects; C stays exterior;
///          - walls must stay OFF and block A/B from C.
/// @return number of cells whose final state differs from the expectation (0 = PASS).
int testRootInteriorFlood()
{
    const nanovdb::Coord dims(7, 5, 5);
    const int P = dims[0], Q = dims[1], R = dims[2], total = P * Q * R;
    auto idx = [&](int i, int j, int k) { return (i * Q + j) * R + k; };

    std::vector<uint8_t> wall(total, 0), sawInt(total, 0), sawExt(total, 0);
    for (int j = 0; j < Q; ++j)
        for (int k = 0; k < R; ++k) { wall[idx(2, j, k)] = 1; wall[idx(4, j, k)] = 1; }
    sawInt[idx(0, 2, 2)] = 1;                             // region A: one interior seed
    sawInt[idx(3, 2, 2)] = 1;                             // region B: one interior seed (disconnected)
    sawInt[idx(5, 2, 2)] = 1; sawExt[idx(5, 2, 2)] = 1;   // region C: mixed evidence -> must stay OFF

    thrust::universal_vector<uint8_t> dWall(wall.begin(), wall.end());
    thrust::universal_vector<uint8_t> dSawInt(sawInt.begin(), sawInt.end());
    thrust::universal_vector<uint8_t> dSawExt(sawExt.begin(), sawExt.end());
    thrust::universal_vector<uint8_t> dOn(std::size_t(total), uint8_t(0));

    using FloodOp = nanovdb::tools::cuda::sdf_detail::RootInteriorFloodFunctor;
    nanovdb::util::cuda::operatorKernel<FloodOp><<<1, FloodOp::MaxThreadsPerBlock>>>(
        dWall.data().get(), dSawInt.data().get(), dSawExt.data().get(), dOn.data().get(), dims);
    cudaCheckError();
    cudaCheck(cudaDeviceSynchronize());

    int mism = 0;
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < Q; ++j)
            for (int k = 0; k < R; ++k) {
                const bool expect = (i == 0 || i == 1 || i == 3);  // A and B fill; walls and C stay OFF
                if (bool(dOn[idx(i, j, k)]) != expect) {
                    if (mism < 10)
                        std::cerr << "  root-flood unit mismatch @ (" << i << "," << j << "," << k
                                  << "): on=" << int(dOn[idx(i, j, k)]) << " expect=" << int(expect) << "\n";
                    ++mism;
                }
            }
    std::cout << "Root-interior flood unit test:          " << (mism == 0 ? "PASS" : "FAIL")
              << " (7x5x5 cells, 2 wall planes, 3 regions, " << mism << " mismatches)\n";
    return mism;
}
