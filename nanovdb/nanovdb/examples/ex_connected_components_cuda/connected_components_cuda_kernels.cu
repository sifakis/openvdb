// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda_kernels.cu
///
/// @brief CUDA / NanoVDB side of the connected-components example (no OpenVDB).
///
///        computeUDF(): uploads the triangle mesh and voxelizes it into a narrow-band
///        ValueOnIndex grid plus a per-active-voxel unsigned-distance-field (UDF)
///        sidecar, via nanovdb::tools::cuda::MeshToGrid.
///
///        The connected-components labeling (a union-find with hierarchical insights)
///        and the derivative-grid step it operates on will be added here as follow-ups.
///        See MeshToSDFDevelopmentPlan.md in this directory for the design notes and roadmap.

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/MeshToSDF.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Util.h>

#include <thrust/universal_vector.h>

#include <algorithm>  // std::sort
#include <cmath>      // std::sqrt, std::fabs
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Must match the aliases in connected_components_cuda.cpp.
using GridHandleT   = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using UDFSidecarT   = nanovdb::cuda::DeviceBuffer;
using IndexSidecarT = nanovdb::cuda::DeviceBuffer;

std::pair<GridHandleT, UDFSidecarT> computeUDF(
    const std::vector<nanovdb::Vec3f>& points,
    const std::vector<nanovdb::Vec3i>& triangles,
    const nanovdb::Map&                map,
    float                              bandWidth)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Upload mesh to the device (managed memory keeps the example simple).
    thrust::universal_vector<nanovdb::Vec3f> dPoints(points.begin(), points.end());
    thrust::universal_vector<nanovdb::Vec3i> dTriangles(triangles.begin(), triangles.end());

    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(
        dPoints.data().get(),    uint32_t(dPoints.size()),
        dTriangles.data().get(), uint32_t(dTriangles.size()),
        map);
    converter.setVerbose(1);
    converter.setNarrowBandWidth(bandWidth);

    // { index-grid handle, UDF sidecar }, both backed by DeviceBuffer.
    return converter.getHandleAndUDF();
}

/// @brief Step 1 (additive sibling of computeUDF): rasterize the mesh into the index grid + UDF
///        sidecar AND a per-active-voxel NEAREST-TRIANGLE-INDEX sidecar (uint32), needed by step 5
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

} // anonymous namespace

void computeCC(const GridHandleT& origHandle, const GridHandleT& derivedHandle)
{
    using BuildT = nanovdb::ValueOnIndex;
    using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

    const auto* d_grid = derivedHandle.deviceGrid<BuildT>();

    // Pure connected-components labeling (domain-agnostic): per-leaf CC -> cross-leaf edges ->
    // global union-find. Steps 3/5/6 of the pipeline.
    nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(d_grid);
    cc.setVerbose(1);
    cc.processLeafConnectedComponents();
    cudaCheck(cudaDeviceSynchronize());
    cc.processCrossLeafEdges();
    cudaCheck(cudaDeviceSynchronize());
    cc.processComponentLabels();
    cudaCheck(cudaDeviceSynchronize());

    // Step 4 (SDF domain): sign the non-barrier voxels (+ exterior / - interior) from the labeling.
    nanovdb::tools::cuda::MeshToSDF<BuildT> sdf;
    sdf.setVerbose(1);
    sdf.signNonBarrier(d_grid, cc);
    cudaCheck(cudaDeviceSynchronize());

    // Injection: carry the derived (non-barrier) signs back onto the original grid; barrier voxels
    // (present only in the original) stay sentinel 0 for the future step-5 barrier signing.
    sdf.injectSignsToOriginal(origHandle.deviceGrid<BuildT>(), d_grid);
    cudaCheck(cudaDeviceSynchronize());

    const uint32_t leafCount = Traits::getTreeData(d_grid).mNodeCount[0];
    if (leafCount == 0) { std::cout << "CC validation: empty grid, nothing to check\n"; return; }

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
    cudaCheck(cudaFreeHost(origBlob));

    cudaCheck(cudaFreeHost(hostBlob));
}
